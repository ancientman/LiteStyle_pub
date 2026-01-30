"""
LiteStyle Unified Garment & Formality Detector
==============================================
Version: 2.6 (Manuscript Safe)
Date: January 30, 2026

Architecture: YOLOv11n (Frozen) + StyleMLP Head
This script handles the fusion of object detection and classification 
into a single ONNX graph with support for INT8 dynamic quantization.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO

# Robust ONNX Quantization Check
try:
    from onnxruntime.quantization import quantize_dynamic, QuantType
    QUANT_AVAILABLE = True
except ImportError:
    QUANT_AVAILABLE = False
    print("[WARNING] onnxruntime-tools not found. INT8 export will be skipped.")

# --- 1. CONFIGURATION ---
YOLO_WEIGHTS    = 'best.pt'
STYLE_WEIGHTS   = 'style_model_v3.pth'
UNIFIED_ONNX    = 'lite_style_combined.onnx'
INT8_ONNX       = 'lite_style_int8.onnx'
IMG_SZ          = 640

# --- 2. STYLEMLP ARCHITECTURE ---
class StyleMLP(nn.Module):
    """Classification head for attire formality scoring (0: Casual, 1: Formal)"""
    def __init__(self, in_channels: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# --- 3. UNIFIED MODEL WRAPPER ---
class UnifiedGarmentFormalityModel(nn.Module):
    def __init__(self, yolo_weights=YOLO_WEIGHTS, style_weights=STYLE_WEIGHTS, sppf_idx=9):
        super().__init__()
        
        # Load and Freeze YOLO Backbone
        print("[INFO] Initializing YOLOv11n backbone...")
        self.yolo_model = YOLO(yolo_weights).model
        self.yolo_model.eval()
        for p in self.yolo_model.parameters():
            p.requires_grad = False

        self.sppf_idx = sppf_idx
        self.sppf_layer = self.yolo_model.model[sppf_idx]
        
        # Determine channels dynamically
        dummy_input = torch.zeros(1, 3, IMG_SZ, IMG_SZ)
        with torch.no_grad():
            # Extract features up to SPPF to find channel count
            feat_out = dummy_input
            for i in range(sppf_idx + 1):
                feat_out = self.yolo_model.model[i](feat_out)
            in_ch = feat_out.shape[1]
            
            # Determine how many detection heads YOLO has
            det_out = self.yolo_model(dummy_input)
            self.num_det_outputs = len(det_out) if isinstance(det_out, (list, tuple)) else 1

        self.style_head = StyleMLP(in_channels=in_ch)
        self._load_weights(style_weights)
        
        # Internal state for feature hooking
        self._features = None
        self._hook = self.sppf_layer.register_forward_hook(self._save_features)

    def _save_features(self, mod, inp, out):
        self._features = out

    def _load_weights(self, path):
        if os.path.exists(path):
            print(f"[INFO] Loading StyleMLP weights: {path}")
            state = torch.load(path, map_location='cpu')
            # Handle potential DataParallel or different prefixing
            clean_state = {k.replace('net.', ''): v for k, v in state.items()}
            self.style_head.net.load_state_dict(clean_state, strict=False)
        else:
            print("[WARNING] StyleMLP weights not found. Using random init.")

    def forward(self, x):
        # Run YOLO backbone (triggers hook)
        det_out = self.yolo_model(x)
        
        # Global Average Pooling on SPPF features
        pooled = F.adaptive_avg_pool2d(self._features, (1, 1)).flatten(1)
        formality_score = self.style_head(pooled)
        
        return det_out, formality_score

# --- 4. EXPORT PIPELINE ---
def run_export():
    model = UnifiedGarmentFormalityModel().eval()
    dummy = torch.randn(1, 3, IMG_SZ, IMG_SZ)
    
    # 1. Export PyTorch
    torch.save(model.state_dict(), "combined_model.pt")
    
    # 2. Export ONNX
    output_names = [f"det_{i}" for i in range(model.num_det_outputs)] + ["formality_score"]
    
    torch.onnx.export(
        model, dummy, UNIFIED_ONNX,
        opset_version=17,
        input_names=["input_tensor"],
        output_names=output_names,
        dynamic_axes={"input_tensor": {0: "batch"}, "formality_score": {0: "batch"}},
        do_constant_folding=True
    )
    print(f"[SUCCESS] Exported FP32 ONNX: {UNIFIED_ONNX}")

    # 3. Dynamic INT8 Quantization
    if QUANT_AVAILABLE:
        quantize_dynamic(
            model_input=UNIFIED_ONNX,
            model_output=INT8_ONNX,
            weight_type=QuantType.QInt8
        )
        print(f"[SUCCESS] Exported INT8 ONNX: {INT8_ONNX}")

if __name__ == "__main__":
    run_export()