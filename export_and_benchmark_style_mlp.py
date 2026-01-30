"""
StyleMLP Benchmark & Export Suite
==================================
Version: 1.5.1
Date: January 30, 2026

Evaluates the StyleMLP classifier across PyTorch, ONNX, and OpenVINO.
Optimized for high-throughput consumer device profiling.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import onnxruntime as ort

# Attempt OpenVINO import
try:
    import openvino.runtime as ov
    OV_AVAILABLE = True
except ImportError:
    OV_AVAILABLE = False

# --- CONFIGURATION ---
FEATURES_TEST_PATH = 'features_test.pt'
STYLE_MLP_WEIGHTS  = 'style_model_v3_modified.pth'
ONNX_PATH          = 'style_mlp.onnx'
OPENVINO_XML_PATH  = 'style_mlp.xml'
LOG_FILE           = 'style_mlp_benchmark_results.txt'

WARMUP_ITERS = 50
BENCHMARK_ITERS = 200

def log_print(message: str):
    print(message)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(message + '\n')

# --- STYLEMLP ARCHITECTURE ---
class StyleMLP(nn.Module):
    def __init__(self, input_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
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

# --- EVALUATION LOGIC ---
def run_benchmark():
    log_print(f"--- Starting StyleMLP Benchmark Session ---")
    
    # 1. Load Test Data
    data = torch.load(FEATURES_TEST_PATH, map_location='cpu')
    X_test = data['X'].numpy().astype(np.float32)
    y_test = data['y'].numpy().astype(int)
    
    # 2. PyTorch Reference
    model_pt = StyleMLP()
    model_pt.load_state_dict(torch.load(STYLE_MLP_WEIGHTS, map_location='cpu'))
    model_pt.eval()
    
    with torch.no_grad():
        start_pt = time.perf_counter()
        probs_pt = model_pt(torch.from_numpy(X_test)).numpy().flatten()
        latency_pt = (time.perf_counter() - start_pt) / len(X_test) * 1000
    
    preds_pt = (probs_pt > 0.5).astype(int)
    log_print(f"[PyTorch] Accuracy: {accuracy_score(y_test, preds_pt):.4f} | Latency: {latency_pt:.4f}ms")

    # 3. ONNX Export & Benchmark
    dummy_in = torch.randn(1, 256)
    torch.onnx.export(model_pt, dummy_in, ONNX_PATH, opset_version=17, 
                      input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}})
    
    ort_sess = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
    
    start_onnx = time.perf_counter()
    probs_onnx = ort_sess.run(None, {'input': X_test})[0].flatten()
    latency_onnx = (time.perf_counter() - start_onnx) / len(X_test) * 1000
    log_print(f"[ONNX]    Accuracy: {accuracy_score(y_test, (probs_onnx > 0.5)):.4f} | Latency: {latency_onnx:.4f}ms")

    # 4. OpenVINO (if available)
    if OV_AVAILABLE and os.path.exists(OPENVINO_XML_PATH):
        core = ov.Core()
        ov_model = core.compile_model(OPENVINO_XML_PATH, 'CPU')
        
        start_ov = time.perf_counter()
        res_ov = ov_model([X_test])[ov_model.output(0)]
        latency_ov = (time.perf_counter() - start_ov) / len(X_test) * 1000
        log_print(f"[OpenVINO] Accuracy: {accuracy_score(y_test, (res_ov.flatten() > 0.5)):.4f} | Latency: {latency_ov:.4f}ms")

if __name__ == "__main__":
    run_benchmark()