"""
YOLOv11 Grad-CAM Diagnostic Visualizer
======================================
Version: 7.0
Date: January 30, 2026

This diagnostic utility implements Gradient-weighted Class Activation Mapping 
(Grad-CAM) for YOLOv11. It visualizes model attention by capturing activations 
and gradients from a target convolutional layer (default: 'model.22') to 
produce interpretability heatmaps.
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO



class YOLO11nGradCAM:
    def __init__(self, model_path, layer_name='model.22'):
        """
        Initializes the Grad-CAM visualizer.
        :param model_path: Path to the .pt YOLOv11 weights.
        :param layer_name: Final convolutional layer before detection heads.
        """
        self.yolo = YOLO(model_path)
        self.model = self.yolo.model.eval()  # Set to evaluation mode
        self.layer_name = layer_name
        self.activations = None
        self.gradients = None
        
        # Locate the target layer within the PyTorch model object
        target_layer = dict(self.model.named_modules()).get(layer_name)
        if target_layer is None:
            raise ValueError(f"Layer {layer_name} not found in model.")
            
        # Register a Forward Hook: Captures data during the forward pass
        target_layer.register_forward_hook(self._save_act)

    def _save_act(self, module, input, output):
        """Hook callback: Saves activations and registers a hook for gradients."""
        self.activations = output
        # Capture gradients during the .backward() call
        output.register_hook(self._save_grad)

    def _save_grad(self, grad):
        """Hook callback: Saves gradients during backprop."""
        self.gradients = grad

    def generate_heatmap(self, img_path):
        """
        Processes an image and returns a Grad-CAM heatmap overlay.
        :param img_path: Path to the input image.
        :return: BGR Image (numpy array) with heatmap overlay.
        """
        # --- PRE-PROCESSING ---
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image at {img_path}")
            
        h_orig, w_orig = img.shape[:2]
        img_resized = cv2.resize(img, (640, 640))
        
        # Prepare tensor: [H, W, C] -> [1, C, H, W]
        input_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        
        # --- INFERENCE & BACKPROP ---
        torch.set_grad_enabled(True) 
        input_tensor.requires_grad_(True)
        
        self.model.zero_grad()
        preds = self.model(input_tensor)
        
        # Find the anchor/cell with the highest confidence score
        conf_scores = preds[0][0, 4:, :].max(dim=0)[0] 
        best_anchor = torch.argmax(conf_scores)
        
        # Backward pass on the specific detection score
        score = preds[0][0, :, best_anchor].sum()
        score.backward() 
        
        # --- GRAD-CAM MATHEMATICS ---
        # 1. Extract saved tensors
        grads = self.gradients.data.cpu().numpy()[0]
        acts = self.activations.data.cpu().numpy()[0]
        
        # 2. Global Average Pooling (GAP) of gradients to find channel 'Weights'
        weights = np.mean(grads, axis=(1, 2))
        
        # 3. Weighted combination of activation maps
        cam = np.zeros(acts.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * acts[i]

        # 4. ReLU: Only keep features that POSITIVELY influence the score
        cam = np.maximum(cam, 0)
        
        # --- POST-PROCESSING ---
        # De-noising: remove low-intensity signals
        cam[cam < (cam.max() * 0.2)] = 0
        cam = cv2.resize(cam, (w_orig, h_orig))
        
        if cam.max() > 0:
            cam = cam / cam.max()
            
        # Create Jet Heatmap (Blue = Low importance, Red = High importance)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        
        # Alpha Blending for visualization
        result = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
        return result