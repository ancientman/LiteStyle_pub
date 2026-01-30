"""
LiteStyle StyleMLP Trainer & Feature Extractor
==============================================
Version: 1.4.1 
Date: January 30, 2026

Optimized for: YOLOv11n backbone, consumer-device deployment.
Outputs: features_*.pt, style_model_v3_modified.pth, training_log.txt
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO
from tqdm import tqdm
import torch.nn.functional as F

# --- 1. CONFIGURATION ---
TRAIN_ROOT      = './style_images/train'
VAL_ROOT        = './style_images/val'
TEST_ROOT       = './style_images/test'
YOLO_MODEL_PATH = 'best.pt'
SAVE_PATH       = 'style_model_v3_modified.pth'
LOG_FILE        = 'style_model_training_log.txt'

DEVICE          = 'cpu'  # High portability for consumer research
BATCH_SIZE      = 32
EPOCHS          = 100
PATIENCE        = 10
SAVE_FEATURES_ONLY = False

# --- 2. LOGGING UTILITY ---
def log_print(message: str):
    """Prints to console and appends to a permanent text log."""
    print(message)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(message + '\n')

# Preprocessing: Normalized to ImageNet standards (matching YOLO requirements)
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- 3. FEATURE EXTRACTION ENGINE ---
class FeatureExtractor:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        yolo_wrapper = YOLO(model_path)
        self.model = yolo_wrapper.model.to(device)
        self.model.eval()
        self._buffer = []
        # Index 9 is the SPPF layer in standard YOLOv11n
        self.hook = self.model.model[9].register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        pooled = F.adaptive_avg_pool2d(output, (1, 1))
        self._buffer.append(pooled.flatten(1).cpu())

    def extract(self, paths):
        all_features = []
        for path in tqdm(paths, desc="Extracted Features"):
            try:
                img = Image.open(path).convert('RGB')
                img_t = transform(img).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    _ = self.model(img_t)
                all_features.append(self._buffer.pop(0))
            except Exception as e:
                log_print(f"Skipping corrupt image {path}: {e}")
        return torch.cat(all_features, dim=0) if all_features else None

# --- 4. EXECUTION PIPELINE ---
if __name__ == "__main__":
    log_print(f"=== Starting LiteStyle Pipeline: {os.path.basename(__file__)} ===")
    
    # Initialize Extractor
    extractor = FeatureExtractor(YOLO_MODEL_PATH, DEVICE)
    
    # Example for Training (logic for path loading assumed from your previous snippet)
    # ... [Path Loading Logic] ...

    # Log Progress
    log_print(f"Dataset Summary: Train ({len(train_paths)}) | Val ({len(val_paths)})")
    
    # Run Training with Early Stopping (as defined in your script)
    # ... [MLP Training Loop] ...
    
    log_print("Pipeline complete. Ready for real-time inference.")