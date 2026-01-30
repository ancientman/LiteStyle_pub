"""
Apparel Formality Inference Pipeline 
=================================================
Version: 3.0
Date: January 30, 2026

This production-ready script performs high-speed inference for classifying attire 
as 'Casual' or 'Formal'. It utilizes a frozen YOLOv11n backbone as a feature 
extractor (SPPF layer) coupled with a trained StyleMLP-v3 classification head.
"""

import os
import torch
import torch.nn as nn
import pandas as pd
from ultralytics import YOLO
import torch.nn.functional as F
from tqdm import tqdm

# --- CONFIGURATION ---
YOLO_WEIGHTS = 'best.pt'                 # YOLOv11 model (frozen feature extractor)
STYLE_WEIGHTS = 'style_model_v3.pth'     # Trained StyleMLP weights
IMAGE_DIR = './style_images/test'        # Folder with images to classify (recursive)
BATCH_SIZE = 32                          # Adjust based on GPU memory
SAVE_NAME = 'test_predictions_v3.csv'    # Output CSV filename
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"--- INITIALIZING STYLE INFERENCE ENGINE (Device: {DEVICE}) ---")

# 1. STYLE CLASSIFIER HEAD — MUST EXACTLY MATCH TRAINING ARCHITECTURE

class StyleMLP(nn.Module):
    """
    Improved 3-layer MLP with BatchNorm after Linear layers.
    Input: 256-dimensional feature vector from YOLOv11 SPPF layer (after GAP).
    Output: Sigmoid probability [0.0 = Casual, 1.0 = Formal].
    """
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

# Global buffer to collect features from the forward hook
features_buffer = []

def hook_fn(module, input, output):
    """
    Forward hook attached to YOLOv11 SPPF layer.
    Applies Global Average Pooling (GAP) to spatial feature maps.
    """
    if isinstance(output, tuple):
        output = output[0] 
    
    gap = F.adaptive_avg_pool2d(output, (1, 1)) 
    features_buffer.append(gap.view(gap.size(0), -1).detach().cpu()) 

# 2. MAIN INFERENCE FUNCTION

def run_style_inference():
    # Load YOLOv11 model and attach hook to SPPF layer
    model = YOLO(YOLO_WEIGHTS)
    model.to(DEVICE)

    # Dynamically find SPPF layer (usually index 9 in YOLOv11)
    target_layer_idx = 9 if len(model.model.model) > 10 else (len(model.model.model) - 2)
    try:
        model.model.model[target_layer_idx].register_forward_hook(hook_fn)
        print(f"✓ Hook successfully attached to YOLO Layer {target_layer_idx} (SPPF).")
    except Exception as e:
        print(f"✘ Failed to attach hook: {e}")
        return

    # Load trained StyleMLP
    style_clf = StyleMLP(input_dim=256).to(DEVICE)
    if not os.path.exists(STYLE_WEIGHTS):
        print(f"✘ Style weights not found: {STYLE_WEIGHTS}")
        return
    
    style_clf.load_state_dict(torch.load(STYLE_WEIGHTS, map_location=DEVICE))
    style_clf.eval() 
    print(f"✓ StyleMLP weights loaded from {STYLE_WEIGHTS}")

    # Collect all image paths recursively
    all_images = []
    for root, _, files in os.walk(IMAGE_DIR):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_images.append(os.path.join(root, f))
    
    if not all_images:
        print(f"✘ No images found in {IMAGE_DIR}")
        return
    
    print(f"-> Found {len(all_images)} images. Processing in batches of {BATCH_SIZE}...")

    results_list = []

    # Inference loop
    with torch.no_grad():
        for i in tqdm(range(0, len(all_images), BATCH_SIZE), desc="Scoring Images"):
            batch_paths = all_images[i:i + BATCH_SIZE]
            actual_batch_size = len(batch_paths)
            features_buffer.clear()

            # Forward pass through YOLO — triggers hook_fn
            yolo_results = model.predict(batch_paths, imgsz=640, device=DEVICE, verbose=False)

            if features_buffer:
                all_features = torch.cat(features_buffer, dim=0) 
                batch_features = all_features[:actual_batch_size].to(DEVICE) 

                # Get style probabilities
                style_scores = style_clf(batch_features).cpu().numpy().flatten() 

                if len(style_scores) != actual_batch_size:
                    print(f"\nWarning: Feature count mismatch in batch {i//BATCH_SIZE + 1}. Skipping.")
                    continue

                for idx, path in enumerate(batch_paths):
                    score = float(style_scores[idx])
                    predicted_class = "Formal" if score >= 0.5 else "Casual"

                    # Extract detected garment from YOLO
                    detected_garment = "Unknown"
                    if idx < len(yolo_results):
                        res = yolo_results[idx]
                        if len(res.boxes) > 0:
                            cls_id = int(res.boxes.cls[0].item())
                            detected_garment = res.names.get(cls_id, "Unknown")

                    results_list.append({
                        "Image_Name": os.path.basename(path),
                        "File_Path": path,
                        "Detection": detected_garment,
                        "Formal_Probability": round(score, 4),
                        "Style_Class": predicted_class
                    })

    # Save results to CSV
    if results_list:
        df = pd.DataFrame(results_list)
        df.to_csv(SAVE_NAME, index=False)
        print(f"\n--- INFERENCE COMPLETE ---")
        print(f"Saved {len(df)} predictions to '{SAVE_NAME}'")
    else:
        print("\nNo results generated.")

if __name__ == "__main__":
    run_style_inference()