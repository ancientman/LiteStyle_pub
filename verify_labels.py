"""
Dataset Annotation Verifier (YOLO Format)
=========================================
Version: 1.1
Date: January 06, 2026

This diagnostic tool randomly samples processed images and their corresponding 
YOLO text labels to verify coordinate scaling and bounding box alignment. 
Essential for ensuring data integrity before starting deep learning training.
"""

import cv2
import os
import random
import matplotlib.pyplot as plt
from pathlib import Path

# --- UPDATED PATHS (Consistent with your /processed/ folder) ---
BASE_DIR = Path(r"./data/processed")
IMG_DIR = BASE_DIR / "images" / "val"
LABEL_DIR = BASE_DIR / "labels" / "val"

# DeepFashion2-based Category Mapping
CLASSES = [
    "Short Sleeve Top", "Long Sleeve Top", "Short Sleeve Outwear", "Long Sleeve Outwear", 
    "Vest", "Sling", "Shorts", "Trousers", "Skirt", "Short Sleeve Dress", 
    "Long Sleeve Dress", "Vest Dress", "Sling Dress"
]



def verify_dataset():
    # 1. Check directories
    if not IMG_DIR.exists() or not LABEL_DIR.exists():
        print(f"❌ Paths not found!")
        print(f"Looking for images in: {IMG_DIR}")
        print(f"Looking for labels in: {LABEL_DIR}")
        return

    # 2. Match Images with Labels
    label_files = list(LABEL_DIR.glob("*.txt"))
    print(f"🔍 Found {len(label_files)} labels in the processed folder.")

    if not label_files:
        print("⚠️ No labels found. Ensure parallel_convert.py was successful.")
        return

    # 3. Pick 3 random samples for quick sanity check
    samples = random.sample(label_files, min(3, len(label_files)))

    for label_path in samples:
        img_id = label_path.stem
        img_path = IMG_DIR / f"{img_id}.jpg"
        
        if not img_path.exists():
            print(f"⚠️ Image missing for label: {img_id}")
            continue
            
        # Load Image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        
        # Draw YOLO Labels
        with open(label_path, 'r') as f:
            for line in f:
                # YOLO format: cls x_center y_center width height (all normalized 0-1)
                cls_id, xc, yc, bw, bh = map(float, line.split())
                
                # Rescale from 0-1 back to pixel coordinates for OpenCV drawing
                x1 = int((xc - bw/2) * w)
                y1 = int((yc - bh/2) * h)
                x2 = int((xc + bw/2) * w)
                y2 = int((yc + bh/2) * h)
                
                # Draw bounding box and label text
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                label_text = f"{CLASSES[int(cls_id)]}"
                cv2.putText(img, label_text, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Plotting using Matplotlib for interactive viewing
        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.title(f"Verification: {img_id}.jpg")
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    verify_dataset()