"""
DeepFashion2 Qualitative Analyzer (Clipping Protection)
======================================================
Version: 1.2
Date: January 30, 2026

This script performs batch inference using YOLOv11n on the DeepFashion2 dataset.
It generates side-by-side comparison strips (Original vs. Predicted) and 
implements dynamic coordinate calculation to prevent label clipping at 
image boundaries.
"""

import cv2
import numpy as np
import os
from ultralytics import YOLO

# --- 1. CONFIGURATION ---
MODEL_PATH = 'best.pt'
SOURCE_PATH = './data/processed/images/val'
OUTPUT_BASE = 'eval_results'

# Define directory structure for organized analysis
dirs = {
    "success": os.path.join(OUTPUT_BASE, 'success'),     # High-confidence detections
    "failure": os.path.join(OUTPUT_BASE, 'failure'),     # Low-confidence/missed detections
    "comparison": os.path.join(OUTPUT_BASE, 'comparison') # Side-by-side strips
}

for d in dirs.values():
    os.makedirs(d, exist_ok=True)

# Initialize YOLO model
model = YOLO(MODEL_PATH)

# Run inference on the source directory
# Using stream=True for memory efficiency with large datasets
results = model.predict(source=SOURCE_PATH, conf=0.25, stream=True)

print("[INFO] Processing images and generating anti-clipping visualizations...")

# --- 2. IMAGE PROCESSING LOOP ---
for result in results:
    img_name = os.path.basename(result.path)
    
    # Logic for categorization: Success vs. Failure
    confidences = result.boxes.conf.tolist() if len(result.boxes) > 0 else []
    is_success = any(c > 0.85 for c in confidences)
    is_failure = any(0.25 < c < 0.45 for c in confidences) or len(confidences) == 0

    # --- 3. GENERATE ANNOTATED PLOT ---
    pred_plot = result.plot(
        line_width=3, 
        font_size=1.5,
        labels=True,
        boxes=True
    ) 
    
    # Load original for comparison and ensure size match
    orig_img = cv2.imread(result.path)
    if orig_img is None:
        continue
        
    h, w, _ = pred_plot.shape
    orig_resized = cv2.resize(orig_img, (w, h))
    
    # Create the horizontal side-by-side strip
    combined = np.hstack((orig_resized, pred_plot))
    
    # --- 4. DYNAMIC HEADER POSITIONING (Anti-Clipping) ---
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 3
    
    # Calculate text width to ensure the second header doesn't clip off the right edge
    text_label = "YOLO11n Pred"
    (tw, th), _ = cv2.getTextSize(text_label, font, font_scale, thickness)
    
    # The X-coordinate for the second image's header
    pred_text_x = w + 20
    
    # Boundary check: If the text is wider than the remaining space, nudge it left
    if (pred_text_x + tw) > (2 * w):
        pred_text_x = (2 * w) - tw - 10 
    
    # Draw headers with black outlines for high-contrast legibility
    for text, x_pos in [("Original", 20), (text_label, pred_text_x)]:
        # Background shadow
        cv2.putText(combined, text, (x_pos, 50), font, font_scale, (0, 0, 0), thickness + 2)
        # Foreground text
        color = (255, 255, 255) if text == "Original" else (0, 255, 0)
        cv2.putText(combined, text, (x_pos, 50), font, font_scale, color, thickness)

    # --- 5. EXPORT RESULTS ---
    comp_path = os.path.join(dirs["comparison"], img_name)
    cv2.imwrite(comp_path, combined)

    if is_success:
        cv2.imwrite(os.path.join(dirs["success"], img_name), pred_plot)
    elif is_failure:
        cv2.imwrite(os.path.join(dirs["failure"], img_name), pred_plot)

print(f"\n[SUCCESS] Qualitative analysis saved to: {OUTPUT_BASE}")