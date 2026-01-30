"""
DeepFashion2 to YOLO Annotation Converter (Parallel)
====================================================
Version: 1.0
Date: January 06, 2026

This high-performance script converts DeepFashion2 JSON annotations into the 
normalized YOLO format (x_center, y_center, width, height). It utilizes 
multiprocessing to handle large-scale datasets and performs coordinate 
normalization based on real-time image dimensions.
"""

import json
import os
import cv2
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor



def convert_one_json(json_path, img_dir, label_dir):
    """
    Parses a single DeepFashion2 JSON file and exports a YOLO-compatible .txt file.
    Note: DeepFashion2 uses [y1, x1, y2, x2] for bounding boxes.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    img_name = Path(json_path).stem + ".jpg"
    img_path = os.path.join(img_dir, img_name)
    
    # Image dimensions are required for coordinate normalization (0.0 to 1.0)
    img = cv2.imread(img_path)
    if img is None: 
        return
    h, w, _ = img.shape
    
    yolo_file = os.path.join(label_dir, Path(json_path).stem + ".txt")
    
    with open(yolo_file, 'w') as f_out:
        # DeepFashion2 stores multiple objects as 'item1', 'item2', etc.
        for key in data.keys():
            if key.startswith('item'):
                item = data[key]
                # Adjusting category_id to 0-indexed for YOLO compatibility
                category_id = item['category_id'] - 1 
                box = item['bounding_box'] # Format: [y1, x1, y2, x2]
                
                # Calculate width and height in pixels
                bw = (box[3] - box[1])
                bh = (box[2] - box[0])
                
                # Calculate center coordinates in pixels
                xc = box[1] + bw / 2
                yc = box[0] + bh / 2
                
                # Write normalized coordinates: <class> <x_center> <y_center> <width> <height>
                f_out.write(f"{category_id} {xc/w:.6f} {yc/h:.6f} {bw/w:.6f} {bh/h:.6f}\n")

# --- EXECUTION BLOCK ---
if __name__ == "__main__":
    # Configuration: Update these paths based on your local directory structure
    RAW_VAL_JSON = "data/raw/val/annos"
    RAW_VAL_IMG = "data/raw/val/image"
    OUT_VAL_LABEL = "data/processed/labels/val"
    
    # Ensure output directory exists
    os.makedirs(OUT_VAL_LABEL, exist_ok=True)
    
    files = [os.path.join(RAW_VAL_JSON, f) for f in os.listdir(RAW_VAL_JSON)]
    
    print(f"--- Parallel Conversion Started: {len(files)} files ---")
    
    # Parallelize the conversion process across available CPU cores
    
    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(convert_one_json, files, 
                               [RAW_VAL_IMG]*len(files), 
                               [OUT_VAL_LABEL]*len(files)), total=len(files)))
        
    print(f"\n✓ Conversion complete. YOLO labels saved to: {OUT_VAL_LABEL}")