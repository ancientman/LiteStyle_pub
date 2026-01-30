"""
Unified YOLO Inference & Batch Processor
========================================
Version: 1.2
Date: January 06, 2026

This code provides a unified pipeline for garment detection using 
YOLOv11. It supports targeted single-file inference or recursive batch folder 
processing, outputting high-resolution annotated results for qualitative review.
"""

import argparse
import sys
from pathlib import Path
from ultralytics import YOLO



def run_inference():
    # 1. SETUP COMMAND LINE PARSER
    parser = argparse.ArgumentParser(description="YOLO Inference Script")
    parser.add_argument(
        "input", 
        nargs="?", 
        default=".", 
        help="Path to an image file or a directory of images (default: current directory)"
    )
    args = parser.parse_args()

    # 2. PATH CONFIGURATION
    BASE_DIR = Path(__file__).resolve().parent
    OUTPUT_DIR = BASE_DIR / "detection_results"
    INPUT_PATH = Path(args.input)

    # 3. LOAD MODEL
    # Ensure this path points to your fine-tuned YOLOv11n weights
    model = YOLO('runs/detect/train/weights/best.pt')

    # 4. DETERMINE SOURCE TYPE
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff')
    
    if INPUT_PATH.is_file():
        # Single mode: Validate extension
        if INPUT_PATH.suffix.lower() in valid_extensions:
            source_list = [str(INPUT_PATH)]
            print(f"--- Targeted Inference: {INPUT_PATH.name} ---")
        else:
            print(f"Error: File {INPUT_PATH.name} is not a supported image format.")
            return
    elif INPUT_PATH.is_dir():
        # Batch mode: Scan directory
        source_list = [
            str(f) for f in INPUT_PATH.iterdir() 
            if f.suffix.lower() in valid_extensions
        ]
        print(f"--- Batch Inference: Found {len(source_list)} images in {INPUT_PATH} ---")
    else:
        print(f"Error: Path '{args.input}' does not exist.")
        return

    # 5. EXECUTE AND DUMP
    if source_list:
        # Predict with a confidence threshold of 0.5 (Formal/Casual threshold baseline)
        model.predict(
            source=source_list,
            conf=0.5,
            save=True,
            project=str(BASE_DIR),
            name=OUTPUT_DIR.name,
            exist_ok=True
        )
        print(f"\nCompleted. Results available in: {OUTPUT_DIR}")
    else:
        print("No valid images found for processing.")

if __name__ == "__main__":
    run_inference()