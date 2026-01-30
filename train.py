"""
YOLOv13-Nano Apparel Training Pipeline
======================================
Version: 1.0
Date: January 06, 2026

This script executes the fine-tuning of a YOLOv13-Nano backbone on a custom 
apparel dataset. Optimized for NVIDIA RTX 4090 hardware, it utilizes high 
batch sizes and standard 640px input resolution to establish a baseline 
for garment detection performance.
"""

from ultralytics import YOLO



# 1. Load the YOLOv13-Nano model
# yolov13n.pt represents the latest 2025 SOTA for mobile and edge efficiency.
model = YOLO('yolov13n.pt') 

# 2. Train on RTX 4090
# High batch sizes (64+) help stabilize BatchNorm statistics during fine-tuning.
results = model.train(
    data='apparel.yaml',
    epochs=100,
    imgsz=640,
    batch=64,       # The 4090 24GB VRAM can handle high throughput comfortably
    device=0,       # Targets the primary local GPU
    name='yolov13_apparel_baseline',
    optimizer='AdamW', # Recommended for transformer-based components in newer YOLO versions
    exist_ok=True
)