"""
LiteStyle OpenVINO Inference Benchmark
======================================
Version: 1.0
Date: January 30, 2026

This script evaluates the inference performance (Latency and Throughput) 
of the combined YOLO + StyleMLP model after conversion to OpenVINO IR format.
Optimized for FP16 precision on Intel CPU/iGPU architectures.
"""

import time
import cv2
import numpy as np
import openvino as ov
import os

# --- 1. CONFIGURATION ---
MODEL_XML = "combined_openvino/IR_fp16.xml"
IMAGE_DIR = "images"
OUTPUT_TXT = "openvino_benchmark_results.txt"

# Model specific parameters
IMG_SZ = 640
WARMUP_RUNS = 10
BENCHMARK_RUNS = 100

# --- 2. LOAD OPENVINO RUNTIME ---
print("[INFO] Initializing OpenVINO Core...")
core = ov.Core()

# Load the Intermediate Representation (IR) model
model = core.read_model(MODEL_XML)

# Compile model for the target device (CPU)
# OpenVINO automatically optimizes threads based on hardware
compiled_model = core.compile_model(
    model=model,
    device_name="CPU"
)

input_layer = compiled_model.input(0)
output_layers = compiled_model.outputs

print(f"[SUCCESS] Model loaded: {MODEL_XML}")
print(f"Backbone Inputs : {len(compiled_model.inputs)}")
print(f"Classification Outputs: {len(compiled_model.outputs)}")

# --- 3. PREPROCESSING PIPELINE ---
def preprocess(img_path):
    """Resizes, normalizes, and reshapes image to NCHW format."""
    img = cv2.imread(img_path)
    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")
    
    img = cv2.resize(img, (IMG_SZ, IMG_SZ))
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1) # HWC to CHW
    return img[None] # Add Batch dimension (NCHW)

# Cache preprocessed images in memory to isolate inference speed
image_paths = [
    os.path.join(IMAGE_DIR, f)
    for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".png", ".jpeg"))
]

if not image_paths:
    raise RuntimeError(f"No test images found in '{IMAGE_DIR}'")

print(f"[INFO] Preloading {len(image_paths)} images for benchmark...")
inputs = [preprocess(p) for p in image_paths]

# --- 4. EXECUTION ---
print(f"[INFO] Running {WARMUP_RUNS} warm-up iterations...")
for i in range(WARMUP_RUNS):
    compiled_model([inputs[i % len(inputs)]])

latencies = []
formality_scores = []

print(f"[INFO] Benchmarking {BENCHMARK_RUNS} runs...")
start_total = time.perf_counter()

for i in range(BENCHMARK_RUNS):
    inp = inputs[i % len(inputs)]

    t0 = time.perf_counter()
    results = compiled_model([inp])
    t1 = time.perf_counter()

    latencies.append((t1 - t0) * 1000.0) # Convert to ms

    # Extract the StyleMLP formality score (assumed terminal output)
    score = results[-1][0][0]
    formality_scores.append(float(score))

end_total = time.perf_counter()

# Statistics Calculation
total_time = end_total - start_total
avg_latency = np.mean(latencies)
std_latency = np.std(latencies)
fps = BENCHMARK_RUNS / total_time

# --- 5. REPORT GENERATION ---
report_content = [
    "LiteStyle OpenVINO FP16 Benchmark",
    "=================================",
    f"Model Path        : {MODEL_XML}",
    "Hardware Device   : Intel CPU (OpenVINO Optimized)",
    "Precision Mode    : FP16",
    f"Input Resolution  : {IMG_SZ}x{IMG_SZ}",
    "",
    f"Total Benchmark   : {BENCHMARK_RUNS} iterations",
    f"Average Latency   : {avg_latency:.2f} ms",
    f"Standard Deviation: {std_latency:.2f} ms",
    f"Throughput        : {fps:.2f} FPS",
    "",
    "Top 10 Prediction Samples:",
    "--------------------------"
]

for i, s in enumerate(formality_scores[:10]):
    label = "Formal" if s > 0.5 else "Casual"
    report_content.append(f"  [{i:02d}] Score: {s:.3f} -> {label}")

with open(OUTPUT_TXT, "w") as f:
    f.write("\n".join(report_content))

print("\n" + "="*30)
print(f"BENCHMARK SUMMARY")
print("="*30)
print(f"Avg Latency: {avg_latency:.2f} ms")
print(f"Throughput : {fps:.2f} FPS")
print(f"Report     : {OUTPUT_TXT}")