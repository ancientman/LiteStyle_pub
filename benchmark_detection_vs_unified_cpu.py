"""
LiteStyle CPU Performance Benchmarking Suite
============================================
Version: 1.5 
Date: January 30, 2026

Evaluates end-to-end CPU inference, comparing Detection baseline vs. Unified 
Pipeline across multiple optimization backends (ONNX, INT8, OpenVINO).
"""

import os
import time
import numpy as np
import cv2
import onnxruntime as ort
from tqdm import tqdm

# --- 1. ENVIRONMENT SETUP ---
try:
    import openvino as ov
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False

# Configuration
TEST_IMAGE_DIR    = './style_images/test'
DETECTION_ONNX    = 'best.onnx'
UNIFIED_ONNX      = 'combined.onnx'
STATIC_INT8_ONNX  = 'combined_static_int8.onnx'
IMG_SZ            = 640
WARMUP_ITERS      = 20
LOG_FILE          = 'benchmark_lite_style_cpu.txt'

def log_print(msg):
    print(msg)
    with open(LOG_FILE, 'a') as f:
        f.write(msg + '\n')

# --- 2. IMAGE PREPROCESSING ---
def load_and_preprocess():
    paths = [os.path.join(TEST_IMAGE_DIR, f) for f in os.listdir(TEST_IMAGE_DIR)
             if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if not paths:
        raise FileNotFoundError("No images found in test directory.")
    
    images = []
    for p in paths:
        img = cv2.imread(p)
        img = cv2.resize(img, (IMG_SZ, IMG_SZ))
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1) # HWC -> CHW
        images.append(np.expand_dims(img, 0)) # NCHW
    return images, paths

# --- 3. ONNX RUNTIME BENCHMARK ---
def benchmark_onnx(model_path, data, label):
    if not os.path.exists(model_path):
        return None, None

    # Optimize for CPU execution
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = os.cpu_count()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # Try OpenVINO Execution Provider, fallback to CPU
    providers = ['OpenVINOExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
    
    input_name = session.get_inputs()[0].name
    
    # Warm-up (Critical for JIT/Provider initialization)
    for _ in range(WARMUP_ITERS):
        session.run(None, {input_name: data[0]})

    start = time.perf_counter()
    for img in data:
        session.run(None, {input_name: img})
    elapsed = time.perf_counter() - start

    latency = (elapsed / len(data)) * 1000
    fps = len(data) / elapsed
    return latency, fps

# --- 4. OPENVINO IR BENCHMARK ---
def benchmark_ov(model_path, data):
    if not OPENVINO_AVAILABLE or not os.path.exists(model_path):
        return None, None
    
    core = ov.Core()
    # Convert ONNX to OpenVINO IR internally for benchmarking
    ov_model = core.read_model(model_path)
    compiled = core.compile_model(ov_model, 'CPU')
    input_layer = compiled.input(0)

    # Warm-up
    for _ in range(WARMUP_ITERS):
        compiled([data[0]])

    start = time.perf_counter()
    for img in data:
        compiled([img])
    elapsed = time.perf_counter() - start

    latency = (elapsed / len(data)) * 1000
    fps = len(data) / elapsed
    return latency, fps

# --- 5. MAIN EXECUTION ---
if __name__ == "__main__":
    log_print(f"Starting LiteStyle Benchmark v1.5...")
    test_data, _ = load_and_preprocess()
    results = {}

    # Run variants
    variants = [
        (DETECTION_ONNX, "Detection-Only (Baseline)"),
        (UNIFIED_ONNX, "Unified (FP32)"),
        (STATIC_INT8_ONNX, "Unified (Static INT8)")
    ]

    for path, name in variants:
        lat, fps = benchmark_onnx(path, test_data, name)
        if lat: results[name] = (lat, fps)

    # Pure OpenVINO Test
    lat_ov, fps_ov = benchmark_ov(UNIFIED_ONNX, test_data)
    if lat_ov: results["Unified (OpenVINO IR)"] = (lat_ov, fps_ov)

    # Summary Table
    log_print("\n" + "="*80)
    log_print(f"{'Model Variant':<30} | {'Latency (ms)':<15} | {'FPS':<10} | {'Overhead'}")
    log_print("-" * 80)
    
    base_lat = results.get("Detection-Only (Baseline)", (0, 0))[0]
    
    for name, (lat, fps) in results.items():
        overhead = f"{((lat - base_lat)/base_lat)*100:+.2f}%" if base_lat else "N/A"
        log_print(f"{name:<30} | {lat:<15.2f} | {fps:<10.2f} | {overhead}")
    log_print("="*80)