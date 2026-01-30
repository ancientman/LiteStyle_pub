"""
Apparel Style Classification Interactive Demo (Gradio)
======================================================
Version: 1.0
Date: January 06, 2026

An interactive web-based interface for real-time apparel classification. 
Combines YOLOv11 detection with the StyleMLP-v3 head to provide garment 
identification and formality scoring in a single unified UI.
"""

import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from PIL import Image
import os

# --- CONFIG ---
YOLO_WEIGHTS = 'best.pt'
STYLE_WEIGHTS = 'style_model_v3.pth'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'



# 1. STYLEMLP ARCHITECTURE
class StyleMLP(nn.Module):
    """Must match the architecture in train_style_v3.py exactly."""
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

# Feature extraction buffer
features_buffer = []

def hook_fn(module, input, output):
    if isinstance(output, tuple):
        output = output[0]
    gap = F.adaptive_avg_pool2d(output, (1, 1))
    features_buffer.append(gap.view(gap.size(0), -1).detach())

# --- MODEL INITIALIZATION ---
print(f"Loading models on {DEVICE}...")
yolo_model = YOLO(YOLO_WEIGHTS)
yolo_model.to(DEVICE)

# Hook into SPPF layer (Index 9)
target_layer_idx = 9
yolo_model.model.model[target_layer_idx].register_forward_hook(hook_fn)

style_clf = StyleMLP(input_dim=256).to(DEVICE)
if not os.path.exists(STYLE_WEIGHTS):
    raise FileNotFoundError(f"Missing weight file: {STYLE_WEIGHTS}")
style_clf.load_state_dict(torch.load(STYLE_WEIGHTS, map_location=DEVICE))
style_clf.eval()

# 2. CORE INFERENCE LOGIC
def predict_style(image: Image.Image):
    features_buffer.clear()
    
    # Run YOLO (extracts features via hook)
    yolo_model.predict(image, imgsz=640, device=DEVICE, verbose=False)
    
    if not features_buffer:
        return "Error: No features extracted", None
    
    # Process StyleMLP prediction
    feature = features_buffer[0].to(DEVICE)
    with torch.no_grad():
        prob = style_clf(feature).cpu().item()
    
    label = "Formal" if prob >= 0.5 else "Casual"
    confidence = prob if prob >= 0.5 else (1 - prob)
    
    # Get metadata from the first detection
    results = yolo_model(image)[0]
    detected = "Unknown"
    if len(results.boxes) > 0:
        cls_id = int(results.boxes.cls[0].item())
        detected = results.names.get(cls_id, "Unknown")
    
    # Format markdown output
    text_output = (
        f"### Results\n"
        f"- **Style Prediction:** `{label}`\n"
        f"- **Formal Probability:** `{prob:.4f}`\n"
        f"- **Confidence:** `{confidence:.2%}`\n"
        f"- **Top Garment Detected:** `{detected}`"
    )
    
    return text_output, results.plot()

# 3. GRADIO USER INTERFACE


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧥 StyleAI: Casual vs Formal Classifier")
    gr.Markdown("Leveraging **YOLOv11** and **StyleMLP-v3** for intelligent apparel analysis.")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="pil", label="Input Image")
            btn = gr.Button("Analyze Style", variant="primary")
        with gr.Column():
            output_text = gr.Markdown(label="Classification Breakdown")
            output_img = gr.Image(label="Detection Visualization")
    
    btn.click(
        fn=predict_style,
        inputs=input_img,
        outputs=[output_text, output_img]
    )

if __name__ == "__main__":
    demo.launch(share=True)