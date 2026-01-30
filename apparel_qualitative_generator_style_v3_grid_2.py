"""
Apparel Qualitative Interpretability Grid Generator
===================================================
Version: 1.8
Date: January 06, 2026

This script generates a 2x3 high-resolution grid for qualitative analysis of the 
hybrid YOLOv11n + StyleMLP-v3 model. It visualizes object detections alongside 
Grad-CAM heatmaps to interpret "Formality" classification on consumer attire.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO
import matplotlib.pyplot as plt
from grad_cam_viz import YOLO11nGradCAM

# ---------------------------------------------------------------------------
# StyleMLP Model Definition
# ---------------------------------------------------------------------------
class StyleMLP(torch.nn.Module):
    """Simple MLP head for binary style classification (Formal vs Casual)."""
    def __init__(self, input_dim=256):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# ---------------------------------------------------------------------------
# Feature Extraction Logic
# ---------------------------------------------------------------------------
def get_formality_score(yolo_model, style_mlp, img_path, device):
    """
    Hooks into the YOLO backbone to extract global features and 
    passes them through StyleMLP for formality prediction.
    """
    features = []
    
    def hook_fn(module, input, output):
        # Global Average Pooling to flatten spatial features
        gap = F.adaptive_avg_pool2d(output, (1, 1))
        features.append(gap.view(gap.size(0), -1).detach().cpu())

    # Register hook on the target layer (Layer 9 in YOLOv11 backbone)
    hook = yolo_model.model.model[9].register_forward_hook(hook_fn)
    _ = yolo_model(img_path, verbose=False)
    hook.remove()

    if not features: 
        return 0.5, "Unknown"
    
    feat = features[0].to(device)
    with torch.no_grad():
        prob = style_mlp(feat).item()

    pred_class = "Formal" if prob >= 0.5 else "Casual"
    return round(prob, 4), pred_class

# ---------------------------------------------------------------------------
# Layout & Image Processing Helpers
# ---------------------------------------------------------------------------
def add_letterbox(image, target_h, target_w):
    """Resizes image to fit canvas while maintaining aspect ratio (padding)."""
    h, w = image.shape[:2]
    scale = min(target_h / h, target_w / w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_h, target_w, 3), 240, dtype=np.uint8) # Light gray bg
    
    y_off, x_off = (target_h - new_h) // 2, (target_w - new_w) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
    return canvas, scale, x_off, y_off

def pad_to_height(image, target_height):
    """Pads bottom of image with white pixels to match a target height."""
    h, w = image.shape[:2]
    if h >= target_height: return image
    return np.vstack((image, np.full((target_height-h, w, 3), 255, dtype=np.uint8)))

# ---------------------------------------------------------------------------
# Visualization Panels
# ---------------------------------------------------------------------------
def draw_predictions_panel(img, results, formality_prob, formality_class,
                           names, scale, x_offset, y_offset):
    """
    Renders YOLO bounding boxes and a custom HUD for Formality scores.
    Uses layered alpha-blending for the score card.
    """
    panel = img.copy()
    panel_h, panel_w = panel.shape[:2]
    font = cv2.FONT_HERSHEY_COMPLEX

    # --- Layer 1: Detections ---
    for box in results.boxes:
        ox1, oy1, ox2, oy2 = map(float, box.xyxy[0].tolist())
        x1, y1 = int(ox1 * scale + x_offset), int(oy1 * scale + y_offset)
        x2, y2 = int(ox2 * scale + x_offset), int(oy2 * scale + y_offset)

        cls_idx = int(box.cls[0].item())
        conf = box.conf[0].item()
        label = f"{names.get(cls_idx, 'Item')} {conf:.2f}"

        cv2.rectangle(panel, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Label Background Logic (Prevents clipping at edges)
        l_scale, l_thick = 0.55, 1
        (ltw, lth), l_base = cv2.getTextSize(label, font, l_scale, l_thick)
        l_h_total = lth + l_base + 10
        
        ly = (y1 - l_h_total) if (y1 - l_h_total) > 0 else (y1 + 5)
        lx = x1 if (x1 + ltw + 10) < panel_w else (panel_w - ltw - 15)

        cv2.rectangle(panel, (lx, ly), (lx + ltw + 10, ly + l_h_total), (0, 255, 0), -1)
        cv2.putText(panel, label, (lx + 5, ly + lth + 5), font, l_scale, (0, 0, 0), l_thick, cv2.LINE_AA)

    # --- Layer 2: Formality HUD ---
    f_text = f"Formality: {formality_class} ({formality_prob:.3f})"
    f_scale, f_thick = 0.65, 1
    (fw, fh), fb = cv2.getTextSize(f_text, font, f_scale, f_thick)
    
    mx, my = 42, 25
    bx1, by1 = mx, panel_h - my - fh - fb - 10
    bx2, by2 = mx + fw + 20, panel_h - my
    
    overlay = panel.copy()
    cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (0, 0, 0), -1)
    panel = cv2.addWeighted(overlay, 0.75, panel, 0.25, 0)
    cv2.rectangle(panel, (bx1, by1), (bx2, by2), (255, 255, 255), 1)
    
    cv2.putText(panel, f_text, (bx1 + 10, by1 + fh + 6), font, f_scale, (0, 165, 255), f_thick, cv2.LINE_AA)
    return panel

def create_case_header(width, text):
    """Generates a dynamic multi-line text header based on string length."""
    font_scale = 0.85
    lines, current = [], ""
    for w in text.split():
        test = current + (" " if current else "") + w
        (tw, th), _ = cv2.getTextSize(test, cv2.FONT_HERSHEY_COMPLEX, font_scale, 2)
        if tw > width - 60: 
            lines.append(current)
            current = w
        else: 
            current = test
    if current: lines.append(current)

    h_per_line = int(45 * font_scale)
    header_h = 40 + (h_per_line * len(lines)) + 30
    header = np.full((header_h, width, 3), 255, dtype=np.uint8)
    
    y = 40
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_COMPLEX, font_scale, 2)
        cv2.putText(header, line, ((width-tw)//2, y+th), cv2.FONT_HERSHEY_COMPLEX, font_scale, (0,0,0), 2, cv2.LINE_AA)
        y += h_per_line
    return header

# ---------------------------------------------------------------------------
# Main Execution Pipeline
# ---------------------------------------------------------------------------
def generate_grid_figure(model_path='best.pt', style_path='style_model_v3.pth', cases=None, output_name='grad_cam_style_qualitative.pdf'):
    """
    Processes cases and assembles the final 2x3 grid PDF.
    """
    if not cases:
        print("Error: No cases provided.")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize models
    yolo = YOLO(model_path)
    style_mlp = StyleMLP(256).to(device)
    style_mlp.load_state_dict(torch.load(style_path, map_location=device))
    style_mlp.eval()
    visualizer = YOLO11nGradCAM(model_path, layer_name='model.22')
    
    # Grid Config
    panel_h, panel_w, cap_h = 520, 400, 80
    rows = [[], []]

    for idx, case in enumerate(cases):
        img = cv2.imread(case['path'])
        if img is None:
            print(f"Warning: Could not load {case['path']}")
            continue
        
        # 1. Processing
        img_lb, sc, xo, yo = add_letterbox(img, panel_h, panel_w)
        prob, cls = get_formality_score(yolo, style_mlp, case['path'], device)
        res = yolo(case['path'], verbose=False)[0]

        # 2. Build Panels
        p_a = draw_predictions_panel(img_lb, res, prob, cls, yolo.names, sc, xo, yo)
        p_b_raw = visualizer.generate_heatmap(case['path'])
        p_b, _, _, _ = add_letterbox(p_b_raw, panel_h, panel_w)

        # 3. Assemble Case Column
        divider = np.full((panel_h + cap_h, 20, 3), 255, dtype=np.uint8)
        col_a = np.vstack((p_a, np.full((cap_h, panel_w, 3), 255, dtype=np.uint8)))
        col_b = np.vstack((p_b, np.full((cap_h, panel_w, 3), 255, dtype=np.uint8)))
        
        combined_case = np.hstack((col_a, divider, col_b))
        
        # Sub-captions
        cv2.putText(combined_case, "YOLOv11 + Formality", (20, panel_h + 50), cv2.FONT_HERSHEY_COMPLEX, 0.7, (50, 50, 50), 2, cv2.LINE_AA)
        cv2.putText(combined_case, "Grad-CAM Attention", (panel_w + 40, panel_h + 50), cv2.FONT_HERSHEY_COMPLEX, 0.7, (50, 50, 50), 2, cv2.LINE_AA)

        header = create_case_header(combined_case.shape[1], f"{chr(65+idx)}. " + case['label'])
        rows[idx // 3].append(np.vstack((header, combined_case)))

    # Ensure row normalization before stacking
    for r in range(len(rows)):
        if not rows[r]: continue
        max_h = max(i.shape[0] for i in rows[r])
        rows[r] = [pad_to_height(i, max_h) for i in rows[r]]

    # Final Stacking
    top = np.hstack(rows[0])
    bot = np.hstack(rows[1])
    sep = np.full((40, top.shape[1], 3), 245, dtype=np.uint8)
    final_img = np.vstack((top, sep, bot))
    
    # Export to High-Res PDF
    final_img_rgb = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
    h, w, _ = final_img_rgb.shape
    dpi = 300
    fig = plt.figure(figsize=(w/dpi, h/dpi))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(final_img_rgb, interpolation='lanczos')
    ax.axis('off')
    
    plt.savefig(output_name, format='pdf', dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Success! Qualitative grid saved to: {output_name}")

if __name__ == "__main__":
    # Define experimental cases for the grid
    TEST_CASES = [
        {'path': './style_images/test/t_formal_11_1.jpeg', 'label': 'True Positive: Correctly Classified as Formal (Structured Suit)'},
        {'path': './style_images/test/t_casual_10_1.jpeg', 'label': 'True Negative: Correctly Classified as Casual (Relaxed Trousers)'},
        {'path': './style_images/test/t_formal_17_3.jpeg', 'label': 'False Negative: Formal Attire Misclassified as Casual (Pose/Lighting)'},
        {'path': './style_images/test/t_casual_6_9.jpeg', 'label': 'False Positive: Casual Shorts Misclassified as Formal (Structured)'},
        {'path': './style_images/test/t_formal_14_3.jpeg', 'label': 'Challenging: Low-Confidence Formal (Unconventional Background)'},
        {'path': './style_images/test/t_casual_7_10.jpeg', 'label': 'Challenging: Ambiguous Outwear (Mixed Formality Cues)'}
    ]
    
    generate_grid_figure(cases=TEST_CASES)