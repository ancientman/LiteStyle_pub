"""
Consumer Garment Formality Classification: Interpretability Analysis
====================================================================
Version: 1.0
Date: January 06, 2026

This module provides deep-dive interpretability for the StyleMLP classification 
head. It maps weights from the final decision layer back to the YOLOv11n 
backbone features, identifying which specific neural pathways drive 
'Formal' vs. 'Casual' predictions.

Key Features:
- Weight attribution visualization (Decision Layer).
- Feature sensitivity mapping (Input Layer tracing).
- Vectorized PDF output optimized for LaTeX/Academic publication.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.lines import Line2D

# --- 1. PUBLICATION QUALITY CONFIGURATION ---
# Settings optimized for legibility in multi-column academic papers.
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'font.family': 'serif',
    'figure.autolayout': True,
    'savefig.dpi': 300
})

# --- 2. ARCHITECTURE DEFINITION ---
class StyleMLP(nn.Module):
    """
    Lightweight MLP head for style classification.
    Input: 256-dimensional feature vector from YOLOv11n backbone.
    """
    def __init__(self, input_dim=256, use_bn=True, use_dropout=True, layers=3):
        super().__init__()
        modules = []
        # Support for architectural ablation studies
        sizes = [input_dim, 128, 64, 1] if layers == 3 else [input_dim, 64, 1]
        
        for i in range(len(sizes)-1):
            modules.append(nn.Linear(sizes[i], sizes[i+1]))
            # Apply Batch Normalization and Dropout to hidden layers only
            if use_bn and i < len(sizes)-2:
                modules.append(nn.BatchNorm1d(sizes[i+1]))
            if i < len(sizes)-2:
                modules.append(nn.ReLU(inplace=True))
                if use_dropout:
                    drop_rate = 0.3 if i == 0 else 0.2
                    modules.append(nn.Dropout(drop_rate))
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)

# --- 3. CONFIGURATION & MODEL LOADING ---
VARIANT_NAME = 'baseline'  # Options: 'baseline', 'no_bn', 'no_dropout', '2_layer'
WEIGHTS_PATH = f'style_{VARIANT_NAME}.pth'

print(f"[INFO] Starting Analysis for Variant: {VARIANT_NAME.upper()}")

# Map configuration keys to match ablation study parameters
configs = {
    'baseline':   {'use_bn': True,  'use_dropout': True,  'layers': 3},
    'no_bn':      {'use_bn': False, 'use_dropout': True,  'layers': 3},
    'no_dropout': {'use_bn': True,  'use_dropout': False, 'layers': 3},
    '2_layer':    {'use_bn': True,  'use_dropout': True,  'layers': 2}
}

model = StyleMLP(**configs[VARIANT_NAME])

# Load pretrained weights with CPU fallback
if os.path.exists(WEIGHTS_PATH):
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location='cpu'))
    model.eval()
    print(f"[SUCCESS] Loaded weights from {WEIGHTS_PATH}")
else:
    print(f"[WARNING] {WEIGHTS_PATH} not found. Using initialized weights for dry run.")

# --- 4. EXTRACTION OF WEIGHT DATA ---
# Extract Layer 1 (Input Features -> Hidden)
first_linear = model.net[0]
w_input_to_hidden = first_linear.weight.data.cpu().numpy()

# Extract Final Layer (Hidden -> Decision Score)
# Iterates through network to find the terminal Linear layer
final_linear = [m for m in model.net if isinstance(m, nn.Linear)][-1]
w_hidden_to_output = final_linear.weight.data.cpu().numpy().flatten()
num_hidden = len(w_hidden_to_output)

# --- 5. PLOT A: HIDDEN NEURON ATTRIBUTION ---
print(f"[LOG] Generating Hidden Influence Plot...")

plt.figure(figsize=(10, 6))
# Green correlates to 'Formal' (+) and Red to 'Casual' (-)
colors = ['#2ca02c' if w > 0 else '#d62728' for w in w_hidden_to_output]

plt.bar(range(num_hidden), w_hidden_to_output, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
plt.axhline(0, color='black', linewidth=1.2)

# Professional Legend for Interpretability
legend_elements = [
    Line2D([0], [0], color='#2ca02c', lw=6, label='Evidence for Formal (+)'),
    Line2D([0], [0], color='#d62728', lw=6, label='Evidence for Casual (-)')
]
plt.legend(handles=legend_elements, loc='best')

plt.xlabel('Hidden Neuron Index ($h_i$)')
plt.ylabel('Weight Value ($w_{out}$)')
plt.title(f'Decision Layer Weight Attribution: {VARIANT_NAME.upper()}')

# Annotate top 5 most influential neurons by magnitude
top_5_idx = np.argsort(np.abs(w_hidden_to_output))[-5:]
for idx in top_5_idx:
    val = w_hidden_to_output[idx]
    plt.annotate(f'{val:+.2f}', (idx, val), xytext=(0, 7 if val > 0 else -18),
                 textcoords='offset points', ha='center', fontsize=11, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.6))

plt.savefig(f'hidden_influence_{VARIANT_NAME}.pdf', bbox_inches='tight')

# --- 6. PLOT B: BACKWARD TRACING SENSITIVITY ---
# Identify the hidden neuron with the single largest impact on output
top_neuron_idx = np.argmax(np.abs(w_hidden_to_output))
# Retrieve the absolute weights of all 256 YOLO channels feeding into this neuron
top_neuron_inputs = np.abs(w_input_to_hidden[top_neuron_idx])

print(f"[LOG] Tracing Neuron #{top_neuron_idx} back to backbone features...")

plt.figure(figsize=(10, 5))
plt.bar(range(256), top_neuron_inputs, color='steelblue', width=1.0)
plt.xlabel('YOLOv11n SPPF Feature Channel Index')
plt.ylabel('Sensitivity (Absolute Weight)')
plt.title(f'Backbone Channel Sensitivity for Dominant Neuron {top_neuron_idx}')
plt.grid(axis='y', linestyle='--', alpha=0.3)

plt.savefig(f'top_neuron_sensitivity_{VARIANT_NAME}.pdf', bbox_inches='tight')

# --- 7. FINAL INTERPRETABILITY SUMMARY ---
print("\n" + "="*60)
print("FEATURE MAPPING SUMMARY")
print("="*60)
print(f"Most Influential Neuron:  #{top_neuron_idx}")
print(f"Classification Bias:      {'FORMAL (+)' if w_hidden_to_output[top_neuron_idx] > 0 else 'CASUAL (-)'}")
print(f"Most Active YOLO Channel: #{np.argmax(top_neuron_inputs)}")
print("="*60)
print("Analysis complete. Figures exported as high-resolution PDF vector graphics.")