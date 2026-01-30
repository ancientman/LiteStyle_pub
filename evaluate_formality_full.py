"""
Apparel Formality Evaluation & Metrics Pipeline
===============================================
Version: 1.1 
Date: January 30, 2026

This script serves as a comprehensive evaluation pipeline for the hybrid 
YOLOv11n-StyleMLP model. It enriches raw inference outputs with ground truth 
labels, computes per-class precision/recall/F1-scores, and generates 
publication-quality ROC curves and confusion matrices (300 DPI).
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_curve, auc,
    confusion_matrix, ConfusionMatrixDisplay
)

# ==================== STEP 1: Generate test_results_with_metrics.csv ====================
# Input: Raw predictions CSV from style_score_v3.py
PREDICTIONS_CSV = 'test_predictions_v3.csv'  

# Load predictions
df_pred = pd.read_csv(PREDICTIONS_CSV)
print(f"Loaded predictions: {df_pred.shape[0]} images")

# Extract ground truth from Image_Name (case-insensitive keyword search)
def extract_ground_truth(filename):
    """
    Parses the filename for 'formal' or 'casual' keywords to assign ground truth.
    Returns 'Unknown' for ambiguous cases, which are filtered later.
    """
    lower = filename.lower()
    if 'formal' in lower:
        return 'Formal'
    elif 'casual' in lower:
        return 'Casual'
    else:
        return 'Unknown'

df_pred['Ground_Truth'] = df_pred['Image_Name'].apply(extract_ground_truth)

# Report distribution
print("\nGround Truth Distribution:")
print(df_pred['Ground_Truth'].value_counts())

# Filter out unknowns
df_full = df_pred[df_pred['Ground_Truth'] != 'Unknown'].copy()

# Save enriched CSV
OUTPUT_METRICS_CSV = 'test_results_with_metrics.csv'
df_full.to_csv(OUTPUT_METRICS_CSV, index=False)
print(f"\nGenerated full metrics CSV: '{OUTPUT_METRICS_CSV}' with {len(df_full)} images")

# ==================== STEP 2: Evaluate on the full enriched dataset ====================
df = pd.read_csv(OUTPUT_METRICS_CSV)

# Binary mapping for metrics
y_true = df['Ground_Truth'].map({'Formal': 1, 'Casual': 0})
y_pred = df['Style_Class'].map({'Formal': 1, 'Casual': 0})
y_prob = df['Formal_Probability']

# Overall accuracy
accuracy = accuracy_score(y_true, y_pred)

# Per-class metrics (Casual=0, Formal=1)
precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, labels=[0, 1])

# AUC-ROC and ROC curve
fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC curve

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve: Formality Classification')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig('roc_curve_formality.png', dpi=300, bbox_inches='tight')
plt.close()

# Confusion matrix

cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Casual', 'Formal'])
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix: Formality Classification')
plt.savefig('confusion_matrix_formality.png', dpi=300, bbox_inches='tight')
plt.close()

# Summary output
print(f"\nFull Test Set Evaluation (n={len(df)} images):")
print(f"Accuracy: {accuracy:.4f}")
print(f"AUC-ROC: {roc_auc:.4f}")
print("\nPer-Class Metrics:")
print(f"Casual (Class 0): Precision={precision[0]:.4f}, Recall={recall[0]:.4f}, F1-Score={f1[0]:.4f}")
print(f"Formal (Class 1): Precision={precision[1]:.4f}, Recall={recall[1]:.4f}, F1-Score={f1[1]:.4f}")
print("\nFigures saved.")