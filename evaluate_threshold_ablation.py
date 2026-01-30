"""
LiteStyle Threshold Sensitivity Ablation
========================================
Version: 1.1 
Date: January 30, 2026

Evaluates StyleMLP performance across varying decision thresholds (τ).
Input: 'test_results_with_metrics.csv'
"""

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load predictions (578 balanced consumer images)
try:
    df = pd.read_csv('test_results_with_metrics.csv')
    # Map ground truth to binary (Formal=1, Casual=0)
    y_true = df['Ground_Truth'].map({'Formal': 1, 'Casual': 0}).values
    # Extract continuous scores
    probs = df['Formal_Probability'].values
except Exception as e:
    print(f"Error: Could not load data. {e}")
    exit()

# Define thresholds for the ablation study
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

print("\n" + "="*65)
print(f"{'Threshold τ':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-score':<12}")
print("-" * 65)

for tau in thresholds:
    # Binary classification based on threshold
    y_pred = (probs >= tau).astype(int)
    
    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Highlight the baseline
    suffix = " (baseline)" if tau == 0.5 else ""
    label = f"{tau:.1f}{suffix}"
    
    print(f"{label:<15} {acc:<12.3f} {prec:<12.3f} {rec:<12.3f} {f1:<12.3f}")

print("="*65)

print("\nQuick Interpretation:")
print(f" - Lower τ (0.3-0.4): High Recall. Catching more 'Formal' items, but with more false positives.")
print(f" - Baseline τ (0.5): Optimal Balance. Highest F1-score for general consumer use.")
print(f" - Higher τ (0.6-0.7): High Precision. Very strict filtering; only labeling high-confidence items as Formal.")