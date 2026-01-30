"""
Garment Style Variance Visualization
====================================
Version: 1.1
Date: January 06, 2026

This script visualizes 'Intra-Class Style Variance' within a specific garment 
category. It utilizes Kernel Density Estimation (KDE) to demonstrate how 
the model differentiates between casual and formal styles for items sharing 
the same object detection label.

Output is a high-resolution PDF optimized for inclusion in academic papers.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_high_res_pdf(csv_path, target_category='Long Sleeve Top'):
    """
    Generates a high-resolution PDF visualization of style distributions
    using clean typography and vector graphics.
    """
    
    # --- 1. PUBLICATION FONT CONFIGURATION ---
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"], 
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.autolayout": True
    })

    # --- 2. DATA LOADING & VALIDATION ---
    if not os.path.exists(csv_path):
        print(f"ERROR: Results file '{csv_path}' not found.")
        return

    df = pd.read_csv(csv_path)
    
    # Filter for the specific garment category
    subset = df[df['Detection'] == target_category]
    
    if subset.empty:
        print(f"ERROR: No detection data found for category: '{target_category}'.")
        return

    scores = subset['Formal_Probability']

    # --- 3. PLOT INITIALIZATION ---
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))

    # --- 4. DISTRIBUTION VISUALIZATION ---
    # KDE Plot for the probability density
    sns.kdeplot(
        scores, 
        bw_adjust=0.6, 
        fill=True, 
        color='#008080', # Teal
        alpha=0.3, 
        linewidth=2,
        label='Score Density',
        ax=ax
    )

    # Rug Plot to show individual data points
    sns.rugplot(
        scores, 
        color='#2F4F4F', # Dark Slate Grey
        alpha=0.6, 
        height=0.05,
        ax=ax
    )

    # --- 5. ANNOTATIONS & THRESHOLDS ---
    # Decision boundary at 0.5
    ax.axvline(
        x=0.5, 
        color='#DC143C', # Crimson
        linestyle='--', 
        linewidth=2, 
        label='Decision Threshold ($S=0.5$)'
    )

    # Dynamic label placement based on data height
    y_limit = ax.get_ylim()[1]
    ax.text(0.20, y_limit * 0.9, 'Casual-leaning', ha='center', weight='bold', color='#2F4F4F')
    ax.text(0.80, y_limit * 0.9, 'Formal-leaning', ha='center', weight='bold', color='#2F4F4F')

    # --- 6. CHART FORMATTING ---
    ax.set_title(f'Intra-Class Style Variance: {target_category}', pad=20)
    ax.set_xlabel('Predicted Formality Score ($S$)')
    ax.set_ylabel('Density')
    ax.set_xlim(0, 1)
    ax.legend(loc='upper center', frameon=True, facecolor='white', framealpha=1)

    # --- 7. VECTOR EXPORT ---
    clean_name = target_category.replace(' ', '_').lower()
    output_filename = f"variance_{clean_name}.pdf"
    
    # Save as PDF for high-fidelity scaling in LaTeX
    plt.savefig(output_filename, format='pdf', bbox_inches='tight')
    
    print(f"SUCCESS: Distribution plot saved as '{output_filename}'")
    plt.show()

if __name__ == "__main__":
    # Ensure these match your actual experiment output
    INPUT_FILE = 'test_results_with_metrics.csv'
    CATEGORY_TO_ANALYZE = 'Long Sleeve Top'
    
    generate_high_res_pdf(INPUT_FILE, CATEGORY_TO_ANALYZE)