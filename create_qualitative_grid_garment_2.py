"""
Qualitative Garment Grid Generator
==================================
Version: 1.1
Date: January 30, 2026

Automates the creation of a 3-column wide-aspect image grid for qualitative 
analysis. It applies a uniform layout with extra-large bold headers and 
high-contrast borders, optimized for vector-quality PDF export.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import rcParams
import os

def create_3_column_grid(image_paths, output_name, labels=None):
    """
    Creates a 3-column grid with extra-large labels and thick borders.
    Automatically converts output extension to .pdf for vector quality.
    """
    # Ensure vector output for LaTeX/Publication compatibility
    if output_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        output_name = os.path.splitext(output_name)[0] + '.pdf'

    # --- Configuration ---
    num_images = len(image_paths)
    num_cols = 3
    num_rows = (num_images + num_cols - 1) // num_cols
    
    # Global Font Setup
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    
    # Initialize figure with dynamic height based on rows
    fig, axes = plt.subplots(num_rows, num_cols, 
                             figsize=(20, 7 * num_rows), 
                             constrained_layout=True)
    
    # Flatten axes array for consistent indexing regardless of row count
    if num_rows == 1:
        axes_flat = axes
    else:
        axes_flat = axes.flatten()

    for i in range(len(axes_flat)):
        ax = axes_flat[i]
        
        if i < num_images:
            try:
                if not os.path.exists(image_paths[i]):
                    raise FileNotFoundError
                
                img = mpimg.imread(image_paths[i])
                ax.imshow(img)
            except Exception as e:
                ax.text(0.5, 0.5, f"Error Loading:\n{os.path.basename(image_paths[i])}", 
                        ha='center', va='center')
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            # --- Extra Large Labeling ---
            if labels and i < len(labels):
                ax.set_title(labels[i], fontsize=28, fontweight='bold', pad=20)
            
            # --- High-Visibility Box ---
            # Thicker lines ensure borders don't disappear in compressed PDFs
            for spine in ax.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(3.5)
            
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            # Hide empty tiles if the image count isn't a multiple of 3
            ax.axis('off')

    # Exporting as PDF (DPI 300 for raster fallback components)
    plt.savefig(output_name, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[SUCCESS] Qualitative grid saved to: {output_name}")

if __name__ == "__main__":
    # Example usage with mock success/failure paths
    img_list = [
        'eval_results/success/000072.jpg', 'eval_results/success/000152.jpg', 
        'eval_results/success/000076.jpg', 'eval_results/success/000194.jpg', 
        'eval_results/failure/026193.jpg', 'eval_results/failure/015534.jpg'
    ]
    
    img_labels = [
        'Success: Layering', 'Success: Pose', 'Success: Viewpoint', 
        'Fail: Occlusion', 'Fail: Small Scale', 'Fail: Confusion'
    ]

    create_3_column_grid(img_list, "qualitative_grid_garment.png", labels=img_labels)