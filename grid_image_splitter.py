"""
PURPOSE:
This script automates the process of splitting images containing a 4x4 regular grid into 16 individual images.
It scans a specific directory for image files, creates a subfolder named 'split_images', 
and saves the individual tiles there with a numbered suffix (e.g., image_1.jpg, image_2.jpg).
"""

import os
from PIL import Image

def split_grid_images(folder_path):
    # Define which file types the script should look for
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    
    # Safety check: Stop the script if the user provided an incorrect folder path
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return

    # Define the destination: a subfolder named 'split_images' inside the source directory
    output_folder = os.path.join(folder_path, "split_images")
    
    # If the subfolder doesn't exist yet, create it automatically
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created subfolder: {output_folder}")

    # Iterate through every file found in the provided folder
    for filename in os.listdir(folder_path):
        # Only process files that match our valid image extensions
        if filename.lower().endswith(valid_extensions):
            img_path = os.path.join(folder_path, filename)
            
            # Open the image file using Pillow
            with Image.open(img_path) as img:
                width, height = img.size
                
                # Divide the total dimensions by 4 to get the size of one individual cell
                # We use floor division (//) to ensure we have whole pixel integers
                cell_width = width // 4
                cell_height = height // 4
                
                # Index to track which tile we are on (1 through 16)
                count = 1
                
                # Nested loops: Outer loop handles rows (top to bottom), Inner handles columns (left to right)
                for row in range(4):
                    for col in range(4):
                        # Calculate the coordinates for the crop box (left, upper, right, lower)
                        left = col * cell_width
                        upper = row * cell_height
                        right = left + cell_width
                        lower = upper + cell_height
                        
                        # Crop the specific rectangle out of the original image
                        grid_cell = img.crop((left, upper, right, lower))
                        
                        # Split the filename and extension (e.g., 'photo' and '.jpg')
                        name_part, ext_part = os.path.splitext(filename)
                        
                        # Format the new name with the suffix (e.g., 'photo_1.jpg')
                        new_filename = f"{name_part}_{count}{ext_part}"
                        
                        # Construct the full destination path inside the subfolder
                        save_path = os.path.join(output_folder, new_filename)
                        
                        # Save the tile to the subfolder
                        grid_cell.save(save_path)
                        
                        # Increment the suffix number for the next tile
                        count += 1
            
            print(f"Successfully split: {filename}")

# --- Set your folder path here ---
# Ensure this path is correct relative to where you run this script
target_folder = './style_images/test'
split_grid_images(target_folder)