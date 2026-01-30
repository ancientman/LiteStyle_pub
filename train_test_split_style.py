import os
import shutil
from sklearn.model_selection import train_test_split

# Configuration
DATA_ROOT = './style_images'  # Root folder with 'formal' and 'casual' subfolders
TRAIN_DIR = os.path.join(DATA_ROOT, 'train')
VAL_DIR = os.path.join(DATA_ROOT, 'val')
VAL_SPLIT = 0.2  # 20% for validation

# Categories
categories = ['formal', 'casual']

# Create directories if they don't exist
for split in [TRAIN_DIR, VAL_DIR]:
    for cat in categories:
        os.makedirs(os.path.join(split, cat), exist_ok=True)

# Split for each category
for cat in categories:
    src_dir = os.path.join(DATA_ROOT, cat)
    if not os.path.exists(src_dir):
        print(f"Warning: {src_dir} not found.")
        continue
    
    # Get all image files
    images = [f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Category '{cat}': Found {len(images)} images.")
    
    # Split into train/val
    train_imgs, val_imgs = train_test_split(images, test_size=VAL_SPLIT, random_state=42)
    
    # Copy to new folders
    for img in train_imgs:
        shutil.copy(os.path.join(src_dir, img), os.path.join(TRAIN_DIR, cat, img))
    for img in val_imgs:
        shutil.copy(os.path.join(src_dir, img), os.path.join(VAL_DIR, cat, img))
    
    print(f"-> Train: {len(train_imgs)}, Val: {len(val_imgs)}")

print("Split complete! Run your training script with DATA_ROOT='./style_images/train' and add val extraction.")