"""
Apparel Style Classification Trainer
====================================
Version: 3.0 
Date: January 30, 2026

This script trains a binary StyleMLP-v3 classifier using a pre-trained YOLOv11n 
backbone as a frozen feature extractor. It captures high-level garment features 
via forward hooks, applies training augmentations, and optimizes a 3-layer 
MLP to distinguish between 'Casual' and 'Formal' attire.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from ultralytics import YOLO
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import transforms
from PIL import Image

# --- CONFIGURATION ---
TRAIN_ROOT = './style_images/train'
VAL_ROOT = './style_images/val'
YOLO_MODEL_PATH = 'best.pt'
SAVE_PATH = 'style_model_v3.pth'
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
PATIENCE = 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data Augmentation Strategy

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.ToPILImage()
])

# 1. ARCHITECTURE: StyleMLP

class StyleMLP(nn.Module):
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

# 2. FEATURE HOOK LOGIC

features_buffer = []

def hook_fn(module, input, output):
    if isinstance(output, tuple):
        output = output[0]
    # Global Average Pooling (GAP) reduces spatial dimensions to 1x1 vectors
    gap = F.adaptive_avg_pool2d(output, (1, 1))
    features_buffer.append(gap.view(gap.size(0), -1).detach().cpu())

# 3. DATA EXTRACTION ENGINE
def get_dataset(yolo_model, root_dir, augment=False):
    X_list, y_list = [], []
    categories = {'casual': 0, 'formal': 1}
    
    print(f"--- EXTRACTING FEATURES FROM {root_dir} (Augment: {augment}) ---")
    for label_name, label_idx in categories.items():
        folder_path = os.path.join(root_dir, label_name)
        if not os.path.exists(folder_path):
            continue
        
        image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for i in tqdm(range(0, len(image_paths), BATCH_SIZE)):
            batch_paths = image_paths[i:i + BATCH_SIZE]
            batch_images = []
            
            for path in batch_paths:
                img = Image.open(path).convert('RGB')
                if augment:
                    img = train_transform(img)
                batch_images.append(img)
            
            features_buffer.clear()
            yolo_model.predict(batch_images, imgsz=640, device=DEVICE, verbose=False)
            
            if features_buffer:
                batch_features = torch.cat(features_buffer, dim=0)
                X_list.append(batch_features)
                y_list.append(torch.full((batch_features.size(0), 1), float(label_idx), dtype=torch.float32))
    
    return torch.cat(X_list), torch.cat(y_list)

# 4. TRAINING PIPELINE
def train():
    print(f"Running on: {DEVICE}")
    yolo = YOLO(YOLO_MODEL_PATH).to(DEVICE)
    
    # Layer 9 is usually the SPPF (Spatial Pyramid Pooling - Fast) layer in YOLOv11
    target_layer_idx = 9 if len(yolo.model.model) > 10 else (len(yolo.model.model) - 2)
    yolo.model.model[target_layer_idx].register_forward_hook(hook_fn)
    print(f"✓ Hook on Layer {target_layer_idx}")

    # Feature Extraction Phase
    X_train, y_train = get_dataset(yolo, TRAIN_ROOT, augment=True)
    X_val, y_val = get_dataset(yolo, VAL_ROOT, augment=False)
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    # Optimization Setup
    mlp = StyleMLP(input_dim=X_train.shape[1]).to(DEVICE)
    optimizer = optim.Adam(mlp.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.BCELoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"\n--- TRAINING ({EPOCHS} Epochs) ---")
    for epoch in range(EPOCHS):
        mlp.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            preds = mlp(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation Phase
        mlp.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                preds = mlp(batch_x)
                loss = criterion(preds, batch_y)
                val_loss += loss.item()
                predicted = (preds > 0.5).float()
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Early Stopping & Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(mlp.state_dict(), SAVE_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break
    
    print(f"✓ Model saved to {SAVE_PATH}")

if __name__ == "__main__":
    train()