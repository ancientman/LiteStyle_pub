"""
Apparel Qualitative Interpretability Grid Generator
===================================================
Version: 1.8
Date: January 30, 2026

This script performs an architectural ablation study on the StyleMLP-v3 classifier.
It evaluates the impact of BatchNorm, Dropout, and Layer Depth on "Formality" 
classification using 256-dimensional features extracted from YOLOv11n.
"""

import torch  # Core deep learning framework for tensors and autograd
import torch.nn as nn  # Neural network layers and utilities
from torch.utils.data import DataLoader, TensorDataset  # Efficient batching and dataset handling
from torch.optim import Adam  # Adaptive optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau  # Learning rate scheduler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support  # Evaluation metrics
import numpy as np  # Numerical operations
import pandas as pd  # Data manipulation and CSV export

# Device configuration: CPU for portability and quick ablation; change to 'cuda' if GPU available
device = 'cpu'

# Hyperparameters
BATCH_SIZE = 32  # Suitable batch size for CPU training and memory efficiency
EPOCHS = 50      # Sufficient epochs for convergence on the extracted feature set (~5285 training samples)

# Load pre-computed 256-dimensional features extracted from YOLOv11n SPPF layer (after GAP)
train_data = torch.load('features_train.pt')  # Large training split (~5285 samples)
X_train = train_data['X'].to(device)         # Feature matrix
y_train = train_data['y'].float().unsqueeze(1).to(device)  # Binary labels reshaped to (N, 1) for BCELoss

val_data = torch.load('features_val.pt')     # Validation split (~1323 samples)
X_val = val_data['X'].to(device)
y_val = val_data['y'].float().unsqueeze(1).to(device)

test_data = torch.load('features_test.pt')   # Independent consumer test set
X_test = test_data['X'].to(device)
y_test = test_data['y'].float().unsqueeze(1).to(device)

# Create DataLoaders for efficient mini-batch processing
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE)

# Define the configurable StyleMLP classifier
class StyleMLP(nn.Module):
    """
    Lightweight multi-layer perceptron for formality scoring.
    Input: 256-dim feature vector from YOLOv11n SPPF + GAP.
    Output: Sigmoid probability (formal if ≥ 0.5).
    """
    def __init__(self, input_dim=256, use_bn=True, use_dropout=True, layers=3):
        super().__init__()
        modules = []  # Sequential module list
        # Determine hidden layer sizes based on depth
        sizes = [256, 128, 64, 1] if layers == 3 else [256, 64, 1]
        
        # Construct layers dynamically
        for i in range(len(sizes) - 1):
            modules.append(nn.Linear(sizes[i], sizes[i + 1]))  # Linear transformation
            if use_bn and i < len(sizes) - 2:                 # BatchNorm on hidden layers if enabled
                modules.append(nn.BatchNorm1d(sizes[i + 1]))
            if i < len(sizes) - 2:                            # ReLU and optional Dropout only on hidden layers
                modules.append(nn.ReLU(inplace=True))
                if use_dropout:
                    modules.append(nn.Dropout(0.3 if i == 0 else 0.2))  # Higher dropout after first hidden layer
        
        modules.append(nn.Sigmoid())  # Final sigmoid activation for probability output
        self.net = nn.Sequential(*modules)  # Assemble sequential network

    def forward(self, x):
        return self.net(x)  # Forward pass

# Ablation variants: Dictionary of architectural configurations
variants = {
    'baseline':    {'use_bn': True,  'use_dropout': True,  'layers': 3},  # Full reference model
    'no_bn':       {'use_bn': False, 'use_dropout': True,  'layers': 3},  # Ablate BatchNorm
    'no_dropout':  {'use_bn': True,  'use_dropout': False, 'layers': 3},  # Ablate Dropout
    '2_layer':     {'use_bn': True,  'use_dropout': True,  'layers': 2}   # Reduce depth
}

# Storage for final test metrics per variant
results = {}

# Binary Cross-Entropy loss for probability regression
criterion = nn.BCELoss()

# Main ablation loop
for name, config in variants.items():
    print(f"Training variant: {name}...")
    
    # Instantiate model with current configuration
    model = StyleMLP(**config).to(device)
    
    # Optimizer and scheduler (maximize validation AUC)
    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, patience=5, mode='max', verbose=False)
    
    best_auc = 0.0       # Track best validation AUC
    patience_cnt = 0     # Early stopping counter
    
    # Training loop
    for epoch in range(EPOCHS):
        model.train()  # Training mode (enables Dropout/BN statistics)
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()          # Clear gradients
            preds = model(batch_x)         # Forward pass
            loss = criterion(preds, batch_y)  # Compute loss
            loss.backward()                # Backpropagation
            optimizer.step()               # Parameter update
        
        # Validation phase
        model.eval()  # Evaluation mode
        with torch.no_grad():
            val_probs = model(X_val).cpu().numpy().flatten()  # Predicted probabilities
            val_auc = roc_auc_score(y_val.cpu(), val_probs)    # AUC-ROC on validation
        
        scheduler.step(val_auc)  # Adjust LR based on AUC plateau
        
        # Model checkpointing
        if val_auc > best_auc + 1e-4:
            best_auc = val_auc
            torch.save(model.state_dict(), f'style_{name}.pth')  # Save best weights
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= 10:  # Early stopping
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break
    
    # Final evaluation on held-out consumer test set
    model.load_state_dict(torch.load(f'style_{name}.pth'))
    model.eval()
    with torch.no_grad():
        test_probs = model(X_test).cpu().numpy().flatten()
        test_pred = (test_probs > 0.5).astype(int)  # Binary predictions
        
        acc = accuracy_score(y_test.cpu(), test_pred)
        auc = roc_auc_score(y_test.cpu(), test_probs)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test.cpu(), test_pred, average='binary', zero_division=0
        )
    
    # Store rounded metrics for manuscript table
    results[name] = {
        'acc': round(acc, 3),
        'auc': round(auc, 3),
        'prec': round(prec, 3),
        'rec': round(rec, 3),
        'f1': round(f1, 3)
    }
    print(f"{name} completed: Acc {acc:.3f}, AUC {auc:.3f}")

# Export consolidated results for inclusion in the Applied Sciences manuscript
pd.DataFrame(results).T.to_csv('ablation_results.csv')
print("\nAblation study complete. Results saved to 'ablation_results.csv'.")