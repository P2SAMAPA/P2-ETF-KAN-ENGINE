import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from sklearn.preprocessing import StandardScaler
from kan_model import TemporalKANForecaster
import joblib

# ... (constants and helper functions same as before) ...

def train_full(module, epochs=300, seq_len=20, batch_size=256, lr=1e-4, patience=100):
    print("Loading raw data...")
    df = load_raw_data()
    assets = FI_ASSETS if module == 'fi' else EQUITY_ASSETS
    X, y, feat_names, target_names = create_features_and_targets(df, assets, seq_len)
    n = len(X)
    print(f"Total samples: {n}")
    print(f"y mean: {y.mean():.6f}, y std: {y.std():.6f}")
    # ... (splitting and scaling same as before) ...
    
    # Model with ReLUKAN, larger grid, more capacity
    model = TemporalKANForecaster(input_dim, hidden_dims=[256, 128, 64], output_dim=len(assets), grid_size=20)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_epoch = 0
    epochs_no_improve = 0
    
    print(f"Training {module} module...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            # Gradient clipping to stabilize training
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val_t), y_val_t).item()
            pred_sample = model(X_val_t[:32])
            pred_var = pred_sample.var().item()
        scheduler.step()
        
        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f} | Pred Var: {pred_var:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), f"models/kan_{module}_full.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # ... (save scalers, test evaluation, etc.) ...
