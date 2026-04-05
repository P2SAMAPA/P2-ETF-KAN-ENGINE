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

FI_ASSETS = ['GLD', 'TLT', 'VCIT', 'LQD', 'HYG', 'VNQ', 'SLV']
EQUITY_ASSETS = ['QQQ', 'XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU', 'XME', 'GDX', 'IWM']
MACRO_COLS = ['VIX', 'DXY', 'T10Y2Y', 'TBILL_3M', 'IG_SPREAD', 'HY_SPREAD']

def load_raw_data():
    ds = load_dataset("P2SAMAPA/fi-etf-macro-signal-master-data", split="train")
    df = ds.to_pandas()
    df['date'] = pd.to_datetime(df['__index_level_0__'], unit='s')
    df.set_index('date', inplace=True)
    df.drop('__index_level_0__', axis=1, inplace=True)
    df.sort_index(inplace=True)
    all_cols = FI_ASSETS + EQUITY_ASSETS + MACRO_COLS
    for col in all_cols:
        if col not in df.columns:
            df[col] = np.nan
    df.ffill(inplace=True)
    df.dropna(inplace=True)
    return df

def create_features_and_targets(df, assets, seq_len=20):
    returns = df[assets].pct_change().shift(-1)
    features = df[MACRO_COLS].copy()
    for lag in range(1, 6):
        for a in assets:
            features[f'{a}_lag{lag}'] = df[a].pct_change().shift(lag)
    data = pd.concat([returns, features], axis=1).dropna()
    X = data[features.columns].values
    y = data[assets].values
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i+seq_len])
        y_seq.append(y[i+seq_len])
    return np.array(X_seq), np.array(y_seq), features.columns.tolist(), assets

def train_full(module, epochs=300, seq_len=20, batch_size=256, lr=1e-4, patience=100):
    print("Loading raw data...")
    df = load_raw_data()
    assets = FI_ASSETS if module == 'fi' else EQUITY_ASSETS
    X, y, feat_names, target_names = create_features_and_targets(df, assets, seq_len)
    n = len(X)
    print(f"Total samples: {n}")
    print(f"y mean: {y.mean():.6f}, y std: {y.std():.6f}")
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)
    X_train_raw, y_train_raw = X[:train_end], y[:train_end]
    X_val_raw, y_val_raw = X[train_end:val_end], y[train_end:val_end]
    X_test_raw, y_test_raw = X[val_end:], y[val_end:]
    # Scaling
    train_flat = X_train_raw.reshape(-1, X_train_raw.shape[-1])
    val_flat = X_val_raw.reshape(-1, X_val_raw.shape[-1])
    test_flat = X_test_raw.reshape(-1, X_test_raw.shape[-1])
    scaler_X = StandardScaler()
    scaler_X.fit(train_flat)
    X_train_scaled = scaler_X.transform(train_flat).reshape(X_train_raw.shape)
    X_val_scaled = scaler_X.transform(val_flat).reshape(X_val_raw.shape)
    X_test_scaled = scaler_X.transform(test_flat).reshape(X_test_raw.shape)
    scaler_y = StandardScaler()
    scaler_y.fit(y_train_raw)
    y_train_scaled = scaler_y.transform(y_train_raw)
    y_val_scaled = scaler_y.transform(y_val_raw)
    y_test_scaled = scaler_y.transform(y_test_raw)
    X_train_t = torch.FloatTensor(X_train_scaled)
    y_train_t = torch.FloatTensor(y_train_scaled)
    X_val_t = torch.FloatTensor(X_val_scaled)
    y_val_t = torch.FloatTensor(y_val_scaled)
    X_test_t = torch.FloatTensor(X_test_scaled)
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    input_dim = seq_len * X_train_scaled.shape[-1]
    # ReLUKAN model
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
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler_X, f'models/scaler_X_{module}_full.pkl')
    joblib.dump(scaler_y, f'models/scaler_y_{module}_full.pkl')
    model.load_state_dict(torch.load(f"models/kan_{module}_full.pt"))
    model.eval()
    with torch.no_grad():
        test_pred_scaled = model(X_test_t).numpy()
    test_pred = scaler_y.inverse_transform(test_pred_scaled)
    results = {
        'test_predictions': test_pred.tolist(),
        'feature_names': feat_names,
        'target_names': target_names,
        'best_val_loss': float(best_val_loss),
        'best_epoch': best_epoch
    }
    joblib.dump(results, f'metrics_{module}_full.pkl')
    print(f"Full model for {module} done. Best val loss: {best_val_loss:.6f} at epoch {best_epoch+1}")
    print(f"Test prediction mean: {test_pred.mean():.6f}, std: {test_pred.std():.6f}")

def train_shrinking(module, start_year, epochs=300, seq_len=20, batch_size=256, lr=1e-4, patience=100):
    # Same structure as train_full, omitted for brevity but should be identical
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['full', 'shrinking'], required=True)
    parser.add_argument('--module', choices=['fi', 'equity'], required=True)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--start-year', type=int)
    args = parser.parse_args()
    if args.mode == 'full':
        train_full(args.module, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, patience=args.patience)
    else:
        if not args.start_year:
            raise ValueError("--start-year required")
        # train_shrinking(...) – implement similarly if needed
        pass
