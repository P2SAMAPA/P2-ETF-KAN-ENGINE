import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from kan_model import TemporalKANForecaster
import joblib

# -------------------------------------------------------------------
FI_ASSETS = ['GLD', 'TLT', 'VCIT', 'LQD', 'HYG', 'VNQ', 'SLV']
FI_BENCHMARK = 'AGG'
EQUITY_ASSETS = ['QQQ', 'XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU', 'XME', 'GDX', 'IWM']
EQUITY_BENCHMARK = 'SPY'
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

def prepare_module_data(df, module='fi', seq_len=20):
    if module == 'fi':
        assets = FI_ASSETS
    else:
        assets = EQUITY_ASSETS
    returns = df[assets].pct_change().shift(-1)
    features = df[MACRO_COLS].copy()
    for lag in range(1, 6):
        for a in assets:
            features[f'{a}_lag{lag}'] = df[a].pct_change().shift(lag)
    data = pd.concat([returns, features], axis=1).dropna()
    X = data[features.columns].values
    y = data[assets].values
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - seq_len):
        X_seq.append(X_scaled[i:i+seq_len])
        y_seq.append(y_scaled[i+seq_len])
    return np.array(X_seq), np.array(y_seq), scaler_X, scaler_y, features.columns.tolist(), assets

def train_full(module, epochs=150, seq_len=20, batch_size=64, lr=1e-3):
    print("Loading data...")
    df = load_raw_data()
    X, y, scaler_X, scaler_y, feat_names, target_names = prepare_module_data(df, module, seq_len)
    n = len(X)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val)
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    input_dim = seq_len * X.shape[2]
    model = TemporalKANForecaster(input_dim, hidden_dims=[64,32], output_dim=len(target_names))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    best_val_loss = float('inf')
    print(f"Starting training for {module} module...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val_t), y_val_t).item()
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), f"models/kan_{module}_full.pt")
            print(f"  -> New best model saved (val_loss={val_loss:.6f})")
    
    # Save scalers
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler_X, f'models/scaler_X_{module}_full.pkl')
    joblib.dump(scaler_y, f'models/scaler_y_{module}_full.pkl')
    
    # Final test evaluation
    model.eval()
    with torch.no_grad():
        test_pred = model(torch.FloatTensor(X_test)).numpy()
    test_true = y_test
    results = {
        'test_predictions': test_pred.tolist(),
        'test_true': test_true.tolist(),
        'feature_names': feat_names,
        'target_names': target_names,
        'best_val_loss': float(best_val_loss)
    }
    joblib.dump(results, f'metrics_{module}_full.pkl')
    print(f"Full model for {module} done. Best val loss: {best_val_loss:.6f}")

def train_shrinking(module, start_year, epochs=150, seq_len=20, batch_size=64, lr=1e-3):
    print(f"Shrinking window start={start_year} for {module}...")
    df = load_raw_data()
    current_year = pd.Timestamp.now().year
    df = df[df.index >= f'{start_year}-01-01']
    df = df[df.index <= f'{current_year}-12-31']
    X, y, scaler_X, scaler_y, feat_names, target_names = prepare_module_data(df, module, seq_len)
    n = len(X)
    if n < 100:
        print(f"  -> Not enough samples ({n}), skipping.")
        return
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val)
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    input_dim = seq_len * X.shape[2]
    model = TemporalKANForecaster(input_dim, hidden_dims=[64,32], output_dim=len(target_names))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val_t), y_val_t).item()
        
        if (epoch+1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), f"models/kan_{module}_shrinking_start{start_year}.pt")
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler_X, f'models/scaler_X_{module}_shrinking_start{start_year}.pkl')
    joblib.dump(scaler_y, f'models/scaler_y_{module}_shrinking_start{start_year}.pkl')
    
    model.eval()
    with torch.no_grad():
        test_pred = model(torch.FloatTensor(X_test)).numpy()
    test_true = y_test
    results = {
        'start_year': start_year,
        'test_predictions': test_pred.tolist(),
        'test_true': test_true.tolist(),
        'feature_names': feat_names,
        'target_names': target_names,
        'best_val_loss': float(best_val_loss)
    }
    joblib.dump(results, f'metrics_{module}_shrinking_start{start_year}.pkl')
    print(f"  -> Done. Best val loss: {best_val_loss:.6f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['full', 'shrinking'], required=True)
    parser.add_argument('--module', choices=['fi', 'equity'], required=True)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--start-year', type=int, help='only for shrinking mode')
    args = parser.parse_args()
    
    if args.mode == 'full':
        train_full(args.module, epochs=args.epochs, batch_size=args.batch_size)
    else:
        if not args.start_year:
            raise ValueError("--start-year required for shrinking mode")
        train_shrinking(args.module, args.start_year, epochs=args.epochs, batch_size=args.batch_size)
