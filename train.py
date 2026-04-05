import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from kan_model import TemporalKANForecaster

# -------------------------------------------------------------------
# ETF definitions
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
    # Fill missing VCIT etc.
    all_cols = FI_ASSETS + EQUITY_ASSETS + MACRO_COLS
    for col in all_cols:
        if col not in df.columns:
            df[col] = np.nan
    df.fillna(method='ffill', inplace=True)
    df.dropna(inplace=True)
    return df

def prepare_module_data(df, module='fi', seq_len=20):
    if module == 'fi':
        assets = FI_ASSETS
    else:
        assets = EQUITY_ASSETS
    
    # target: next day return
    returns = df[assets].pct_change().shift(-1)
    # features: macro + lagged returns of assets
    features = df[MACRO_COLS].copy()
    for lag in range(1, 6):
        for a in assets:
            features[f'{a}_lag{lag}'] = df[a].pct_change().shift(lag)
    
    data = pd.concat([returns, features], axis=1).dropna()
    X = data[features.columns].values
    y = data[assets].values
    
    # Scale
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    # Create sequences
    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - seq_len):
        X_seq.append(X_scaled[i:i+seq_len])
        y_seq.append(y_scaled[i+seq_len])
    return np.array(X_seq), np.array(y_seq), scaler_X, scaler_y, features.columns.tolist(), assets

def train_full(module, epochs=150, seq_len=20, batch_size=64, lr=1e-3):
    df = load_raw_data()
    X, y, scaler_X, scaler_y, feat_names, target_names = prepare_module_data(df, module, seq_len)
    
    # 80/10/10 chronological split
    n = len(X)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val)
    
    input_dim = seq_len * X.shape[2]
    model = TemporalKANForecaster(input_dim, hidden_dims=[64,32], output_dim=len(target_names))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    best_val_loss = float('inf')
    for epoch in tqdm(range(epochs)):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = loss_fn(pred, y_train_t)
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val_t), y_val_t)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"models/kan_{module}_full.pt")
    
    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        test_pred = model(torch.FloatTensor(X_test)).numpy()
    test_true = y_test
    
    # Save scalers
    import joblib
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler_X, f'models/scaler_X_{module}_full.pkl')
    joblib.dump(scaler_y, f'models/scaler_y_{module}_full.pkl')
    
    # Store results
    results = {
        'test_predictions': test_pred.tolist(),
        'test_true': test_true.tolist(),
        'feature_names': feat_names,
        'target_names': target_names,
        'best_val_loss': best_val_loss
    }
    with open(f'metrics_{module}_full.json', 'w') as f:
        json.dump(results, f)
    print(f"Full model for {module} done. Test predictions saved.")

def train_shrinking(module, start_year, epochs=150, seq_len=20):
    df = load_raw_data()
    # Filter data from start_year to 2026 YTD (today's year)
    current_year = pd.Timestamp.now().year
    df = df[df.index >= f'{start_year}-01-01']
    df = df[df.index <= f'{current_year}-12-31']  # up to end of current year
    
    X, y, scaler_X, scaler_y, feat_names, target_names = prepare_module_data(df, module, seq_len)
    
    # 80/10/10 split on this window
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
    X_test_t = torch.FloatTensor(X_test)
    
    input_dim = seq_len * X.shape[2]
    model = TemporalKANForecaster(input_dim, hidden_dims=[64,32], output_dim=len(target_names))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    
    best_val_loss = float('inf')
    for epoch in tqdm(range(epochs)):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(X_train_t), y_train_t)
        loss.backward()
        optimizer.step()
        model.eval()
        val_loss = loss_fn(model(X_val_t), y_val_t)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"models/kan_{module}_shrinking_start{start_year}.pt")
    
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test_t).numpy()
    test_true = y_test
    
    import joblib
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler_X, f'models/scaler_X_{module}_shrinking_start{start_year}.pkl')
    joblib.dump(scaler_y, f'models/scaler_y_{module}_shrinking_start{start_year}.pkl')
    
    # Compute per-ETF metrics over test period
    # Denormalize predictions and true values (need scaler_y)
    # For now store raw scaled values; later upload_model will compute metrics
    results = {
        'start_year': start_year,
        'test_predictions': test_pred.tolist(),
        'test_true': test_true.tolist(),
        'feature_names': feat_names,
        'target_names': target_names,
        'best_val_loss': best_val_loss
    }
    with open(f'metrics_{module}_shrinking_start{start_year}.json', 'w') as f:
        json.dump(results, f)
    print(f"Shrinking window start={start_year}, module={module} done.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['full', 'shrinking'], required=True)
    parser.add_argument('--module', choices=['fi', 'equity'], required=True)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--start-year', type=int, help='only for shrinking mode')
    args = parser.parse_args()
    
    if args.mode == 'full':
        train_full(args.module, epochs=args.epochs)
    else:
        if not args.start_year:
            raise ValueError("--start-year required for shrinking mode")
        train_shrinking(args.module, args.start_year, epochs=args.epochs)
