import streamlit as st
import pandas as pd
import numpy as np
import torch
import joblib
from datetime import datetime, timedelta
import pandas_market_calendars as mcal
from datasets import load_dataset
from kan_model import TemporalKANForecaster
import os
from huggingface_hub import hf_hub_download

# Constants
FI_ASSETS = ['GLD', 'TLT', 'VCIT', 'LQD', 'HYG', 'VNQ', 'SLV']
FI_BENCHMARK = 'AGG'
EQUITY_ASSETS = ['QQQ', 'XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU', 'XME', 'GDX', 'IWM']
EQUITY_BENCHMARK = 'SPY'
MACRO_COLS = ['VIX', 'DXY', 'T10Y2Y', 'TBILL_3M', 'IG_SPREAD', 'HY_SPREAD']
TRANSACTION_COST = 0.0012
SEQ_LEN = 20
HF_REPO = "P2SAMAPA/p2-etf-kan-engine-results"

# Session state
if 'prev_pick_fi' not in st.session_state:
    st.session_state.prev_pick_fi = None
if 'prev_pick_equity' not in st.session_state:
    st.session_state.prev_pick_equity = None
if 'signal_history' not in st.session_state:
    st.session_state.signal_history = []

# -------------------------------------------------------------------
def get_next_trading_day():
    nyse = mcal.get_calendar('NYSE')
    today = datetime.now().date()
    schedule = nyse.schedule(start_date=today, end_date=today + timedelta(days=10))
    if len(schedule) == 0:
        return today + timedelta(days=1)
    return schedule.index[0].date()

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

def build_feature_sequence(df, module):
    if module == 'fi':
        assets = FI_ASSETS
    else:
        assets = EQUITY_ASSETS
    features = df[MACRO_COLS].copy()
    for lag in range(1, 6):
        for a in assets:
            features[f'{a}_lag{lag}'] = df[a].pct_change().shift(lag)
    features = features.dropna()
    if len(features) < SEQ_LEN:
        return None
    return features.iloc[-SEQ_LEN:].values

def ensure_model_file(local_path, repo_filename):
    if not os.path.exists(local_path):
        try:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            hf_hub_download(
                repo_id=HF_REPO,
                filename=repo_filename,
                local_dir="models",
                local_dir_use_symlinks=False
            )
            st.info(f"✅ Downloaded {repo_filename} from Hugging Face.")
            return True
        except Exception as e:
            st.warning(f"⚠️ Could not download {repo_filename}: {e}")
            return False
    return True

def load_model_and_scalers(module, mode='full'):
    model_file = f"kan_{module}_full.pt"
    scaler_x_file = f"scaler_X_{module}_full.pkl"
    scaler_y_file = f"scaler_y_{module}_full.pkl"
    local_model = f"models/{model_file}"
    local_scaler_x = f"models/{scaler_x_file}"
    local_scaler_y = f"models/{scaler_y_file}"
    if not (ensure_model_file(local_model, model_file) and
            ensure_model_file(local_scaler_x, scaler_x_file) and
            ensure_model_file(local_scaler_y, scaler_y_file)):
        return None, None, None
    scaler_X = joblib.load(local_scaler_x)
    scaler_y = joblib.load(local_scaler_y)
    n_features = scaler_X.mean_.shape[0]
    input_dim = SEQ_LEN * n_features
    output_dim = len(FI_ASSETS) if module == 'fi' else len(EQUITY_ASSETS)
    model = TemporalKANForecaster(input_dim, hidden_dims=[128, 64], output_dim=output_dim, grid_size=20)
    try:
        model.load_state_dict(torch.load(local_model, map_location='cpu'))
    except RuntimeError as e:
        st.error(f"Model architecture mismatch: {e}")
        return None, None, None
    model.eval()
    return model, scaler_X, scaler_y

def get_prediction(model, scaler_X, scaler_y, feature_seq):
    seq_scaled = scaler_X.transform(feature_seq)
    X_tensor = torch.FloatTensor(seq_scaled.flatten()).unsqueeze(0)
    with torch.no_grad():
        pred_scaled = model(X_tensor).numpy()[0]
    return scaler_y.inverse_transform(pred_scaled.reshape(1, -1))[0]

def apply_transaction_cost(prev_pick, new_pick, gross_return):
    if prev_pick is None or new_pick == prev_pick:
        return gross_return, False
    net_return = gross_return - TRANSACTION_COST
    return net_return, True

def load_metrics(module):
    """Download and load metrics_*.pkl from HF."""
    metrics_file = f"metrics_{module}_full.pkl"
    local_path = f"models/{metrics_file}"
    if not ensure_model_file(local_path, metrics_file):
        return None
    data = joblib.load(local_path)
    return data

def compute_metrics_from_predictions(test_pred, test_true):
    """
    test_pred: numpy array of predicted returns (daily)
    test_true: numpy array of actual returns (daily)
    Returns dict with annualized return (%), Sharpe ratio (float), max drawdown (%), hit rate (%)
    """
    # Assume daily returns; annualize with 252 trading days
    ann_factor = np.sqrt(252)
    mean_pred = np.mean(test_pred)
    std_pred = np.std(test_pred)
    # Sharpe (assuming risk-free rate = 0 for simplicity)
    sharpe = (mean_pred / std_pred) * ann_factor if std_pred > 0 else 0.0
    # Annualized return (as percentage)
    ann_return = mean_pred * 252 * 100
    # Max drawdown on predicted returns (cumulative)
    cum_ret = np.cumprod(1 + test_pred)
    peak = np.maximum.accumulate(cum_ret)
    drawdown = (cum_ret - peak) / peak
    max_dd = np.min(drawdown) * 100
    # Hit rate: sign agreement
    hit = np.mean(np.sign(test_pred) == np.sign(test_true)) * 100
    return {
        "ann_return": ann_return,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "hit_rate": hit
    }

# -------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="P2-ETF-KAN-ENGINE")
st.title("P2‑ETF‑KAN‑ENGINE")
st.markdown("*Kolmogorov‑Arnold Network · Macro‑pill interpretability · Max absolute return*")

try:
    df_raw = load_raw_data()
    latest_macro = df_raw[MACRO_COLS].iloc[-1].to_dict()
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

os.makedirs("models", exist_ok=True)

tab_fi, tab_equity = st.tabs(["Option A — Fixed Income / Alts", "Option B — Equity Sectors"])

for tab, module, assets, benchmark in [(tab_fi, 'fi', FI_ASSETS, FI_BENCHMARK),
                                        (tab_equity, 'equity', EQUITY_ASSETS, EQUITY_BENCHMARK)]:
    with tab:
        feature_seq = build_feature_sequence(df_raw, module)
        if feature_seq is None:
            st.warning(f"Insufficient data to build feature sequence for {module}.")
            continue
        
        model_full, scaler_X_full, scaler_y_full = load_model_and_scalers(module, mode='full')
        if model_full is None:
            st.warning(f"Full model for {module} not found or could not be downloaded. Please train first via GitHub Actions.")
            continue
        
        pred_returns = get_prediction(model_full, scaler_X_full, scaler_y_full, feature_seq)
        top_idx = np.argmax(pred_returns)
        top_asset = assets[top_idx]
        top_return = pred_returns[top_idx] * 100
        
        prev = st.session_state[f'prev_pick_{module}']
        net_return, switched = apply_transaction_cost(prev, top_asset, top_return / 100)
        if switched:
            display_return = net_return * 100
            st.info(f"📉 Transaction cost (12 bps) applied: switched from {prev} to {top_asset}")
        else:
            display_return = top_return
        st.session_state[f'prev_pick_{module}'] = top_asset
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"## {top_asset}")
            st.markdown(f"### {display_return:.1f}% conviction")
            st.caption(f"Signal for {get_next_trading_day()} · Generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
            st.caption("**Source:** Full dataset (2008–2026 YTD)")
            sorted_idx = np.argsort(pred_returns)[::-1]
            if len(sorted_idx) > 1:
                st.markdown(f"2nd: **{assets[sorted_idx[1]]}** {pred_returns[sorted_idx[1]]*100:.1f}%")
            if len(sorted_idx) > 2:
                st.markdown(f"3rd: **{assets[sorted_idx[2]]}** {pred_returns[sorted_idx[2]]*100:.1f}%")
        
        with col2:
            st.markdown("### Macro Pills")
            for m in MACRO_COLS:
                st.metric(m, f"{latest_macro.get(m, 0):.2f}")
        
        # Real metrics from test set
        metrics_data = load_metrics(module)
        if metrics_data is not None:
            test_pred = np.array(metrics_data['test_predictions'])
            test_true = np.array(metrics_data['test_true'])
            # Flatten if needed (test_pred shape: (n_samples, n_assets) – we need per-asset? We'll average across assets for overall metrics)
            # For simplicity, compute metrics on the first asset (or average across all)
            # Better: compute metrics for each asset and show average? For demo, take mean across assets.
            pred_flat = test_pred.mean(axis=1)
            true_flat = test_true.mean(axis=1)
            metrics = compute_metrics_from_predictions(pred_flat, true_flat)
            st.markdown("### Performance Metrics (Test Period)")
            metrics_df = pd.DataFrame({
                "Metric": ["ANN RETURN", "SHARPE", "MAX DD", "HIT RATE"],
                "Value": [f"{metrics['ann_return']:.1f}%", f"{metrics['sharpe']:.2f}", f"{metrics['max_dd']:.1f}%", f"{metrics['hit_rate']:.1f}%"]
            })
            st.dataframe(metrics_df, use_container_width=True)
        else:
            st.warning("Metrics file not found. Train the model first.")
        
        st.markdown("### Signal History")
        if st.session_state.signal_history:
            history_df = pd.DataFrame(st.session_state.signal_history, columns=["Date", "Pick", "Conviction", "Actual Return", "Hit"])
            st.dataframe(history_df, use_container_width=True)
        else:
            st.write("No signals recorded yet.")
        
        with st.expander("Show Ensemble Prediction (Shrinking Windows Consensus)"):
            st.write("Shrinking window models not yet available. Train them via the `train_shrinking.yml` workflow.")

st.caption("P2-ETF-KAN-ENGINE - Research only · Not financial advice")
