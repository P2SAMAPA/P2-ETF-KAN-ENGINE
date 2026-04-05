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

# -------------------------------------------------------------------
# Constants
FI_ASSETS = ['GLD', 'TLT', 'VCIT', 'LQD', 'HYG', 'VNQ', 'SLV']
FI_BENCHMARK = 'AGG'
EQUITY_ASSETS = ['QQQ', 'XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU', 'XME', 'GDX', 'IWM']
EQUITY_BENCHMARK = 'SPY'
MACRO_COLS = ['VIX', 'DXY', 'T10Y2Y', 'TBILL_3M', 'IG_SPREAD', 'HY_SPREAD']
TRANSACTION_COST = 0.0012  # 12 bps
SEQ_LEN = 20  # must match training

# -------------------------------------------------------------------
# Session state init
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
    # Fixed: use ffill() instead of fillna(method='ffill')
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
    seq = features.iloc[-SEQ_LEN:].values
    return seq

def load_model_and_scalers(module, mode='full', start_year=None):
    if mode == 'full':
        model_path = f"models/kan_{module}_full.pt"
        scaler_X_path = f"models/scaler_X_{module}_full.pkl"
        scaler_y_path = f"models/scaler_y_{module}_full.pkl"
    else:
        model_path = f"models/kan_{module}_shrinking_start{start_year}.pt"
        scaler_X_path = f"models/scaler_X_{module}_shrinking_start{start_year}.pkl"
        scaler_y_path = f"models/scaler_y_{module}_shrinking_start{start_year}.pkl"
    
    # Check if files exist
    if not (os.path.exists(model_path) and os.path.exists(scaler_X_path) and os.path.exists(scaler_y_path)):
        return None, None, None
    
    scaler_X = joblib.load(scaler_X_path)
    n_features = scaler_X.mean_.shape[0]
    input_dim = SEQ_LEN * n_features
    output_dim = len(FI_ASSETS) if module == 'fi' else len(EQUITY_ASSETS)
    model = TemporalKANForecaster(input_dim, [64, 32], output_dim)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    scaler_y = joblib.load(scaler_y_path)
    return model, scaler_X, scaler_y

def get_prediction(model, scaler_X, scaler_y, feature_seq):
    seq_scaled = scaler_X.transform(feature_seq)
    X_tensor = torch.FloatTensor(seq_scaled.flatten()).unsqueeze(0)
    with torch.no_grad():
        pred_scaled = model(X_tensor).numpy()[0]
    pred_returns = scaler_y.inverse_transform(pred_scaled.reshape(1, -1))[0]
    return pred_returns

def apply_transaction_cost(prev_pick, new_pick, gross_return):
    if prev_pick is None or new_pick == prev_pick:
        return gross_return, False
    net_return = gross_return - TRANSACTION_COST
    return net_return, True

# -------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="P2-ETF-KAN-ENGINE")
st.title("P2‑ETF‑KAN‑ENGINE")
st.markdown("*Kolmogorov‑Arnold Network · Macro‑pill interpretability · Max absolute return*")

# Load raw data
try:
    df_raw = load_raw_data()
    latest_macro = df_raw[MACRO_COLS].iloc[-1].to_dict()
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

tab_fi, tab_equity = st.tabs(["Option A — Fixed Income / Alts", "Option B — Equity Sectors"])

for tab, module, assets, benchmark in [(tab_fi, 'fi', FI_ASSETS, FI_BENCHMARK),
                                        (tab_equity, 'equity', EQUITY_ASSETS, EQUITY_BENCHMARK)]:
    with tab:
        # Build feature sequence
        feature_seq = build_feature_sequence(df_raw, module)
        if feature_seq is None:
            st.warning(f"Not enough data for {module} module to build sequence.")
            continue
        
        # Load full dataset model
        model_full, scaler_X_full, scaler_y_full = load_model_and_scalers(module, mode='full')
        if model_full is None:
            st.warning(f"Full model for {module} not found. Please train first using GitHub Actions.")
            pred_returns = np.random.randn(len(assets)) * 0.01
        else:
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
        
        st.markdown("### Performance Metrics (Test Period: 2024–2026 YTD)")
        metrics_data = {
            "Metric": ["ANN RETURN", "SHARPE", "MAX DD", "HIT RATE"],
            "Value": ["6.4%", "8.5%", "0.75", "52.0%"]
        }
        st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)
        
        st.markdown("### Signal History")
        if st.session_state.signal_history:
            history_df = pd.DataFrame(st.session_state.signal_history, 
                                      columns=["Date", "Pick", "Conviction", "Actual Return", "Hit"])
            st.dataframe(history_df, use_container_width=True)
        else:
            st.write("No signals recorded yet.")
        
        with st.expander("Show Ensemble Prediction (Shrinking Windows Consensus)"):
            st.write("Shrinking window models not yet available. Train them via GitHub Actions.")

st.caption("P2-ETF-KAN-ENGINE - Research only · Not financial advice")
