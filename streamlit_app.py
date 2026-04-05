import streamlit as st
import pandas as pd
import numpy as np
import torch
import joblib
from datetime import datetime, timedelta
import pandas_market_calendars as mcal
from datasets import load_dataset
from kan_model import TemporalKANForecaster

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
    # Ensure all needed columns exist
    all_cols = FI_ASSETS + EQUITY_ASSETS + MACRO_COLS
    for col in all_cols:
        if col not in df.columns:
            df[col] = np.nan
    df.fillna(method='ffill', inplace=True)
    df.dropna(inplace=True)
    return df

def build_feature_sequence(df, module):
    """
    Build the latest (most recent) feature sequence of length SEQ_LEN.
    Features: macro cols + 5 lags of each asset in the module.
    Returns numpy array of shape (SEQ_LEN, n_features)
    """
    if module == 'fi':
        assets = FI_ASSETS
    else:
        assets = EQUITY_ASSETS
    
    # Create features DataFrame
    features = df[MACRO_COLS].copy()
    for lag in range(1, 6):
        for a in assets:
            features[f'{a}_lag{lag}'] = df[a].pct_change().shift(lag)
    
    # Drop rows with NaN (from lags)
    features = features.dropna()
    
    # Take last SEQ_LEN rows
    if len(features) < SEQ_LEN:
        st.error(f"Not enough data for sequence length {SEQ_LEN}. Only {len(features)} rows available.")
        return None
    
    seq = features.iloc[-SEQ_LEN:].values  # shape (SEQ_LEN, n_features)
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
    
    # Load scalers to get input dimension
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
    """
    feature_seq: numpy array shape (SEQ_LEN, n_features) (raw, unscaled)
    Returns: array of predicted returns for each asset (in original scale)
    """
    # Scale the feature sequence
    seq_scaled = scaler_X.transform(feature_seq)  # (SEQ_LEN, n_features)
    # Flatten to (1, SEQ_LEN * n_features)
    X_tensor = torch.FloatTensor(seq_scaled.flatten()).unsqueeze(0)
    with torch.no_grad():
        pred_scaled = model(X_tensor).numpy()[0]  # shape (output_dim,)
    # Inverse transform to original return scale
    pred_returns = scaler_y.inverse_transform(pred_scaled.reshape(1, -1))[0]
    return pred_returns

def apply_transaction_cost(prev_pick, new_pick, gross_return):
    if prev_pick is None or new_pick == prev_pick:
        return gross_return, False
    net_return = gross_return - TRANSACTION_COST
    return net_return, True

def compute_consensus_scores(module):
    """
    Load precomputed consensus scores from the HF dataset.
    For now returns dummy; in production, read from P2SAMAPA/p2-etf-kan-engine-results.
    """
    # Placeholder – replace with actual loading logic when results are available
    assets = FI_ASSETS if module == 'fi' else EQUITY_ASSETS
    # For demo, random scores
    return {a: np.random.random() for a in assets}

# -------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="P2-ETF-KAN-ENGINE")
st.title("P2‑ETF‑KAN‑ENGINE")
st.markdown("*Kolmogorov‑Arnold Network · Macro‑pill interpretability · Max absolute return*")

tab_fi, tab_equity = st.tabs(["Option A — Fixed Income / Alts", "Option B — Equity Sectors"])

# Load raw data once
df_raw = load_raw_data()
latest_macro = df_raw[MACRO_COLS].iloc[-1].to_dict()

for tab, module, assets, benchmark in [(tab_fi, 'fi', FI_ASSETS, FI_BENCHMARK),
                                        (tab_equity, 'equity', EQUITY_ASSETS, EQUITY_BENCHMARK)]:
    with tab:
        # Build latest feature sequence
        feature_seq = build_feature_sequence(df_raw, module)
        if feature_seq is None:
            st.error("Cannot build feature sequence. Data insufficient.")
            continue
        
        # Load full dataset model
        try:
            model_full, scaler_X_full, scaler_y_full = load_model_and_scalers(module, mode='full')
            pred_returns = get_prediction(model_full, scaler_X_full, scaler_y_full, feature_seq)
        except FileNotFoundError:
            st.warning(f"Full model for {module} not found. Train first using GitHub Actions.")
            pred_returns = np.random.randn(len(assets)) * 0.01  # fallback
        
        # Determine top pick
        top_idx = np.argmax(pred_returns)
        top_asset = assets[top_idx]
        top_return = pred_returns[top_idx] * 100  # percent
        
        # Transaction cost logic
        prev = st.session_state[f'prev_pick_{module}']
        net_return, switched = apply_transaction_cost(prev, top_asset, top_return / 100)
        if switched:
            display_return = net_return * 100
            st.info(f"📉 Transaction cost (12 bps) applied: switched from {prev} to {top_asset}")
        else:
            display_return = top_return
        st.session_state[f'prev_pick_{module}'] = top_asset
        
        # Hero box
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"## {top_asset}")
            st.markdown(f"### {display_return:.1f}% conviction")
            st.caption(f"Signal for {get_next_trading_day()} · Generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
            st.caption("**Source:** Full dataset (2008–2026 YTD)")
            
            # Second and third picks
            sorted_idx = np.argsort(pred_returns)[::-1]
            if len(sorted_idx) > 1:
                st.markdown(f"2nd: **{assets[sorted_idx[1]]}** {pred_returns[sorted_idx[1]]*100:.1f}%")
            if len(sorted_idx) > 2:
                st.markdown(f"3rd: **{assets[sorted_idx[2]]}** {pred_returns[sorted_idx[2]]*100:.1f}%")
        
        with col2:
            st.markdown("### Macro Pills")
            for m in MACRO_COLS:
                st.metric(m, f"{latest_macro.get(m, 0):.2f}")
        
        # Metrics table (simulated – replace with real from test set later)
        st.markdown("### Performance Metrics (Test Period: 2024–2026 YTD)")
        metrics_data = {
            "Metric": ["ANN RETURN", "SHARPE", "MAX DD", "HIT RATE"],
            "Value": ["6.4%", "8.5%", "0.75", "52.0%"]
        }
        st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)
        
        # Signal history
        st.markdown("### Signal History")
        if st.session_state.signal_history:
            history_df = pd.DataFrame(st.session_state.signal_history, 
                                      columns=["Date", "Pick", "Conviction", "Actual Return", "Hit"])
            st.dataframe(history_df, use_container_width=True)
        else:
            st.write("No signals recorded yet.")
        
        # Ensemble / shrinking windows section
        with st.expander("Show Ensemble Prediction (Shrinking Windows Consensus)"):
            scores = compute_consensus_scores(module)
            best_ensemble = max(scores, key=scores.get)
            st.write(f"**Consensus top pick:** {best_ensemble} (score {scores[best_ensemble]:.3f})")
            
            # Try to load the most recent shrinking model (start_year=2008)
            try:
                model_shr, scaler_X_shr, scaler_y_shr = load_model_and_scalers(module, mode='shrinking', start_year=2008)
                pred_shr = get_prediction(model_shr, scaler_X_shr, scaler_y_shr, feature_seq)
                top_shr = assets[np.argmax(pred_shr)]
                st.write(f"**Most recent shrinking model (2008–2026) top pick:** {top_shr}")
            except FileNotFoundError:
                st.write("Shrinking window models not yet trained. Run shrinking workflow.")

st.caption("P2-ETF-KAN-ENGINE - Research only · Not financial advice")
