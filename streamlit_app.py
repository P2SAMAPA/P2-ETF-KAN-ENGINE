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
from huggingface_hub import hf_hub_download, list_repo_files

# Constants
FI_ASSETS = ['GLD', 'TLT', 'VCIT', 'LQD', 'HYG', 'VNQ', 'SLV']
FI_BENCHMARK = 'AGG'
EQUITY_ASSETS = ['QQQ', 'XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU', 'XME', 'GDX', 'IWM']
EQUITY_BENCHMARK = 'SPY'
MACRO_COLS = ['VIX', 'DXY', 'T10Y2Y', 'TBILL_3M', 'IG_SPREAD', 'HY_SPREAD']
TRANSACTION_COST = 0.0012
SEQ_LEN = 20
HF_REPO = "P2SAMAPA/p2-etf-kan-engine-results"

# Get HF token from secrets or environment (optional for public datasets)
def get_hf_token():
    """Get HuggingFace token from Streamlit secrets or environment."""
    try:
        if hasattr(st, 'secrets') and "HF_TOKEN" in st.secrets:
            return st.secrets["HF_TOKEN"]
    except:
        pass
    return os.environ.get("HF_TOKEN", None)

HF_TOKEN = get_hf_token()

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

def download_file(filename, subfolder="", max_retries=3):
    """Download a file using huggingface_hub (token optional for public datasets)."""
    local_dir = "models"
    os.makedirs(local_dir, exist_ok=True)

    for attempt in range(max_retries):
        try:
            # Don't pass token for public datasets - let huggingface_hub handle it
            path = hf_hub_download(
                repo_id=HF_REPO,
                filename=filename,
                subfolder=subfolder if subfolder else None,
                repo_type="dataset",
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                token=HF_TOKEN  # Will be None if not set, which is fine for public repos
            )
            return path
        except Exception as e:
            error_msg = str(e)
            # Don't retry on auth/404 errors
            if "401" in error_msg or "403" in error_msg or "404" in error_msg or "not found" in error_msg.lower():
                st.warning(f"File not found: {filename} (may not be uploaded yet)")
                return None
            if attempt < max_retries - 1:
                if "connection" in error_msg.lower() or "timeout" in error_msg.lower() or "429" in error_msg:
                    continue
            st.warning(f"Download failed for {filename}: {error_msg}")
            return None

    return None

def load_model_and_scalers(module, mode='full', start_year=None):
    if mode == 'full':
        model_file = f"kan_{module}_full.pt"
        scaler_x_file = f"scaler_X_{module}_full.pkl"
        scaler_y_file = f"scaler_y_{module}_full.pkl"
        subfolder = ""
    else:
        model_file = f"kan_{module}_shrinking_start{start_year}.pt"
        scaler_x_file = f"scaler_X_{module}_shrinking_start{start_year}.pkl"
        scaler_y_file = f"scaler_y_{module}_shrinking_start{start_year}.pkl"
        subfolder = "shrinking_models"

    model_path = download_file(model_file, subfolder)
    scaler_x_path = download_file(scaler_x_file, subfolder)
    scaler_y_path = download_file(scaler_y_file, subfolder)

    if not (model_path and scaler_x_path and scaler_y_path):
        return None, None, None

    scaler_X = joblib.load(scaler_x_path)
    scaler_y = joblib.load(scaler_y_path)
    n_features = scaler_X.mean_.shape[0]
    input_dim = SEQ_LEN * n_features
    output_dim = len(FI_ASSETS) if module == 'fi' else len(EQUITY_ASSETS)
    model = TemporalKANForecaster(input_dim, hidden_dims=[256,128], output_dim=output_dim, grid_size=20)
    try:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    except Exception as e:
        st.error(f"Model mismatch: {e}")
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
    fname = f"metrics_{module}_full.pkl"
    path = download_file(fname, "")
    if path:
        return joblib.load(path)
    return None

def compute_metrics(test_pred, test_true):
    pred_avg = test_pred.mean(axis=1)
    true_avg = test_true.mean(axis=1)
    ann_factor = np.sqrt(252)
    mean_pred = np.mean(pred_avg)
    std_pred = np.std(pred_avg)
    sharpe = (mean_pred / std_pred) * ann_factor if std_pred > 0 else 0.0
    ann_return = mean_pred * 252 * 100
    cum_ret = np.cumprod(1 + pred_avg)
    peak = np.maximum.accumulate(cum_ret)
    drawdown = (cum_ret - peak) / peak
    max_dd = np.min(drawdown) * 100
    hit = np.mean(np.sign(pred_avg) == np.sign(true_avg)) * 100
    return ann_return, sharpe, max_dd, hit

def get_shrinking_consensus(module, feature_seq):
    """Load all shrinking models and average predictions."""
    try:
        files = list_repo_files(HF_REPO, repo_type="dataset", token=HF_TOKEN)
        pattern = f"shrinking_models/kan_{module}_shrinking_start"
        model_files = [f for f in files if f.startswith(pattern) and f.endswith(".pt")]
        if not model_files:
            st.info(f"No shrinking models found for {module} module in the repository.")
            return None
    except Exception as e:
        st.warning(f"Could not list shrinking models: {e}")
        return None

    preds = []
    for mf in model_files:
        start_year = int(mf.split("_start")[-1].split(".pt")[0])
        model, scaler_X, scaler_y = load_model_and_scalers(module, mode='shrinking', start_year=start_year)
        if model is None:
            continue
        pred = get_prediction(model, scaler_X, scaler_y, feature_seq)
        preds.append(pred)

    if not preds:
        st.info(f"Successfully found {len(model_files)} shrinking models but failed to load any of them.")
        return None

    return np.mean(preds, axis=0)

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
        feat_seq = build_feature_sequence(df_raw, module)
        if feat_seq is None:
            st.warning(f"Insufficient data for {module}.")
            continue

        # --- Full dataset model ---
        model_full, scaler_X_full, scaler_y_full = load_model_and_scalers(module, mode='full')
        if model_full is None:
            st.warning(f"Full model for {module} not available. Train via GitHub Actions.")
        else:
            pred_returns = get_prediction(model_full, scaler_X_full, scaler_y_full, feat_seq)
            top_idx = np.argmax(pred_returns)
            top_asset = assets[top_idx]
            top_return = pred_returns[top_idx] * 100

            prev = st.session_state[f'prev_pick_{module}']
            net_return, switched = apply_transaction_cost(prev, top_asset, top_return/100)
            display_return = net_return * 100 if switched else top_return
            if switched:
                st.info(f"📉 Transaction cost (12bps): switched from {prev} to {top_asset}")
            st.session_state[f'prev_pick_{module}'] = top_asset

            col1, col2 = st.columns([2,1])
            with col1:
                st.markdown(f"## {top_asset}")
                st.markdown(f"### {display_return:.1f}% conviction")
                st.caption(f"Signal for {get_next_trading_day()} · Generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
                st.caption("Source: Full dataset (2008–2026 YTD)")
                sorted_idx = np.argsort(pred_returns)[::-1]
                if len(sorted_idx) > 1:
                    st.markdown(f"2nd: **{assets[sorted_idx[1]]}** {pred_returns[sorted_idx[1]]*100:.1f}%")
                if len(sorted_idx) > 2:
                    st.markdown(f"3rd: **{assets[sorted_idx[2]]}** {pred_returns[sorted_idx[2]]*100:.1f}%")
            with col2:
                st.markdown("### Macro Pills")
                for m in MACRO_COLS:
                    st.metric(m, f"{latest_macro.get(m, 0):.2f}")

        # --- Shrinking consensus ---
        with st.expander("Show Ensemble Prediction (Shrinking Windows Consensus)"):
            consensus_pred = get_shrinking_consensus(module, feat_seq)
            if consensus_pred is None:
                st.write("Shrinking window models not yet available. Train via `train_shrinking.yml`.")
            else:
                top_idx_cons = np.argmax(consensus_pred)
                top_asset_cons = assets[top_idx_cons]
                top_return_cons = consensus_pred[top_idx_cons] * 100
                st.markdown(f"**Consensus top pick (across all windows):** {top_asset_cons} ({top_return_cons:.1f}% predicted return)")
                sorted_idx_cons = np.argsort(consensus_pred)[::-1]
                if len(sorted_idx_cons) > 1:
                    st.write(f"2nd: {assets[sorted_idx_cons[1]]} ({consensus_pred[sorted_idx_cons[1]]*100:.1f}%)")
                if len(sorted_idx_cons) > 2:
                    st.write(f"3rd: {assets[sorted_idx_cons[2]]} ({consensus_pred[sorted_idx_cons[2]]*100:.1f}%)")

        # --- Metrics ---
        metrics_data = load_metrics(module)
        if metrics_data is not None:
            test_pred = np.array(metrics_data['test_predictions'])
            test_true = np.array(metrics_data['test_true'])
            ann_ret, sharpe, max_dd, hit = compute_metrics(test_pred, test_true)
            st.markdown("### Performance Metrics (Test Period)")
            df_metrics = pd.DataFrame({
                "Metric": ["ANN RETURN", "SHARPE", "MAX DD", "HIT RATE"],
                "Value": [f"{ann_ret:.1f}%", f"{sharpe:.2f}", f"{max_dd:.1f}%", f"{hit:.1f}%"]
            })
            st.dataframe(df_metrics, use_container_width=True)
        else:
            st.warning("Metrics not available yet.")

        # --- Signal History ---
        st.markdown("### Signal History")
        if st.session_state.signal_history:
            hist_df = pd.DataFrame(st.session_state.signal_history, columns=["Date","Pick","Conviction","Actual Return","Hit"])
            st.dataframe(hist_df, use_container_width=True)
        else:
            st.write("No signals recorded yet.")

st.caption("P2-ETF-KAN-ENGINE - Research only · Not financial advice")
