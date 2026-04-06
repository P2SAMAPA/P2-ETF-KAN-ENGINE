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
import warnings

# ── Constants ────────────────────────────────────────────────────────────────
FI_ASSETS = ['GLD', 'TLT', 'VCIT', 'LQD', 'HYG', 'VNQ', 'SLV']
FI_BENCHMARK = 'AGG'
EQUITY_ASSETS = ['QQQ', 'XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU', 'XME', 'GDX', 'IWM']
EQUITY_BENCHMARK = 'SPY'
MACRO_COLS = ['VIX', 'DXY', 'T10Y2Y', 'TBILL_3M', 'IG_SPREAD', 'HY_SPREAD']
TRANSACTION_COST = 0.0012
SEQ_LEN = 20
HF_REPO = "P2SAMAPA/p2-etf-kan-engine-results"

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="P2‑ETF‑KAN‑ENGINE",
    page_icon="⚡",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.hero-box{border:1px solid #e0e0e0;border-radius:8px;padding:1.5rem;margin-bottom:1rem;background:#fafafa}
.hero-ticker{font-size:2.5rem;font-weight:700;color:#1a1a1a;margin-bottom:0.2rem}
.hero-return{font-size:1.8rem;font-weight:600;color:#2e7d32;margin-bottom:0.5rem}
.hero-meta{font-size:0.85rem;color:#666;margin-top:0.5rem}
.runner-box{display:inline-block;width:48%;padding:0.8rem;background:#fff;border:1px solid #e0e0e0;border-radius:6px;margin-right:2%;vertical-align:top}
.runner-label{font-size:0.75rem;color:#888;text-transform:uppercase;letter-spacing:0.5px}
.runner-ticker{font-size:1.2rem;font-weight:600;color:#333}
.runner-return{font-size:1rem;color:#555}
.switch-warning{color:#d32f2f;font-size:0.9rem;margin-top:0.5rem}
.metrics-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin-top:1rem}
.metric-card{border:1px solid #e0e0e0;border-radius:6px;padding:1rem;text-align:center;background:#fff}
.metric-label{font-size:0.75rem;color:#888;text-transform:uppercase;margin-bottom:0.3rem}
.metric-value{font-size:1.3rem;font-weight:700;color:#1a1a1a}
.not-available{color:#999;font-style:italic;text-align:center;padding:2rem}
</style>
""", unsafe_allow_html=True)

# ── HF Token ──────────────────────────────────────────────────────────────────
def get_hf_token():
    try:
        if hasattr(st, 'secrets') and "HF_TOKEN" in st.secrets:
            return st.secrets["HF_TOKEN"]
    except Exception:
        pass
    return os.environ.get("HF_TOKEN", None)

HF_TOKEN = get_hf_token()

# ── Session state ─────────────────────────────────────────────────────────────
for _k in ('prev_pick_fi', 'prev_pick_equity'):
    if _k not in st.session_state:
        st.session_state[_k] = None
if 'signal_history' not in st.session_state:
    st.session_state.signal_history = []

# ── Utilities ─────────────────────────────────────────────────────────────────
def get_next_trading_day():
    nyse = mcal.get_calendar('NYSE')
    today = datetime.now().date()
    schedule = nyse.schedule(start_date=today, end_date=today + timedelta(days=10))
    return schedule.index[0].date() if len(schedule) > 0 else today + timedelta(days=1)

@st.cache_data(show_spinner=False)
def load_raw_data():
    ds = load_dataset("P2SAMAPA/fi-etf-macro-signal-master-data", split="train")
    df = ds.to_pandas()
    df['date'] = pd.to_datetime(df['__index_level_0__'], unit='s')
    df.set_index('date', inplace=True)
    df.drop('__index_level_0__', axis=1, inplace=True)
    df.sort_index(inplace=True)
    for col in FI_ASSETS + EQUITY_ASSETS + MACRO_COLS:
        if col not in df.columns:
            df[col] = np.nan
    df.ffill(inplace=True)
    df.dropna(inplace=True)
    return df

def build_feature_sequence(df, module):
    assets = FI_ASSETS if module == 'fi' else EQUITY_ASSETS
    features = df[MACRO_COLS].copy()
    for lag in range(1, 6):
        for a in assets:
            features[f'{a}_lag{lag}'] = df[a].pct_change().shift(lag)
    features = features.dropna()
    if len(features) < SEQ_LEN:
        return None
    return features.iloc[-SEQ_LEN:].values

def download_file(filename, subfolder="", max_retries=3):
    os.makedirs("models", exist_ok=True)
    for attempt in range(max_retries):
        try:
            # Try with subfolder first
            if subfolder:
                try:
                    return hf_hub_download(
                        repo_id=HF_REPO,
                        filename=filename,
                        subfolder=subfolder,
                        repo_type="model",
                        local_dir="models",
                        local_dir_use_symlinks=False,
                        token=HF_TOKEN,
                    )
                except:
                    pass
            
            # Try without subfolder (root of repo)
            return hf_hub_download(
                repo_id=HF_REPO,
                filename=filename,
                repo_type="model",
                local_dir="models",
                local_dir_use_symlinks=False,
                token=HF_TOKEN,
            )
        except Exception as e:
            msg = str(e)
            if any(c in msg for c in ("401", "403")):
                st.error(f"HF Authentication error: {msg}")
                return None
            if "404" in msg or "not found" in msg.lower():
                return None
            if attempt == max_retries - 1:
                st.warning(f"Failed to download {filename}: {msg}")
                return None
    return None

def load_model_and_scalers(module, mode='full', start_year=None):
    if mode == 'full':
        mf, sxf, syf = (
            f"kan_{module}_full.pt",
            f"scaler_X_{module}_full.pkl",
            f"scaler_y_{module}_full.pkl",
        )
        # Try both root and shrinking_models subfolder for full models
        paths = [download_file(f, "") for f in (mf, sxf, syf)]
        if not all(paths):
            paths = [download_file(f, "shrinking_models") for f in (mf, sxf, syf)]
    else:
        mf, sxf, syf = (
            f"kan_{module}_shrinking_start{start_year}.pt",
            f"scaler_X_{module}_shrinking_start{start_year}.pkl",
            f"scaler_y_{module}_shrinking_start{start_year}.pkl",
        )
        # Try both root and shrinking_models subfolder
        paths = [download_file(f, "shrinking_models") for f in (mf, sxf, syf)]
        if not all(paths):
            paths = [download_file(f, "") for f in (mf, sxf, syf)]
    
    if not all(paths):
        st.warning(f"Missing files for {module} {mode}: {[mf, sxf, syf]}")
        return None, None, None
    
    try:
        scaler_X = joblib.load(paths[1])
        scaler_y = joblib.load(paths[2])
    except Exception as e:
        st.error(f"Failed to load scalers: {e}")
        return None, None, None
    
    input_dim = SEQ_LEN * scaler_X.mean_.shape[0]
    output_dim = len(FI_ASSETS) if module == 'fi' else len(EQUITY_ASSETS)
    
    # CRITICAL FIX: Added seq_len=SEQ_LEN parameter
    model = TemporalKANForecaster(
        input_dim, 
        hidden_dims=[256, 128], 
        output_dim=output_dim, 
        grid_size=20,
        seq_len=SEQ_LEN  # THIS WAS MISSING!
    )
    
    try:
        state_dict = torch.load(paths[0], map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()
        return model, scaler_X, scaler_y
    except RuntimeError as e:
        st.error(f"Model architecture mismatch for {mf}: {e}. Try retraining with updated train.py.")
        return None, None, None
    except Exception as e:
        st.error(f"Failed to load model {mf}: {e}")
        return None, None, None

def get_prediction(model, scaler_X, scaler_y, feature_seq):
    seq_scaled = scaler_X.transform(feature_seq)
    X_tensor = torch.FloatTensor(seq_scaled.flatten()).unsqueeze(0)
    with torch.no_grad():
        pred_scaled = model(X_tensor).numpy()[0]
    return scaler_y.inverse_transform(pred_scaled.reshape(1, -1))[0]

def apply_transaction_cost(prev_pick, new_pick, gross_return):
    if prev_pick is None or new_pick == prev_pick:
        return gross_return, False
    return gross_return - TRANSACTION_COST, True

def load_metrics(module):
    # Try root first, then shrinking_models
    path = download_file(f"metrics_{module}_full.pkl", "")
    if not path:
        path = download_file(f"metrics_{module}_full.pkl", "shrinking_models")
    if path:
        try:
            return joblib.load(path)
        except Exception as e:
            st.warning(f"Failed to load metrics: {e}")
    return None

def compute_metrics(test_pred, test_true):
    pred_avg = np.array(test_pred).mean(axis=1)
    true_avg = np.array(test_true).mean(axis=1)
    mean_p = np.mean(pred_avg)
    std_p = np.std(pred_avg)
    sharpe = (mean_p / std_p) * np.sqrt(252) if std_p > 0 else 0.0
    ann_ret = mean_p * 252 * 100
    cum = np.cumprod(1 + pred_avg)
    peak = np.maximum.accumulate(cum)
    max_dd = np.min((cum - peak) / peak) * 100
    hit = np.mean(np.sign(pred_avg) == np.sign(true_avg)) * 100
    return ann_ret, sharpe, max_dd, hit

def get_shrinking_consensus(module, feature_seq):
    try:
        files = list_repo_files(HF_REPO, repo_type="model", token=HF_TOKEN)
    except Exception as e:
        st.warning(f"Could not list repo files: {e}")
        return None
    
    # Look for shrinking model files in both root and subfolder
    patterns = [
        f"shrinking_models/kan_{module}_shrinking_start",
        f"kan_{module}_shrinking_start"
    ]
    model_files = []
    for pattern in patterns:
        model_files.extend([f for f in files if pattern in f and f.endswith(".pt")])
    
    if not model_files:
        st.info(f"No shrinking models found for {module}")
        return None
    
    preds = []
    for mf in model_files:
        # Extract year from filename
        try:
            if "_start" in mf:
                yr = int(mf.split("_start")[-1].split(".pt")[0])
            else:
                continue
        except:
            continue
            
        model, sx, sy = load_model_and_scalers(module, mode='shrinking', start_year=yr)
        if model is None:
            continue
        try:
            pred = get_prediction(model, sx, sy, feature_seq)
            preds.append(pred)
        except Exception as e:
            st.warning(f"Prediction failed for {mf}: {e}")
            continue
    
    return np.mean(preds, axis=0) if preds else None

def runners_html(assets, sorted_idx, pred):
    """Build compact single-line HTML for 2nd/3rd runners-up."""
    items = ""
    for rank, label in enumerate(["2nd", "3rd"], start=1):
        if len(sorted_idx) > rank:
            i = sorted_idx[rank]
            items += (
                f'<div class="runner-box">'
                f'<div class="runner-label">{label}</div>'
                f'<div class="runner-ticker">{assets[i]}</div>'
                f'<div class="runner-return">{pred[i]*100:+.2f}%</div>'
                f'</div>'
            )
    return f'<div style="margin-top:1rem">{items}</div>' if items else ""

def render_html(html: str):
    """Wrap HTML in a no-margin div to guarantee Streamlit renders it as HTML, not markdown."""
    st.markdown(f'<div style="margin:0">{html}</div>', unsafe_allow_html=True)

# ── App header ────────────────────────────────────────────────────────────────
now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
next_day = get_next_trading_day()

st.markdown(f"""
<div style="text-align:center;margin-bottom:2rem">
    <h1 style="margin-bottom:0.2rem">⚡ P2‑ETF‑KAN‑ENGINE</h1>
    <p style="color:#666;font-size:0.9rem">
        Kolmogorov‑Arnold Network for ETF Return Forecasting · {now_str}
    </p>
</div>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("Loading market data…"):
    try:
        df_raw = load_raw_data()
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        st.stop()

os.makedirs("models", exist_ok=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_fi, tab_eq = st.tabs([
    "Option A — Fixed Income / Alts",
    "Option B — Equity Sectors",
])

for tab, module, assets, benchmark in [
    (tab_fi, 'fi', FI_ASSETS, FI_BENCHMARK),
    (tab_eq, 'equity', EQUITY_ASSETS, EQUITY_BENCHMARK),
]:
    with tab:
        feat_seq = build_feature_sequence(df_raw, module)
        if feat_seq is None:
            st.warning("Insufficient data to build feature sequence.")
            continue

        col_full, col_cons = st.columns(2, gap="large")

        # ── LEFT card: Full dataset ───────────────────────────────────────
        with col_full:
            with st.spinner("Loading full model…"):
                model_full, sx_full, sy_full = load_model_and_scalers(module, mode='full')

            if model_full is None:
                render_html(
                    '<div class="hero-box not-available">'
                    '<div style="font-size:3rem;color:#ddd">◌</div>'
                    '<h3>Model not available</h3>'
                    '<p>Train via GitHub Actions (train.yml)<br>Check logs for errors</p>'
                    '</div>'
                )
            else:
                pred = get_prediction(model_full, sx_full, sy_full, feat_seq)
                top_idx = int(np.argmax(pred))
                top_asset = assets[top_idx]
                top_ret = pred[top_idx]

                prev = st.session_state[f'prev_pick_{module}']
                net_ret, switched = apply_transaction_cost(prev, top_asset, top_ret)
                display_ret = net_ret if switched else top_ret
                st.session_state[f'prev_pick_{module}'] = top_asset

                sorted_idx = np.argsort(pred)[::-1]
                switch_html = ""
                if switched and prev:
                    switch_html = f'<div class="switch-warning">⚠ Switched from {prev} · −12 bps applied</div>'

                meta = (
                    f'<div class="hero-meta">'
                    f'Signal for &nbsp;&nbsp;<strong>{next_day}</strong><br>'
                    f'Generated &nbsp;&nbsp;<strong>{now_str}</strong><br>'
                    f'Benchmark &nbsp;&nbsp;<strong>{benchmark}</strong>'
                    f'</div>'
                )
                render_html(
                    f'<div class="hero-box">'
                    f'<div style="font-size:0.85rem;color:#888;margin-bottom:0.3rem">Full Dataset · 2008–2026 YTD</div>'
                    f'<div class="hero-ticker">{top_asset}</div>'
                    f'<div class="hero-return">{display_ret*100:+.2f}% predicted return</div>'
                    f'{switch_html}'
                    f'{meta}'
                    f'{runners_html(assets, sorted_idx, pred)}'
                    f'</div>'
                )

        # ── RIGHT card: Shrinking consensus ───────────────────────────────
        with col_cons:
            with st.spinner("Loading shrinking ensemble…"):
                cons_pred = get_shrinking_consensus(module, feat_seq)

            if cons_pred is None:
                render_html(
                    '<div class="hero-box not-available">'
                    '<div style="font-size:3rem;color:#ddd">◌</div>'
                    '<h3>Ensemble not available</h3>'
                    '<p>Train via GitHub Actions (train_shrinking.yml)<br>Check logs for errors</p>'
                    '</div>'
                )
            else:
                top_idx_c = int(np.argmax(cons_pred))
                top_asset_c = assets[top_idx_c]
                sorted_c = np.argsort(cons_pred)[::-1]

                meta_c = (
                    f'<div class="hero-meta">'
                    f'Signal for &nbsp;&nbsp;<strong>{next_day}</strong><br>'
                    f'Generated &nbsp;&nbsp;<strong>{now_str}</strong><br>'
                    f'Benchmark &nbsp;&nbsp;<strong>{benchmark}</strong>'
                    f'</div>'
                )
                render_html(
                    f'<div class="hero-box">'
                    f'<div style="font-size:0.85rem;color:#888;margin-bottom:0.3rem">Shrinking Windows Ensemble</div>'
                    f'<div class="hero-ticker">{top_asset_c}</div>'
                    f'<div class="hero-return">{cons_pred[top_idx_c]*100:+.2f}% consensus return</div>'
                    f'{meta_c}'
                    f'{runners_html(assets, sorted_c, cons_pred)}'
                    f'</div>'
                )

        # ── Performance metrics ───────────────────────────────────────────
        st.markdown('<h4 style="margin-top:1.5rem">Performance Metrics · Test Period</h4>', unsafe_allow_html=True)

        metrics_data = load_metrics(module)
        if metrics_data:
            ann_ret, sharpe, max_dd, hit = compute_metrics(
                metrics_data['test_predictions'], metrics_data['test_true']
            )
            render_html(
                f'<div class="metrics-grid">'
                f'<div class="metric-card">'
                f'<div class="metric-label">Ann. Return</div>'
                f'<div class="metric-value" style="color:{"#2e7d32" if ann_ret>=0 else "#d32f2f"}">{ann_ret:+.1f}%</div>'
                f'</div>'
                f'<div class="metric-card">'
                f'<div class="metric-label">Sharpe Ratio</div>'
                f'<div class="metric-value" style="color:{"#2e7d32" if sharpe>=0 else "#d32f2f"}">{sharpe:.2f}</div>'
                f'</div>'
                f'<div class="metric-card">'
                f'<div class="metric-label">Max Drawdown</div>'
                f'<div class="metric-value" style="color:#d32f2f">{max_dd:.1f}%</div>'
                f'</div>'
                f'<div class="metric-card">'
                f'<div class="metric-label">Hit Rate</div>'
                f'<div class="metric-value" style="color:{"#2e7d32" if hit>=50 else "#d32f2f"}">{hit:.1f}%</div>'
                f'</div>'
                f'</div>'
            )
        else:
            st.markdown(
                '<div style="color:#999;font-style:italic">Metrics not available — run training to generate.</div>',
                unsafe_allow_html=True,
            )

        # ── Signal history ────────────────────────────────────────────────
        st.markdown('<h4 style="margin-top:1.5rem">Signal History</h4>', unsafe_allow_html=True)

        if st.session_state.signal_history:
            hist_df = pd.DataFrame(
                st.session_state.signal_history,
                columns=["Date", "Pick", "Conviction", "Actual Return", "Hit"],
            )
            st.dataframe(hist_df, use_container_width=True, hide_index=True)
        else:
            st.markdown(
                '<div style="color:#999;font-style:italic">No signals recorded yet.</div>',
                unsafe_allow_html=True,
            )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    '<div style="text-align:center;color:#999;font-size:0.8rem;margin-top:2rem;padding-top:1rem;border-top:1px solid #eee">'
    'Research purposes only · Not financial advice'
    '</div>',
    unsafe_allow_html=True,
)
