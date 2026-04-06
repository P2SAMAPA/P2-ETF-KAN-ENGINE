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

# ── Constants ────────────────────────────────────────────────────────────────
FI_ASSETS        = ['GLD', 'TLT', 'VCIT', 'LQD', 'HYG', 'VNQ', 'SLV']
FI_BENCHMARK     = 'AGG'
EQUITY_ASSETS    = ['QQQ', 'XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU', 'XME', 'GDX', 'IWM']
EQUITY_BENCHMARK = 'SPY'
MACRO_COLS       = ['VIX', 'DXY', 'T10Y2Y', 'TBILL_3M', 'IG_SPREAD', 'HY_SPREAD']
TRANSACTION_COST = 0.0012
SEQ_LEN          = 20
HF_REPO          = "P2SAMAPA/p2-etf-kan-engine-results"

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
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
    background: #f4f5fa;
    color: #1a1d2e;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem; max-width: 1400px; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    gap: 4px;
    border-bottom: 2px solid #e2e4f0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #9099b8;
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    border-radius: 0;
    padding: 0.65rem 1.4rem;
    margin-bottom: -2px;
}
.stTabs [aria-selected="true"] {
    color: #1a1d2e !important;
    background: transparent !important;
    border-bottom: 2px solid #2952cc !important;
    font-weight: 600 !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 2rem; }

/* Header */
.app-header {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    padding-bottom: 1.4rem;
    margin-bottom: 0.5rem;
    border-bottom: 1px solid #e2e4f0;
}
.app-title {
    font-family: 'Sora', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #1a1d2e;
}
.app-title span { color: #2952cc; }
.app-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: #9099b8;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 0.3rem;
}
.app-ts {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.66rem;
    color: #b0b8d0;
    text-align: right;
    line-height: 1.9;
}

/* Hero cards */
.hero-card {
    background: #ffffff;
    border: 1px solid #e2e4f0;
    border-radius: 16px;
    padding: 2rem 2.2rem 1.8rem;
    position: relative;
    overflow: hidden;
    min-height: 340px;
    box-shadow: 0 2px 8px rgba(30,40,100,0.06), 0 0 0 0 transparent;
}
.hero-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 4px;
    background: linear-gradient(90deg, #2952cc, #6690ff);
    border-radius: 16px 16px 0 0;
}
.hero-card.consensus::after {
    background: linear-gradient(90deg, #0ba360, #3dd68c);
}
.hero-card-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #2952cc;
    margin-bottom: 1.4rem;
    display: flex;
    align-items: center;
    gap: 7px;
}
.hero-card.consensus .hero-card-label { color: #0ba360; }
.hero-card-label::before {
    content: '';
    display: inline-block;
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #2952cc;
    flex-shrink: 0;
}
.hero-card.consensus .hero-card-label::before { background: #0ba360; }

.hero-ticker {
    font-family: 'Sora', sans-serif;
    font-size: 3.6rem;
    font-weight: 700;
    color: #1a1d2e;
    line-height: 1;
    letter-spacing: -0.03em;
    margin-bottom: 0.5rem;
}
.hero-return {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.3rem;
    font-weight: 600;
    color: #2952cc;
    margin-bottom: 1.4rem;
}
.hero-card.consensus .hero-return { color: #0ba360; }
.hero-meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: #9099b8;
    line-height: 2;
}
.hero-divider {
    border: none;
    border-top: 1px solid #f0f1f8;
    margin: 1.2rem 0 1rem;
}
.hero-runners { display: flex; gap: 1.5rem; }
.runner-item { flex: 1; }
.runner-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #c8ccde;
    margin-bottom: 3px;
}
.runner-ticker {
    font-family: 'Sora', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    color: #4a5280;
}
.runner-pct {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.74rem;
    color: #9099b8;
}

/* Switch badge */
.switch-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: #fff8ed;
    border: 1px solid #f5d9a0;
    color: #a06010;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.04em;
    padding: 3px 9px;
    border-radius: 4px;
    margin-bottom: 0.8rem;
    display: block;
}

/* Unavailable card */
.unavailable-card {
    background: #fafbfd;
    border: 1.5px dashed #dde0f0;
    border-radius: 16px;
    padding: 2rem;
    min-height: 340px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
}
.unavailable-icon { font-size: 2rem; margin-bottom: 0.8rem; opacity: 0.3; }
.unavailable-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #c0c8e0;
    margin-bottom: 0.4rem;
}
.unavailable-sub { font-size: 0.72rem; color: #d0d5e8; }

/* Section heading */
.section-heading {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #9099b8;
    margin: 2.5rem 0 1rem;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid #eceef8;
}

/* Metrics grid */
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1px;
    background: #e8eaf5;
    border: 1px solid #e8eaf5;
    border-radius: 12px;
    overflow: hidden;
}
.metric-cell {
    background: #ffffff;
    padding: 1.1rem 1.5rem;
}
.metric-name {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #b0b8d0;
    margin-bottom: 0.4rem;
}
.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.3rem;
    font-weight: 600;
    color: #1a1d2e;
}
.metric-value.positive { color: #0ba360; }
.metric-value.negative { color: #e0404a; }

/* Dataframe */
[data-testid="stDataFrame"] {
    border-radius: 10px !important;
    border: 1px solid #e8eaf5 !important;
    overflow: hidden;
}

/* Footer */
.app-footer {
    margin-top: 3.5rem;
    border-top: 1px solid #eceef8;
    padding-top: 1rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    color: #c0c8e0;
    letter-spacing: 0.06em;
    text-align: center;
}
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
    nyse     = mcal.get_calendar('NYSE')
    today    = datetime.now().date()
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
    assets   = FI_ASSETS if module == 'fi' else EQUITY_ASSETS
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
            return hf_hub_download(
                repo_id=HF_REPO,
                filename=filename,
                subfolder=subfolder if subfolder else None,
                repo_type="model",
                local_dir="models",
                local_dir_use_symlinks=False,
                token=HF_TOKEN,
            )
        except Exception as e:
            msg = str(e)
            if any(c in msg for c in ("401", "403", "404")) or "not found" in msg.lower():
                return None
            if attempt == max_retries - 1:
                return None
    return None


def load_model_and_scalers(module, mode='full', start_year=None):
    if mode == 'full':
        mf, sxf, syf, sub = (
            f"kan_{module}_full.pt",
            f"scaler_X_{module}_full.pkl",
            f"scaler_y_{module}_full.pkl",
            ""
        )
    else:
        mf, sxf, syf, sub = (
            f"kan_{module}_shrinking_start{start_year}.pt",
            f"scaler_X_{module}_shrinking_start{start_year}.pkl",
            f"scaler_y_{module}_shrinking_start{start_year}.pkl",
            "shrinking_models"
        )
    paths = [download_file(f, sub) for f in (mf, sxf, syf)]
    if not all(paths):
        return None, None, None
    scaler_X   = joblib.load(paths[1])
    scaler_y   = joblib.load(paths[2])
    input_dim  = SEQ_LEN * scaler_X.mean_.shape[0]
    output_dim = len(FI_ASSETS) if module == 'fi' else len(EQUITY_ASSETS)
    model = TemporalKANForecaster(input_dim, hidden_dims=[256, 128], output_dim=output_dim, grid_size=20)
    try:
        model.load_state_dict(torch.load(paths[0], map_location='cpu'))
    except Exception:
        return None, None, None
    model.eval()
    return model, scaler_X, scaler_y


def get_prediction(model, scaler_X, scaler_y, feature_seq):
    seq_scaled  = scaler_X.transform(feature_seq)
    X_tensor    = torch.FloatTensor(seq_scaled.flatten()).unsqueeze(0)
    with torch.no_grad():
        pred_scaled = model(X_tensor).numpy()[0]
    return scaler_y.inverse_transform(pred_scaled.reshape(1, -1))[0]


def apply_transaction_cost(prev_pick, new_pick, gross_return):
    if prev_pick is None or new_pick == prev_pick:
        return gross_return, False
    return gross_return - TRANSACTION_COST, True


def load_metrics(module):
    path = download_file(f"metrics_{module}_full.pkl", "")
    return joblib.load(path) if path else None


def compute_metrics(test_pred, test_true):
    pred_avg = np.array(test_pred).mean(axis=1)
    true_avg = np.array(test_true).mean(axis=1)
    mean_p   = np.mean(pred_avg)
    std_p    = np.std(pred_avg)
    sharpe   = (mean_p / std_p) * np.sqrt(252) if std_p > 0 else 0.0
    ann_ret  = mean_p * 252 * 100
    cum      = np.cumprod(1 + pred_avg)
    peak     = np.maximum.accumulate(cum)
    max_dd   = np.min((cum - peak) / peak) * 100
    hit      = np.mean(np.sign(pred_avg) == np.sign(true_avg)) * 100
    return ann_ret, sharpe, max_dd, hit


def get_shrinking_consensus(module, feature_seq):
    try:
        files = list_repo_files(HF_REPO, repo_type="model", token=HF_TOKEN)
    except Exception:
        return None
    pattern     = f"shrinking_models/kan_{module}_shrinking_start"
    model_files = [f for f in files if f.startswith(pattern) and f.endswith(".pt")]
    if not model_files:
        return None
    preds = []
    for mf in model_files:
        yr = int(mf.split("_start")[-1].split(".pt")[0])
        model, sx, sy = load_model_and_scalers(module, mode='shrinking', start_year=yr)
        if model is None:
            continue
        preds.append(get_prediction(model, sx, sy, feature_seq))
    return np.mean(preds, axis=0) if preds else None


def runners_html(assets, sorted_idx, pred):
    html = '<hr class="hero-divider"><div class="hero-runners">'
    for rank, label in enumerate(["2nd", "3rd"], start=1):
        if len(sorted_idx) > rank:
            i = sorted_idx[rank]
            html += f"""
            <div class="runner-item">
                <div class="runner-label">{label}</div>
                <div class="runner-ticker">{assets[i]}</div>
                <div class="runner-pct">{pred[i]*100:+.2f}%</div>
            </div>"""
    html += "</div>"
    return html


# ── App header ────────────────────────────────────────────────────────────────
now_str  = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
next_day = get_next_trading_day()

st.markdown(f"""
<div class="app-header">
    <div>
        <div class="app-title">P2‑ETF‑<span>KAN</span>‑ENGINE</div>
        <div class="app-sub">Kolmogorov‑Arnold Network · Max Absolute Return · 12 bps Transaction Cost</div>
    </div>
    <div class="app-ts">
        Signal date &nbsp;&nbsp;{next_day}<br>
        Generated &nbsp;&nbsp;&nbsp;{now_str}
    </div>
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
    (tab_fi, 'fi',     FI_ASSETS,     FI_BENCHMARK),
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
                st.markdown("""
                <div class="unavailable-card">
                    <div class="unavailable-icon">◌</div>
                    <div class="unavailable-title">Model not available</div>
                    <div class="unavailable-sub">Train via GitHub Actions (train.yml)</div>
                </div>""", unsafe_allow_html=True)
            else:
                pred      = get_prediction(model_full, sx_full, sy_full, feat_seq)
                top_idx   = int(np.argmax(pred))
                top_asset = assets[top_idx]
                top_ret   = pred[top_idx]

                prev = st.session_state[f'prev_pick_{module}']
                net_ret, switched = apply_transaction_cost(prev, top_asset, top_ret)
                display_ret = net_ret if switched else top_ret
                st.session_state[f'prev_pick_{module}'] = top_asset

                sorted_idx  = np.argsort(pred)[::-1]
                switch_html = ""
                if switched and prev:
                    switch_html = f'<div class="switch-badge">⚠ Switched from {prev} · −12 bps applied</div>'

                st.markdown(f"""
                <div class="hero-card">
                    <div class="hero-card-label">Full Dataset · 2008–2026 YTD</div>
                    <div class="hero-ticker">{top_asset}</div>
                    <div class="hero-return">{display_ret*100:+.2f}% predicted return</div>
                    {switch_html}
                    <div class="hero-meta">
                        Signal for &nbsp;&nbsp;<strong>{next_day}</strong><br>
                        Generated &nbsp;&nbsp;<strong>{now_str}</strong><br>
                        Benchmark &nbsp;&nbsp;<strong>{benchmark}</strong>
                    </div>
                    {runners_html(assets, sorted_idx, pred)}
                </div>
                """, unsafe_allow_html=True)

        # ── RIGHT card: Shrinking consensus ───────────────────────────────
        with col_cons:
            with st.spinner("Loading shrinking ensemble…"):
                cons_pred = get_shrinking_consensus(module, feat_seq)

            if cons_pred is None:
                st.markdown("""
                <div class="unavailable-card">
                    <div class="unavailable-icon">◌</div>
                    <div class="unavailable-title">Ensemble not available</div>
                    <div class="unavailable-sub">Train via GitHub Actions (train_shrinking.yml)</div>
                </div>""", unsafe_allow_html=True)
            else:
                top_idx_c   = int(np.argmax(cons_pred))
                top_asset_c = assets[top_idx_c]
                sorted_c    = np.argsort(cons_pred)[::-1]

                st.markdown(f"""
                <div class="hero-card consensus">
                    <div class="hero-card-label">Shrinking Windows Ensemble</div>
                    <div class="hero-ticker">{top_asset_c}</div>
                    <div class="hero-return">{cons_pred[top_idx_c]*100:+.2f}% consensus return</div>
                    <div class="hero-meta">
                        Signal for &nbsp;&nbsp;<strong>{next_day}</strong><br>
                        Generated &nbsp;&nbsp;<strong>{now_str}</strong><br>
                        Benchmark &nbsp;&nbsp;<strong>{benchmark}</strong>
                    </div>
                    {runners_html(assets, sorted_c, cons_pred)}
                </div>
                """, unsafe_allow_html=True)

        # ── Performance metrics ───────────────────────────────────────────
        st.markdown('<div class="section-heading">Performance Metrics · Test Period</div>', unsafe_allow_html=True)

        metrics_data = load_metrics(module)
        if metrics_data:
            ann_ret, sharpe, max_dd, hit = compute_metrics(
                metrics_data['test_predictions'], metrics_data['test_true']
            )
            st.markdown(f"""
            <div class="metrics-grid">
                <div class="metric-cell">
                    <div class="metric-name">Ann. Return</div>
                    <div class="metric-value {'positive' if ann_ret>=0 else 'negative'}">{ann_ret:+.1f}%</div>
                </div>
                <div class="metric-cell">
                    <div class="metric-name">Sharpe Ratio</div>
                    <div class="metric-value {'positive' if sharpe>=0 else 'negative'}">{sharpe:.2f}</div>
                </div>
                <div class="metric-cell">
                    <div class="metric-name">Max Drawdown</div>
                    <div class="metric-value negative">{max_dd:.1f}%</div>
                </div>
                <div class="metric-cell">
                    <div class="metric-name">Hit Rate</div>
                    <div class="metric-value {'positive' if hit>=50 else 'negative'}">{hit:.1f}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(
                '<p style="font-family:JetBrains Mono,monospace;font-size:0.72rem;'
                'color:#c0c8e0;margin:0;">Metrics not available — run training to generate.</p>',
                unsafe_allow_html=True,
            )

        # ── Signal history ────────────────────────────────────────────────
        st.markdown('<div class="section-heading">Signal History</div>', unsafe_allow_html=True)

        if st.session_state.signal_history:
            hist_df = pd.DataFrame(
                st.session_state.signal_history,
                columns=["Date", "Pick", "Conviction", "Actual Return", "Hit"],
            )
            st.dataframe(hist_df, use_container_width=True, hide_index=True)
        else:
            st.markdown(
                '<p style="font-family:JetBrains Mono,monospace;font-size:0.72rem;'
                'color:#c0c8e0;margin:0;">No signals recorded yet.</p>',
                unsafe_allow_html=True,
            )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="app-footer">'
    'P2‑ETF‑KAN‑ENGINE &nbsp;·&nbsp; Research only &nbsp;·&nbsp; Not financial advice &nbsp;·&nbsp; '
    'Past performance does not guarantee future results'
    '</div>',
    unsafe_allow_html=True,
)
