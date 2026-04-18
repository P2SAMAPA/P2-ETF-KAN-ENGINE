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
FI_ASSETS     = ['GLD', 'TLT', 'VCIT', 'LQD', 'HYG', 'VNQ', 'SLV']
FI_BENCHMARK  = 'AGG'
EQUITY_ASSETS = ['QQQ', 'XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU', 'XME', 'GDX', "IWF", "XSD", "XBI", 'IWM']
EQUITY_BENCHMARK = 'SPY'
MACRO_COLS    = ['VIX', 'DXY', 'T10Y2Y', 'TBILL_3M', 'IG_SPREAD', 'HY_SPREAD']

TRANSACTION_COST   = 0.0012
SEQ_LEN            = 20
HF_REPO            = "P2SAMAPA/p2-etf-kan-engine-results"
HF_REPO_TYPE       = "dataset"   # ← repo_type used everywhere for this repo
RISK_FREE_RATE     = 0.05
MAX_SHARPE_FOR_WEIGHT = 3.0

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    layout="wide", page_title="P2‑ETF‑KAN‑ENGINE",
    page_icon="⚡", initial_sidebar_state="collapsed"
)

# ── CSS ──────────────────────────────────────────────────────────────────────
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
.debug-box{background:#f5f5f5;border:1px solid #ddd;border-radius:4px;padding:0.5rem;font-size:0.8rem;margin-top:0.5rem}
.consensus-info{font-size:0.75rem;color:#666;margin-top:0.5rem;font-style:italic}
.metrics-table{font-size:0.75rem;border-collapse:collapse;width:100%}
.metrics-table th{background:#f0f0f0;padding:4px;text-align:left;border:1px solid #ddd}
.metrics-table td{padding:4px;border:1px solid #ddd}
.metrics-table tr:hover{background:#f9f9f9}
.positive{color:#2e7d32;font-weight:600}
.negative{color:#d32f2f;font-weight:600}
.zero-weight{background:#ffebee}
.valid-weight{background:#e8f5e9}
.top-etf{font-weight:700;color:#1a1a1a}
.sharpe-capped{color:#ff9800;font-size:0.7rem}
</style>
""", unsafe_allow_html=True)

# ── HF Token ─────────────────────────────────────────────────────────────────
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
    ds  = load_dataset("P2SAMAPA/fi-etf-macro-signal-master-data", split="train")
    df  = ds.to_pandas()
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
    """Download a file from the HF dataset repo, trying with and without subfolder."""
    os.makedirs("models", exist_ok=True)
    for attempt in range(max_retries):
        try:
            kwargs = dict(
                repo_id=HF_REPO,
                filename=filename,
                repo_type=HF_REPO_TYPE,
                local_dir="models",
                local_dir_use_symlinks=False,
                token=HF_TOKEN,
            )
            if subfolder:
                kwargs["subfolder"] = subfolder
            return hf_hub_download(**kwargs)
        except Exception as e:
            msg = str(e)
            if any(c in msg for c in ("401", "403", "404")) or "not found" in msg.lower():
                return None
            if attempt == max_retries - 1:
                return None
    return None


def load_model_and_scalers(module, mode='full', start_year=None):
    if mode == 'full':
        mf  = f"kan_{module}_full.pt"
        sxf = f"scaler_X_{module}_full.pkl"
        syf = f"scaler_y_{module}_full.pkl"
        # Try root first, then shrinking_models subfolder as legacy fallback
        paths = [download_file(f, "") for f in (mf, sxf, syf)]
        if not all(paths):
            paths = [download_file(f, "shrinking_models") for f in (mf, sxf, syf)]
    else:
        mf  = f"kan_{module}_shrinking_start{start_year}.pt"
        sxf = f"scaler_X_{module}_shrinking_start{start_year}.pkl"
        syf = f"scaler_y_{module}_shrinking_start{start_year}.pkl"
        paths = [download_file(f, "shrinking_models") for f in (mf, sxf, syf)]
        if not all(paths):
            paths = [download_file(f, "") for f in (mf, sxf, syf)]

    if not all(paths):
        return None, None, None, None

    try:
        scaler_X = joblib.load(paths[1])
        scaler_y = joblib.load(paths[2])
    except Exception:
        return None, None, None, None

    input_dim  = SEQ_LEN * scaler_X.mean_.shape[0]
    output_dim = len(FI_ASSETS) if module == 'fi' else len(EQUITY_ASSETS)

    model = TemporalKANForecaster(
        input_dim, hidden_dims=[256, 128],
        output_dim=output_dim, grid_size=20, seq_len=SEQ_LEN
    )
    try:
        state_dict = torch.load(paths[0], map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()
    except Exception:
        return None, None, None, None

    # Load metrics
    metrics = None
    if mode == 'shrinking':
        mpath = download_file(f"metrics_{module}_shrinking_start{start_year}.pkl", "shrinking_models")
        if not mpath:
            mpath = download_file(f"metrics_{module}_shrinking_start{start_year}.pkl", "")
    else:
        mpath = download_file(f"metrics_{module}_full.pkl", "")
        if not mpath:
            mpath = download_file(f"metrics_{module}_full.pkl", "shrinking_models")

    if mpath:
        try:
            metrics = joblib.load(mpath)
        except Exception:
            pass

    return model, scaler_X, scaler_y, metrics


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


def compute_metrics_fixed(test_pred, test_true, risk_free_annual=RISK_FREE_RATE):
    pred_avg  = np.array(test_pred).mean(axis=1)
    true_avg  = np.array(test_true).mean(axis=1)
    mean_p    = np.mean(pred_avg)
    std_p     = np.std(pred_avg)
    daily_rf  = risk_free_annual / 252
    excess_mean = mean_p - daily_rf
    sharpe    = (excess_mean / std_p) * np.sqrt(252) if std_p > 0 else 0.0
    ann_ret   = mean_p * 252 * 100
    cum       = np.cumprod(1 + pred_avg)
    peak      = np.maximum.accumulate(cum)
    max_dd    = np.min((cum - peak) / peak) * 100
    hit       = np.mean(np.sign(pred_avg) == np.sign(true_avg)) * 100
    pred_var  = np.var(pred_avg)
    return ann_ret, sharpe, max_dd, hit, pred_var


def load_metrics_full(module):
    path = download_file(f"metrics_{module}_full.pkl", "")
    if not path:
        path = download_file(f"metrics_{module}_full.pkl", "shrinking_models")
    if path:
        try:
            return joblib.load(path)
        except Exception:
            pass
    return None


# ── WEIGHTED CONSENSUS ────────────────────────────────────────────────────────
def get_weighted_shrinking_consensus_with_metrics(module, feature_seq, assets):
    try:
        files = list_repo_files(HF_REPO, repo_type=HF_REPO_TYPE, token=HF_TOKEN)
    except Exception:
        return None, None, None

    patterns    = [
        f"shrinking_models/kan_{module}_shrinking_start",
        f"kan_{module}_shrinking_start",
    ]
    model_files = []
    for pattern in patterns:
        model_files.extend([f for f in files if pattern in f and f.endswith(".pt")])

    if not model_files:
        return None, None, None

    processed_years   = set()
    window_data       = []
    all_metrics_display = []

    for mf in model_files:
        try:
            year = int(mf.split("_start")[-1].split(".pt")[0])
        except Exception:
            continue

        if year in processed_years:
            continue
        processed_years.add(year)

        model, sx, sy, metrics = load_model_and_scalers(module, mode='shrinking', start_year=year)

        if model is None:
            all_metrics_display.append({
                'year': year, 'top_etf': 'N/A', 'top_etf_return': None,
                'status': 'Model Load Failed', 'ann_ret': None, 'sharpe': None,
                'max_dd': None, 'sharpe_capped': None, 'weight': 0, 'raw_score': 0,
            })
            continue

        try:
            pred = get_prediction(model, sx, sy, feature_seq)
        except Exception:
            all_metrics_display.append({
                'year': year, 'top_etf': 'N/A', 'top_etf_return': None,
                'status': 'Prediction Failed', 'ann_ret': None, 'sharpe': None,
                'max_dd': None, 'sharpe_capped': None, 'weight': 0, 'raw_score': 0,
            })
            continue

        top_etf_idx = np.argmax(pred)
        top_etf     = assets[top_etf_idx]
        top_etf_return = pred[top_etf_idx]

        if metrics and 'test_predictions' in metrics and 'test_true' in metrics:
            test_pred_m = np.array(metrics['test_predictions'])
            test_true_m = np.array(metrics['test_true'])
            ann_ret, sharpe, max_dd, hit, pred_var = compute_metrics_fixed(test_pred_m, test_true_m)
        else:
            ann_ret  = np.mean(pred) * 252 * 100
            sharpe   = 0
            max_dd   = -10.0
            pred_var = np.var(pred)

        sharpe_capped = min(abs(sharpe), MAX_SHARPE_FOR_WEIGHT) * (1 if sharpe >= 0 else -1)
        is_negative   = ann_ret < 0

        window_data.append({
            'year': year, 'prediction': pred,
            'ann_ret': ann_ret, 'sharpe': sharpe, 'sharpe_capped': sharpe_capped,
            'max_dd': max_dd, 'is_negative': is_negative,
            'top_etf': top_etf, 'top_etf_return': top_etf_return,
        })
        all_metrics_display.append({
            'year': year, 'top_etf': top_etf, 'top_etf_return': top_etf_return,
            'status': 'Valid' if not is_negative else 'Excluded (Negative)',
            'ann_ret': ann_ret, 'sharpe': sharpe, 'sharpe_capped': sharpe_capped,
            'max_dd': max_dd, 'weight': 0, 'raw_score': 0,
        })

    if not window_data:
        return None, None, all_metrics_display

    n_etfs = len(assets)

    window_scores = []
    valid_windows = []
    for i, data in enumerate(window_data):
        if data['is_negative']:
            window_scores.append(0)
        else:
            dd_score    = 100 + data['max_dd']
            sharpe_score = data['sharpe_capped'] * 10
            raw_score   = (0.60 * data['ann_ret']) + (0.20 * dd_score) + (0.10 * sharpe_score)
            window_scores.append(raw_score)
            valid_windows.append(i)
            all_metrics_display[i]['raw_score'] = raw_score

    frequency_scores = np.zeros(n_etfs)
    for etf_idx in range(n_etfs):
        positive_count = sum(1 for w in valid_windows if window_data[w]['prediction'][etf_idx] > 0)
        freq = positive_count / len(valid_windows) * 100 if valid_windows else 0
        frequency_scores[etf_idx] = freq

    final_window_scores = []
    for idx, w in enumerate(valid_windows):
        avg_frequency    = np.mean(frequency_scores)
        freq_component   = 0.10 * avg_frequency
        final_score      = window_scores[w] + freq_component
        final_window_scores.append(final_score)
        all_metrics_display[w]['raw_score'] = final_score

    total_weight = sum(final_window_scores)
    if total_weight == 0:
        return None, None, all_metrics_display

    normalized_weights = [w / total_weight for w in final_window_scores]

    for idx, w in enumerate(valid_windows):
        all_metrics_display[w]['weight'] = normalized_weights[idx]

    weighted_pred = np.zeros(n_etfs)
    for idx, w in enumerate(valid_windows):
        weighted_pred += normalized_weights[idx] * window_data[w]['prediction']

    consensus_info = {
        'valid_windows': len(valid_windows),
        'total_windows': len(window_data),
        'years':   [window_data[w]['year']   for w in valid_windows],
        'weights': normalized_weights,
        'frequency_scores': frequency_scores,
    }

    return weighted_pred, consensus_info, all_metrics_display


# ── UI helpers ────────────────────────────────────────────────────────────────
def runners_html(assets, sorted_idx, pred):
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
    st.markdown(f'<div style="margin:0">{html}</div>', unsafe_allow_html=True)


def render_metrics_table(all_metrics, assets):
    seen_years    = set()
    unique_metrics = []
    for m in sorted(all_metrics, key=lambda x: x['year']):
        if m['year'] not in seen_years:
            seen_years.add(m['year'])
            unique_metrics.append(m)

    rows = ""
    for m in unique_metrics:
        year   = m['year']
        status = m['status']
        ann_ret = f"{m['ann_ret']:+.1f}%" if m['ann_ret'] is not None else "N/A"
        sharpe  = f"{m['sharpe']:.2f}"    if m['sharpe']  is not None else "N/A"
        sharpe_capped_txt = (
            f" (capped: {m['sharpe_capped']:.1f})"
            if m.get('sharpe_capped') and abs(m['sharpe_capped']) != abs(m['sharpe'] or 0)
            else ""
        )
        max_dd   = f"{m['max_dd']:.1f}%"   if m['max_dd']   is not None else "N/A"
        weight   = f"{m['weight']:.1%}"    if m.get('weight', 0) > 0    else "0%"
        top_etf  = m.get('top_etf', 'N/A')
        top_ret  = f"{m['top_etf_return']*100:+.2f}%" if m.get('top_etf_return') is not None else "N/A"

        status_class = (
            "zero-weight"  if "Excluded" in status else
            "valid-weight" if status == "Valid"     else ""
        )
        ann_class = (
            "positive" if m['ann_ret'] and m['ann_ret'] >= 0 else
            "negative" if m['ann_ret'] and m['ann_ret']  < 0 else ""
        )
        rows += (
            f'<tr class="{status_class}">'
            f'<td>{year}</td>'
            f'<td class="top-etf">{top_etf}</td>'
            f'<td>{top_ret}</td>'
            f'<td>{status}</td>'
            f'<td class="{ann_class}">{ann_ret}</td>'
            f'<td>{sharpe}<span class="sharpe-capped">{sharpe_capped_txt}</span></td>'
            f'<td>{max_dd}</td>'
            f'<td>{weight}</td>'
            f'</tr>'
        )

    return (
        '<table class="metrics-table">'
        '<tr><th>Year</th><th>Top ETF</th><th>Top ETF Return</th><th>Status</th>'
        '<th>Ann Return</th><th>Sharpe</th><th>Max DD</th><th>Weight</th></tr>'
        f'{rows}</table>'
    )


# ── App header ────────────────────────────────────────────────────────────────
now_str  = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
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

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Debug Info")
    if 'pred_var_full'      not in st.session_state: st.session_state.pred_var_full      = None
    if 'pred_var_consensus' not in st.session_state: st.session_state.pred_var_consensus = None

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_fi, tab_eq = st.tabs(["Option A — Fixed Income / Alts", "Option B — Equity Sectors"])

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

        # ── LEFT: Full dataset model ─────────────────────────────────────────
        with col_full:
            with st.spinner("Loading full model…"):
                model_full, sx_full, sy_full, _ = load_model_and_scalers(module, mode='full')

            if model_full is None:
                render_html(
                    '<div class="hero-box not-available">'
                    '<div style="font-size:3rem;color:#ddd">◌</div>'
                    '<h3>Model not available</h3>'
                    '<p>Train via GitHub Actions (train_full.yml)</p>'
                    '</div>'
                )
            else:
                pred       = get_prediction(model_full, sx_full, sy_full, feat_seq)
                top_idx    = int(np.argmax(pred))
                top_asset  = assets[top_idx]
                top_ret    = pred[top_idx]

                prev           = st.session_state[f'prev_pick_{module}']
                net_ret, switched = apply_transaction_cost(prev, top_asset, top_ret)
                display_ret    = net_ret if switched else top_ret
                st.session_state[f'prev_pick_{module}'] = top_asset

                sorted_idx   = np.argsort(pred)[::-1]
                switch_html  = (
                    f'<div class="switch-warning">⚠ Switched from {prev} · −12 bps</div>'
                    if switched and prev else ""
                )
                meta = (
                    f'<div class="hero-meta">'
                    f'Signal for &nbsp;&nbsp;<strong>{next_day}</strong><br>'
                    f'Generated &nbsp;&nbsp;<strong>{now_str}</strong><br>'
                    f'Benchmark &nbsp;&nbsp;<strong>{benchmark}</strong>'
                    f'</div>'
                )
                pred_var = np.var(pred)
                st.session_state.pred_var_full = pred_var
                debug_html = f'<div class="debug-box">Pred variance: {pred_var:.6f}</div>'

                render_html(
                    f'<div class="hero-box">'
                    f'<div style="font-size:0.85rem;color:#888;margin-bottom:0.3rem">Full Dataset · 2008–2026</div>'
                    f'<div class="hero-ticker">{top_asset}</div>'
                    f'<div class="hero-return">{display_ret*100:+.2f}% predicted</div>'
                    f'{switch_html}{meta}{runners_html(assets, sorted_idx, pred)}{debug_html}'
                    f'</div>'
                )

        # ── RIGHT: Weighted shrinking consensus ──────────────────────────────
        with col_cons:
            with st.spinner("Loading weighted consensus…"):
                cons_pred, cons_info, all_metrics = \
                    get_weighted_shrinking_consensus_with_metrics(module, feat_seq, assets)

            if cons_pred is None:
                render_html(
                    '<div class="hero-box not-available">'
                    '<div style="font-size:3rem;color:#ddd">◌</div>'
                    '<h3>Ensemble not available</h3>'
                    '<p>Train via GitHub Actions (train_shrinking.yml)</p>'
                    '</div>'
                )
            else:
                top_idx_c   = int(np.argmax(cons_pred))
                top_asset_c = assets[top_idx_c]
                sorted_c    = np.argsort(cons_pred)[::-1]
                meta_c = (
                    f'<div class="hero-meta">'
                    f'Signal for &nbsp;&nbsp;<strong>{next_day}</strong><br>'
                    f'Generated &nbsp;&nbsp;<strong>{now_str}</strong><br>'
                    f'Benchmark &nbsp;&nbsp;<strong>{benchmark}</strong>'
                    f'</div>'
                )
                consensus_html = (
                    f'<div class="consensus-info">'
                    f'{cons_info["valid_windows"]}/{cons_info["total_windows"]} windows used · '
                    f'Sharpe capped at {MAX_SHARPE_FOR_WEIGHT} for weighting'
                    f'</div>'
                ) if cons_info else ''

                pred_var_c = np.var(cons_pred)
                st.session_state.pred_var_consensus = pred_var_c
                debug_html_c = f'<div class="debug-box">Pred variance: {pred_var_c:.6f}</div>'

                render_html(
                    f'<div class="hero-box">'
                    f'<div style="font-size:0.85rem;color:#888;margin-bottom:0.3rem">Weighted Shrinking Consensus</div>'
                    f'<div class="hero-ticker">{top_asset_c}</div>'
                    f'<div class="hero-return">{cons_pred[top_idx_c]*100:+.2f}% consensus</div>'
                    f'{meta_c}{consensus_html}{runners_html(assets, sorted_c, cons_pred)}{debug_html_c}'
                    f'</div>'
                )

        # ── Shrinking metrics expander ────────────────────────────────────────
        if all_metrics and len(all_metrics) > 0:
            with st.expander(f"📊 Shrinking Windows Metrics ({module.upper()})", expanded=False):
                st.markdown("**All years with top ETF pick and metrics:**")
                st.markdown("- 🟢 Green rows: Valid windows (positive return, included in consensus)")
                st.markdown("- 🔴 Red rows: Excluded windows (negative return, zero weight)")
                st.markdown(f"- **Sharpe (capped: X)**: Shows if Sharpe was capped at {MAX_SHARPE_FOR_WEIGHT} for weight calculation")
                st.markdown("- **Weight**: Normalized weight in final consensus (sums to 100%)")
                render_html(render_metrics_table(all_metrics, assets))
                st.markdown(f"""
**Weight Formula (Sharpe capped at {MAX_SHARPE_FOR_WEIGHT}):**
```
Score = (0.60 × AnnReturn) + (0.20 × (100 + MaxDD)) + (0.10 × min(Sharpe, {MAX_SHARPE_FOR_WEIGHT}) × 10) + (0.10 × AvgFreq)
If AnnReturn < 0: Weight = 0 (excluded)
Else: Weight = Score / Sum(All Valid Scores)
```
""")

        # ── Performance metrics ───────────────────────────────────────────────
        st.markdown('<h4 style="margin-top:1.5rem">Performance Metrics · Test Period</h4>', unsafe_allow_html=True)
        metrics_data = load_metrics_full(module)

        if metrics_data:
            ann_ret, sharpe, max_dd, hit, pred_var = compute_metrics_fixed(
                metrics_data['test_predictions'], metrics_data['test_true']
            )
            var_warning = (
                '<br><span style="color:#d32f2f;font-size:0.75rem">⚠ Low variance</span>'
                if pred_var < 0.0001 else ""
            )
            render_html(
                f'<div class="metrics-grid">'
                f'<div class="metric-card"><div class="metric-label">Ann. Return</div>'
                f'<div class="metric-value" style="color:{"#2e7d32" if ann_ret>=0 else "#d32f2f"}">{ann_ret:+.1f}%</div></div>'
                f'<div class="metric-card"><div class="metric-label">Sharpe Ratio</div>'
                f'<div class="metric-value" style="color:{"#2e7d32" if sharpe>=0 else "#d32f2f"}">{sharpe:.2f}</div></div>'
                f'<div class="metric-card"><div class="metric-label">Max Drawdown</div>'
                f'<div class="metric-value" style="color:#d32f2f">{max_dd:.1f}%</div></div>'
                f'<div class="metric-card"><div class="metric-label">Hit Rate</div>'
                f'<div class="metric-value" style="color:{"#2e7d32" if hit>=50 else "#d32f2f"}">{hit:.1f}%</div>'
                f'{var_warning}</div>'
                f'</div>'
            )
            with st.sidebar:
                st.metric(f"{module.upper()} Pred Var", f"{pred_var:.6f}")
        else:
            st.markdown('<div style="color:#999;font-style:italic">Metrics not available</div>', unsafe_allow_html=True)

        # ── Signal history ────────────────────────────────────────────────────
        st.markdown('<h4 style="margin-top:1.5rem">Signal History</h4>', unsafe_allow_html=True)
        if st.session_state.signal_history:
            hist_df = pd.DataFrame(
                st.session_state.signal_history,
                columns=["Date", "Pick", "Conviction", "Actual Return", "Hit"]
            )
            st.dataframe(hist_df, use_container_width=True, hide_index=True)
        else:
            st.markdown('<div style="color:#999;font-style:italic">No signals recorded yet.</div>', unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    '<div style="text-align:center;color:#999;font-size:0.8rem;margin-top:2rem;'
    'padding-top:1rem;border-top:1px solid #eee">Research purposes only · Not financial advice</div>',
    unsafe_allow_html=True
)
