# P2-ETF-KAN-ENGINE

**Kolmogorov-Arnold Network for ETF Return Forecasting**  
Maximise absolute returns · Macro‑pill interpretability · Transaction‑cost aware

---

## Overview

This engine replaces traditional MLP forecasters with **Kolmogorov‑Arnold Networks (KANs)** and **Temporal KANs (T‑KANs)**. Instead of fixed activation functions, KANs learn univariate spline functions on each feature, offering:

- **Higher accuracy** on noisy financial time series  
- **Interpretability** – visualise how each macro variable (VIX, DXY, yield curve, spreads) affects each ETF  
- **Drop‑in replacement** for any forecasting head in a portfolio construction pipeline

The engine is designed for **absolute return maximisation** (no risk‑adjustment penalty) and includes a **transaction cost penalty (12 bps)** to avoid excessive switching of the top pick.

---

## Data

- **Source**: Hugging Face dataset [`P2SAMAPA/fi-etf-macro-signal-master-data`](https://huggingface.co/datasets/P2SAMAPA/fi-etf-macro-signal-master-data)  
- **Frequency**: Daily closing prices (2008–2026 YTD)  
- **Timestamp**: UNIX seconds – converted to NYSE trading calendar  
- **Macro features**: `VIX`, `DXY`, `T10Y2Y`, `TBILL_3M`, `IG_SPREAD`, `HY_SPREAD`  
- **Lagged returns**: 5 days of each ETF in the current module (to capture momentum)

### ETF Modules

| Module       | Assets (for prediction)                     | Benchmark |
|--------------|---------------------------------------------|-----------|
| **FI / Alts** | GLD, TLT, VCIT, LQD, HYG, VNQ, SLV         | AGG       |
| **Equity**    | QQQ, XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, XME, GDX, IWM | SPY |

---

## Training Strategies

The engine supports two evaluation modes – both can be selected in the Streamlit UI.

### 1. Full Dataset (80/10/10 Split)

- **Training**: earliest 80% of the full 2008–2026 YTD period  
- **Validation**: next 10% (for early stopping / hyper‑parameters)  
- **Test**: most recent 10% (out‑of‑sample evaluation)  
- A single KAN model is trained once and used to predict the **next trading day**.

### 2. Shrinking Dataset (Moving Start Date, Fixed End Date)

We generate multiple overlapping training windows, all ending at 2026 YTD:

| Window | Start year | End (YTD) | 80/10/10 split → test set (most recent 10%) |
|--------|------------|-----------|-----------------------------------------------|
| 1      | 2008       | 2026      | e.g. 2024–2026 YTD |
| 2      | 2009       | 2026      | e.g. 2024–2026 YTD (shorter test) |
| 3      | 2010       | 2026      | … |
| …      | …          | …         | … |

For **each window**:
- Train a separate KAN model (150 epochs, CPU)
- Evaluate it on the **test set** (out‑of‑sample for that window)
- Record per‑ETF metrics:
  - Total return (cumulative)
  - Number of positive years in the test period
  - Maximum drawdown (peak‑to‑trough)
  - Sharpe ratio (annualised, using `TBILL_3M` as risk‑free rate)

> **Negative return windows** are excluded entirely from the ETF’s average – they contribute zero to return and positive‑years count, and the window is not averaged.

### Consensus Scoring (Shrinking Windows only)

After all windows are processed, each ETF gets an average of its metrics (excluding negative‑return windows). The final **Consensus Score** is:
Score = 0.50 * (Return percentile)

0.20 * (PositiveYears percentile)

0.10 * (1 - Drawdown percentile)

0.10 * (Sharpe percentile)

The ETF with the highest consensus score is considered the **historically robust best pick** across different start dates.

---

## Prediction for the Next Trading Day

The Streamlit UI shows **two** prediction sources:

1. **Most recent model** – trained on the largest window (2008–2026 YTD)  
2. **Ensemble average** – average predicted return across all shrinking‑window models

For each source, the top ETF (highest predicted raw return) is displayed in a hero box.

### Transaction Cost Penalty (12 bps)

To avoid excessive daily switching:

- The engine remembers the **previous day’s recommended ETF** (per module) using session state.
- When a new top ETF is proposed:
  - If it is **different** from the previous pick, subtract **0.0012 (12 bps)** from its predicted return.
  - Compare the **net return** to the previous pick’s expected return.
  - Only **switch** the recommendation if net return exceeds the hold return.
- The UI shows both gross and net predicted returns, and flags whether a switch occurred.

The **Signal History** table logs the actual ETF recommended (after cost filter) and later the realised return (minus 12 bps if a switch was made) to compute a realistic hit rate.

---

## UI Structure (Streamlit)

The interface follows the design from the P2-ETF-DEEPM-ENGINE:

- **Header**: title + tagline
- **Two tabs / option buttons**:  
  - Option A – Fixed Income / Alts (FI module)  
  - Option B – Equity Sectors
- **Hero box** per tab:  
  - Large ticker symbol  
  - Conviction % (model confidence)  
  - Next trading date (NYSE calendar)  
  - Generation timestamp  
  - Source indicator (Full dataset / Shrinking window / Ensemble)
- **Second & third best** picks with smaller conviction percentages
- **Macro pills**: current VIX, T10Y2Y, HY spread, Stress (or other indicators)
- **Metrics table**: test period, annualised return, Sharpe ratio, max drawdown, hit rate (with horizontal bars)
- **Signal history table**: date, pick, conviction, actual return, hit (boolean)
- **Footer**: research disclaimer

---

## GitHub Actions Training

The `.github/workflows/train.yml` workflow:

- Runs on `ubuntu-latest` (CPU)  
- Triggered on push to `main` or manually (workflow_dispatch)  
- Installs dependencies, loads the HF dataset  
- Trains the KAN model with configurable epochs and window size  
- Saves model checkpoints and metrics  
- Uploads results to [`P2SAMAPA/p2-etf-kan-engine-results`](https://huggingface.co/datasets/P2SAMAPA/p2-etf-kan-engine-results)  

> **Parallel training** for shrinking windows is supported via a matrix strategy to fit within free GitHub Actions limits. Daily runs are initially scheduled to measure compute time.

---

## File Structure (Flat Root)

All code files reside in the repository root (except the workflow YAML):
P2-ETF-KAN-ENGINE/
├── .github/workflows/train.yml
├── train.py # Full + shrinking window training
├── kan_model.py # KANLayer, TemporalKANForecaster
├── streamlit_app.py # UI with tabs, hero boxes, transaction cost logic
├── upload_model.py # Push results to HF dataset
├── requirements.txt
├── README.md
├── .gitignore
└── .env.example

---

## Getting Started

1. **Clone the repo**  
   ```bash
   git clone https://github.com/P2SAMAPA/P2-ETF-KAN-ENGINE.git
   cd P2-ETF-KAN-ENGINE
   License & Disclaimer
This engine is for research and educational purposes only. It does not constitute financial advice. Past performance does not guarantee future results. Use at your own risk.

References
Liu, Z. et al. (2024). KAN: Kolmogorov‑Arnold Networks.

Temporal KAN extensions for financial time series (2024–2025).

NYSE trading calendar: pandas_market_calendars (integrated in UI).
