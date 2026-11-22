"""StockSense: High-Frequency & Big Data Financial Analytics Platform.

Academic Activity-Based Learning (ABL) Platform:
- Big Data ingestion & cleaning with Apache PySpark
- Descriptive, Diagnostic, Predictive, and Prescriptive Analytics
- Chronological Machine Learning predictions with Naive Baseline Benchmarks
- Empirical PySpark vs. Pandas Performance Benchmarking
- Institutional Wall-Street / TradingView Financial Terminal UI
"""

from __future__ import annotations

import io
import sys
from datetime import date, timedelta
from math import sqrt
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── Resolve project root ──────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.analytics import (
    compute_descriptive_stats,
    compute_diagnostic_analytics,
    compute_market_performance_rankings,
    generate_prescriptive_signals,
)
from src.benchmarking import run_full_benchmark
from src.data_fetcher import (
    NSE_PRESETS,
    US_PRESETS,
    get_sample_data_dir,
    get_stock_data,
    get_stock_info,
    list_sample_datasets,
    load_local_dataset,
    parse_uploaded_file,
)
from src.ml_predictor import compare_all_models, train_and_evaluate_model
from src.spark_engine import get_spark_status, process_stock_data_spark

# ─────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StockSense | Financial Analytics Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# ELITE INSTITUTIONAL FINANCIAL TERMINAL THEME (CSS)
# ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    :root {
        --bg-main: #07090e;
        --bg-panel: #0d121d;
        --bg-card: #111726;
        --bg-card-hover: #161e31;
        --border-subtle: #1c2638;
        --border-strong: #2a3852;
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --accent-cyan: #0ea5e9;
        --accent-blue: #38bdf8;
        --accent-indigo: #6366f1;
        --accent-violet: #818cf8;
        --pos-green: #10b981;
        --pos-green-bg: rgba(16, 185, 129, 0.12);
        --neg-red: #f43f5e;
        --neg-red-bg: rgba(244, 63, 94, 0.12);
        --neutral-gold: #f59e0b;
        --neutral-gold-bg: rgba(245, 158, 11, 0.12);
    }

    /* Global Typography & Canvas */
    html, body, [data-testid="stAppViewContainer"],
    [data-testid="stMain"], section.main {
        background-color: var(--bg-main) !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        letter-spacing: -0.01em;
    }

    /* Monospace for Numbers */
    .mono-num {
        font-family: 'JetBrains Mono', monospace !important;
        font-feature-settings: "tnum" 1;
    }

    /* Sidebar Refinement */
    [data-testid="stSidebar"] {
        background-color: var(--bg-panel) !important;
        border-right: 1px solid var(--border-subtle) !important;
        box-shadow: 4px 0 24px rgba(0, 0, 0, 0.4);
    }
    [data-testid="stSidebar"] * {
        color: #cbd5e1 !important;
    }

    /* Top Executive Header Bar */
    .terminal-nav-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: linear-gradient(180deg, #111726 0%, #0d121d 100%);
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        padding: 10px 18px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    .terminal-logo-box {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .terminal-badge-pulse {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(14, 165, 233, 0.12);
        border: 1px solid rgba(14, 165, 233, 0.3);
        color: var(--accent-blue) !important;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.74rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .pulse-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: #38bdf8;
        box-shadow: 0 0 8px #38bdf8;
        animation: pulseAnimation 2s infinite;
    }
    @keyframes pulseAnimation {
        0% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.4; transform: scale(0.85); }
        100% { opacity: 1; transform: scale(1); }
    }

    /* Metric Tiles (KPI Cards) */
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #121827 0%, #0d131f 100%) !important;
        border: 1px solid var(--border-subtle) !important;
        border-top: 2px solid var(--accent-cyan) !important;
        border-radius: 8px !important;
        padding: 14px 16px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3) !important;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: var(--accent-cyan) !important;
    }
    [data-testid="stMetricLabel"] {
        color: var(--text-secondary) !important;
        font-size: 0.74rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
    }
    [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
        font-size: 1.45rem !important;
        font-weight: 700 !important;
        font-family: 'JetBrains Mono', monospace !important;
        letter-spacing: -0.02em !important;
    }
    [data-testid="stMetricDelta"] {
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Chips & Badges */
    .chip-green {
        display: inline-block;
        background: var(--pos-green-bg);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: var(--pos-green) !important;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }
    .chip-red {
        display: inline-block;
        background: var(--neg-red-bg);
        border: 1px solid rgba(244, 63, 94, 0.3);
        color: var(--neg-red) !important;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }
    .chip-blue {
        display: inline-block;
        background: rgba(14, 165, 233, 0.12);
        border: 1px solid rgba(14, 165, 233, 0.3);
        color: #38bdf8 !important;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }
    .chip-gold {
        display: inline-block;
        background: var(--neutral-gold-bg);
        border: 1px solid rgba(245, 158, 11, 0.3);
        color: #fbbf24 !important;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Section Headers */
    .term-header {
        font-size: 0.86rem;
        font-weight: 700;
        color: #38bdf8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 18px 0 10px 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .term-header::after {
        content: "";
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, #1e293b 0%, rgba(30, 41, 59, 0) 100%);
    }

    /* Wall Street Prescriptive Action Card */
    .action-card {
        background: linear-gradient(145deg, #131b2c 0%, #0e1320 100%);
        border: 1px solid var(--border-strong);
        border-radius: 8px;
        padding: 20px 24px;
        margin: 16px 0;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.35);
    }

    /* Stepper Workflow Visualizer */
    .workflow-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #0f1523;
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        padding: 16px 20px;
        margin: 14px 0 20px 0;
        overflow-x: auto;
    }
    .step-item {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        min-width: 110px;
    }
    .step-badge {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background: #1e293b;
        color: #94a3b8;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.85rem;
        margin-bottom: 6px;
        border: 1px solid #334155;
    }
    .step-badge.active {
        background: linear-gradient(135deg, #0ea5e9, #6366f1);
        color: #ffffff;
        border: none;
        box-shadow: 0 0 12px rgba(14, 165, 233, 0.5);
    }
    .step-title {
        font-size: 0.78rem;
        font-weight: 600;
        color: #f1f5f9;
        margin-bottom: 2px;
    }
    .step-desc {
        font-size: 0.68rem;
        color: #64748b;
    }
    .step-arrow {
        color: #334155;
        font-size: 1.2rem;
        margin: 0 4px;
    }

    /* 52-Week Range Progress Gauge */
    .range-bar-container {
        width: 100%;
        margin-top: 6px;
    }
    .range-bar-bg {
        width: 100%;
        height: 4px;
        background: #1e293b;
        border-radius: 2px;
        position: relative;
    }
    .range-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, #0ea5e9, #38bdf8);
        border-radius: 2px;
    }
    .range-pin {
        position: absolute;
        top: -4px;
        width: 4px;
        height: 12px;
        background: #f8fafc;
        border-radius: 2px;
        box-shadow: 0 0 6px rgba(255, 255, 255, 0.8);
    }

    /* Streamlit Tabs & Buttons */
    [data-testid="stTabs"] button {
        color: #94a3b8 !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        padding: 8px 16px !important;
    }
    [data-testid="stTabs"] button[aria-selected="true"] {
        color: #38bdf8 !important;
        border-bottom: 2px solid #38bdf8 !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        padding: 8px 16px !important;
        box-shadow: 0 2px 10px rgba(2, 132, 199, 0.3) !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        box-shadow: 0 4px 16px rgba(2, 132, 199, 0.5) !important;
        transform: translateY(-1px);
    }
    hr {
        border-color: var(--border-subtle) !important;
        margin: 16px 0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────
# PLOTLY THEME & COLOR PALETTE
# ─────────────────────────────────────────────────────────────
PLOTLY_TEMPLATE = "plotly_dark"
PLOTLY_BG = "#07090e"
PLOTLY_CARD_BG = "#0d121d"
GRID_COLOR = "#182234"

PALETTE = [
    "#38bdf8", "#34d399", "#f43f5e", "#a78bfa",
    "#fb923c", "#f472b6", "#facc15", "#4ade80",
]

def _plotly_layout(title: str = "", height: int = 400, **extra) -> dict:
    """Returns a unified, high-contrast Plotly layout matching the terminal theme."""
    return dict(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title=dict(text=title, font=dict(family="Inter, sans-serif", size=13, color="#e2e8f0")),
        font=dict(family="Inter, sans-serif", color="#94a3b8", size=11),
        height=height,
        margin=dict(l=45, r=20, t=45, b=35),
        legend=dict(
            bgcolor="rgba(13, 18, 29, 0.6)",
            bordercolor="#1e293b",
            borderwidth=1,
            font=dict(color="#cbd5e1", size=11),
        ),
        xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor="#223147", showline=False),
        yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor="#223147", showline=False),
        **extra,
    )

def _color_for(ticker: str, ticker_list: list[str]) -> str:
    idx = ticker_list.index(ticker) if ticker in ticker_list else 0
    return PALETTE[idx % len(PALETTE)]

def _get_close_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return 2D DataFrame of Close prices for multi-ticker data."""
    if isinstance(df.columns, pd.MultiIndex):
        first_level = df.columns.get_level_values(0)
        field = "Adj Close" if "Adj Close" in first_level else "Close"
        return df[field].copy()
    if "Adj Close" in df.columns:
        return df[["Adj Close"]].copy()
    if "Close" in df.columns:
        return df[["Close"]].copy()
    return df.iloc[:, [0]].copy()

# ─────────────────────────────────────────────────────────────
# CACHED DATA LOADER
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _cached_api_load(tickers: tuple[str, ...], period: str) -> pd.DataFrame:
    return get_stock_data(tickers=list(tickers), period=period)

@st.cache_data(show_spinner=False)
def _cached_stock_info(ticker: str) -> dict[str, Any]:
    try:
        return get_stock_info(ticker)
    except Exception:
        return {"company_name": ticker, "market_cap": None, "sector": "N/A", "currency": "USD"}

# ─────────────────────────────────────────────────────────────
# TOP EXECUTIVE BRANDING BANNER
# ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="terminal-nav-header">
        <div class="terminal-logo-box">
            <div style="background: linear-gradient(135deg, #0ea5e9, #6366f1); width: 34px; height: 34px; border-radius: 6px; display: flex; align-items: center; justify-content: center; font-weight: 800; color: white; font-size: 1rem; box-shadow: 0 0 12px rgba(14,165,233,0.4);">⚡</div>
            <div>
                <span style="font-weight: 700; font-size: 1.15rem; color: #f8fafc; letter-spacing: -0.02em;">StockSense</span>
                <span style="font-size: 0.72rem; color: #64748b; margin-left: 8px; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600;">Enterprise Terminal v2.4</span>
            </div>
        </div>
        <div style="display: flex; align-items: center; gap: 10px;">
            <div class="terminal-badge-pulse">
                <div class="pulse-dot"></div>
                Apache Spark 3.5 Active
            </div>
            <div style="background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.3); color: #34d399; padding: 4px 10px; border-radius: 20px; font-size: 0.74rem; font-weight: 600; letter-spacing: 0.04em;">
                NASDAQ · NYSE · NSE Feed
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION & DATA CONTROLS
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 4px 0 12px 0;">
            <div style="font-weight: 700; font-size: 1.05rem; color: #f8fafc;">Terminal Navigation</div>
            <div style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em;">ABL Research Dossier</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation Menu",
        [
            "Market Overview",
            "Stock Analysis",
            "Big Data Processing",
            "Big Data Analytics",
            "Stock Prediction",
            "Performance Benchmark",
            "Project Information",
        ],
        index=0,
        label_visibility="collapsed",
    )
    st.markdown("---")

    st.markdown('<p class="term-header">Data Feed Ingestion</p>', unsafe_allow_html=True)
    ingestion_mode = st.selectbox(
        "Ingestion Feed",
        ["Live / Yahoo Finance API", "Local Offline Datasets", "Upload File (CSV/Parquet)"],
        index=0,
    )

    df: pd.DataFrame = pd.DataFrame()
    data_origin_tag = "LIVE_API"
    data_origin_label = "Yahoo Finance API"
    selected_tickers: list[str] = []

    # Required ABL Universe
    CORE_STOCK_PRESETS = ["NVDA", "AAPL", "MSFT", "RELIANCE.NS", "TCS.NS"]

    if ingestion_mode == "Live / Yahoo Finance API":
        market_universe = st.selectbox(
            "Watchlist Basket",
            [
                "ABL Core Set (NVDA, AAPL, MSFT, RELIANCE, TCS)",
                "US Mega-Cap & Indices (NVDA, AAPL, MSFT, VOO, SPY)",
                "Indian NSE Bluechips (RELIANCE.NS, TCS.NS, INFY.NS)",
                "Custom Equities",
            ],
            index=0,
        )

        if "ABL Core Set" in market_universe:
            default_selection = CORE_STOCK_PRESETS
            options_pool = CORE_STOCK_PRESETS + ["GOOGL", "AMZN", "INFY.NS"]
        elif "US Mega-Cap" in market_universe:
            default_selection = ["NVDA", "AAPL", "MSFT", "VOO"]
            options_pool = US_PRESETS
        elif "Indian NSE" in market_universe:
            default_selection = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
            options_pool = NSE_PRESETS
        else:
            default_selection = ["NVDA", "AAPL", "RELIANCE.NS"]
            options_pool = CORE_STOCK_PRESETS + US_PRESETS + NSE_PRESETS

        chosen_tickers = st.multiselect("Selected Equities", options=options_pool, default=default_selection)
        custom_input = st.text_input("Enter Ticker Symbol", "", placeholder="e.g. TSLA, HDFCBANK.NS")
        custom_tokens = [t.strip().upper() for t in custom_input.split(",") if t.strip()]

        seen = set()
        for t in chosen_tickers + custom_tokens:
            if t not in seen:
                seen.add(t)
                selected_tickers.append(t)

        if not selected_tickers:
            selected_tickers = CORE_STOCK_PRESETS[:3]

        today = date.today()
        default_start = today - timedelta(days=365)
        start_date = st.date_input("Start Window", value=default_start, max_value=today - timedelta(days=7))
        end_date = st.date_input("End Window", value=today, min_value=start_date + timedelta(days=7), max_value=today)

        days_diff = max((end_date - start_date).days, 1)
        period = "1y" if days_diff <= 370 else ("2y" if days_diff <= 740 else "5y")

        if st.button("Purge & Refresh Ingestion Cache", use_container_width=True):
            st.cache_data.clear()
            st.success("Cache cleared — pulling latest quotes.")

        with st.spinner("Streaming market data..."):
            try:
                raw_df = _cached_api_load(tuple(sorted(selected_tickers)), period)
                mask = (raw_df.index.date >= start_date) & (raw_df.index.date <= end_date)
                df = raw_df.loc[mask]
                data_origin_tag = "LIVE_API"
                data_origin_label = f"Live Market Feed ({len(df)} days)"
            except Exception as err:
                st.error(f"API Ingestion notice: {err}")
                st.info("Tip: Select 'Local Offline Datasets' in sidebar for offline mode.")
                st.stop()

    elif ingestion_mode == "Local Offline Datasets":
        sample_files = list_sample_datasets()
        if not sample_files:
            st.warning("No sample files found in data/sample/. Run scripts/generate_sample_data.py.")
            st.stop()

        sample_labels = [f"{s['ticker']} ({s['format']})" for s in sample_files]
        chosen_sample_lbl = st.selectbox("Select Offline Dataset", sample_labels, index=0)
        chosen_file = sample_files[sample_labels.index(chosen_sample_lbl)]

        try:
            df = load_local_dataset(chosen_file["path"])
            selected_tickers = [chosen_file["ticker"]]
            data_origin_tag = "OFFLINE_SAMPLE"
            data_origin_label = f"Local Offline File ({chosen_file['filename']})"
        except Exception as exc:
            st.error(f"File load error: {exc}")
            st.stop()

    elif ingestion_mode == "Upload File (CSV/Parquet)":
        uploaded_file = st.file_uploader("Upload Market File (.csv, .parquet)", type=["csv", "parquet"])
        if uploaded_file is not None:
            try:
                df, up_ticker = parse_uploaded_file(uploaded_file)
                selected_tickers = [up_ticker]
                data_origin_tag = "USER_UPLOAD"
                data_origin_label = f"Uploaded File ({uploaded_file.name})"
            except Exception as exc:
                st.error(f"Upload error: {exc}")
                st.stop()
        else:
            st.info("Upload a historical stock CSV or Parquet file to proceed.")
            st.stop()

    st.markdown("---")
    st.markdown('<p class="term-header">Cluster Telemetry</p>', unsafe_allow_html=True)
    spark_status = get_spark_status()
    if spark_status["pyspark_installed"]:
        st.markdown('<span class="chip-blue">Engine: Apache Spark 3.5</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="chip-gold">Engine: Pandas Fallback</span>', unsafe_allow_html=True)

    if data_origin_tag == "LIVE_API":
        st.markdown('<span class="chip-green">Feed: Real-Time Active</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="chip-gold">Feed: Offline / Cached</span>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# DATA SANITY CHECK
# ─────────────────────────────────────────────────────────────
if df.empty:
    st.warning("Dataset is empty. Please select active tickers or verify date range.")
    st.stop()

close_matrix = _get_close_matrix(df)
available_tickers = close_matrix.columns.tolist() if isinstance(close_matrix, pd.DataFrame) else [selected_tickers[0]]
if not available_tickers:
    available_tickers = selected_tickers


# ══════════════════════════════════════════════════════════════
# PAGE A: MARKET OVERVIEW
# ══════════════════════════════════════════════════════════════
if page == "Market Overview":
    st.markdown("### Market Overview")
    st.caption(f"Real-Time Telemetry & Cross-Asset Multi-Index Monitoring · {data_origin_label}")
    st.markdown("---")

    # 1. Primary KPI Grid with 52-Week Range Gauge
    st.markdown('<p class="term-header">Live Market Snapshot & 52-Week Range Position</p>', unsafe_allow_html=True)
    daily_returns_df = close_matrix.pct_change()
    
    cols = st.columns(min(len(available_tickers), 5))
    for i, ticker in enumerate(available_tickers[:5]):
        col = cols[i % len(cols)]
        s = close_matrix[ticker].dropna() if ticker in close_matrix else close_matrix.iloc[:, 0].dropna()
        if s.empty:
            continue

        current_val = float(s.iloc[-1])
        daily_chg = float(daily_returns_df[ticker].iloc[-1] * 100) if ticker in daily_returns_df else 0.0

        # Calculate 52-week High and Low
        high_52w = float(s.max())
        low_52w = float(s.min())

        # Calculate position percentage within 52-week range (0% at low, 100% at high)
        if high_52w > low_52w:
            pos_pct = min(max(((current_val - low_52w) / (high_52w - low_52w)) * 100, 0), 100)
        else:
            pos_pct = 50.0

        # Metadata
        info = _cached_stock_info(ticker)
        mcap = info.get("market_cap")
        if mcap:
            mcap_str = f"${mcap/1e12:.2f}T" if mcap >= 1e12 else (f"${mcap/1e9:.2f}B" if mcap >= 1e9 else f"${mcap/1e6:.1f}M")
        else:
            mcap_str = "N/A"

        curr_sym = "₹" if ".NS" in ticker else "$"
        with col:
            st.metric(
                label=ticker,
                value=f"{curr_sym}{current_val:,.2f}",
                delta=f"{daily_chg:+.2f}% today",
            )
            # Custom 52-week visual gauge
            st.markdown(
                f"""
                <div class="range-bar-container">
                    <div style="display: flex; justify-content: space-between; font-size: 0.68rem; color: #64748b; font-family: monospace;">
                        <span>{curr_sym}{low_52w:,.1f}</span>
                        <span style="color: #94a3b8; font-weight: 600;">52W RANGE</span>
                        <span>{curr_sym}{high_52w:,.1f}</span>
                    </div>
                    <div class="range-bar-bg">
                        <div class="range-bar-fill" style="width: {pos_pct:.1f}%;"></div>
                        <div class="range-pin" style="left: calc({pos_pct:.1f}% - 2px);"></div>
                    </div>
                    <div style="font-size: 0.70rem; color: #94a3b8; margin-top: 4px;">
                        Cap: <span style="color: #cbd5e1; font-weight: 600;">{mcap_str}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # 2. Normalized Performance Chart (Base = 100)
    st.markdown('<p class="term-header">Relative Cumulative Performance (Indexed to Base = 100)</p>', unsafe_allow_html=True)
    normalized = (close_matrix / close_matrix.iloc[0]) * 100.0

    fig_norm = go.Figure()
    for ticker in available_tickers:
        fig_norm.add_trace(
            go.Scatter(
                x=normalized.index,
                y=normalized[ticker].round(2),
                mode="lines",
                name=ticker,
                line=dict(width=2.0, color=_color_for(ticker, available_tickers)),
                hovertemplate=f"<b>{ticker}</b><br>Date: %{{x|%b %d, %Y}}<br>Indexed Value: %{{y:.2f}}<extra></extra>",
            )
        )
    fig_norm.add_hline(
        y=100,
        line_dash="dot",
        line_color="#475569",
        annotation_text="Base 100 Baseline",
        annotation_font_color="#94a3b8",
        annotation_position="bottom right",
    )
    fig_norm.update_layout(**_plotly_layout(height=420))
    st.plotly_chart(fig_norm, use_container_width=True)

    # 3. Top Gainers & Losers + Performance Rankings
    st.markdown('<p class="term-header">Watchlist Performance & Volatility Rankings</p>', unsafe_allow_html=True)
    rank_df = compute_market_performance_rankings(df, available_tickers)
    if not rank_df.empty:
        col_t1, col_t2 = st.columns([3, 1])
        with col_t1:
            st.dataframe(rank_df, use_container_width=True, hide_index=True)
        with col_t2:
            st.markdown("##### Performance Extremes")
            best_asset = rank_df.iloc[0]
            worst_asset = rank_df.iloc[-1]
            ret_col = (
                "Period Return (%)" if "Period Return (%)" in rank_df.columns
                else ("Period Return" if "Period Return" in rank_df.columns
                else [c for c in rank_df.columns if "return" in c.lower()][0])
            )
            try:
                best_ret = float(str(best_asset[ret_col]).replace("%", "").replace("+", "").strip())
                worst_ret = float(str(worst_asset[ret_col]).replace("%", "").replace("+", "").strip())
            except Exception:
                best_ret = 0.0
                worst_ret = 0.0

            st.markdown(
                f"""
                <div style="background: linear-gradient(145deg, #121827 0%, #0d131f 100%); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 6px; padding: 14px; margin-bottom: 10px;">
                    <div style="font-size: 0.68rem; color: #10b981; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">Top Outperformer</div>
                    <div style="display: flex; justify-content: space-between; align-items: baseline; margin-top: 4px;">
                        <span style="color: #f8fafc; font-size: 1.25rem; font-weight: 700;">{best_asset['Ticker']}</span>
                        <span class="chip-green">+{best_ret:.2f}%</span>
                    </div>
                </div>
                <div style="background: linear-gradient(145deg, #121827 0%, #0d131f 100%); border: 1px solid rgba(244, 63, 94, 0.3); border-radius: 6px; padding: 14px;">
                    <div style="font-size: 0.68rem; color: #f43f5e; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">Lagging Instrument</div>
                    <div style="display: flex; justify-content: space-between; align-items: baseline; margin-top: 4px;">
                        <span style="color: #f8fafc; font-size: 1.25rem; font-weight: 700;">{worst_asset['Ticker']}</span>
                        <span class="chip-red">{worst_ret:.2f}%</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════
# PAGE B: STOCK ANALYSIS
# ══════════════════════════════════════════════════════════════
elif page == "Stock Analysis":
    st.markdown("### Stock Analysis")
    st.caption("Technical OHLCV Candlesticks, Moving Averages, Volatility, and Cross-Asset Correlation")
    st.markdown("---")

    col_s1, col_s2 = st.columns([2, 1])
    with col_s1:
        sel_stock = st.selectbox("Select Target Equity", options=available_tickers, index=0)
    with col_s2:
        info_meta = _cached_stock_info(sel_stock)
        st.markdown(
            f"""
            <div style="background: #0f1523; border: 1px solid #1e293b; border-radius: 6px; padding: 10px 14px; margin-top: 4px;">
                <span style="font-size: 0.75rem; color: #94a3b8;">Sector: <b>{info_meta.get('sector', 'N/A')}</b></span> &nbsp;|&nbsp;
                <span style="font-size: 0.75rem; color: #94a3b8;">Currency: <b>{info_meta.get('currency', 'USD')}</b></span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Extract OHLCV series
    if isinstance(df.columns, pd.MultiIndex):
        t_data = df.xs(sel_stock, axis=1, level=1).copy() if sel_stock in df.columns.get_level_values(1) else df.copy()
    else:
        t_data = df.copy()

    open_s = t_data["Open"] if "Open" in t_data else t_data.iloc[:, 0]
    high_s = t_data["High"] if "High" in t_data else t_data.iloc[:, 0]
    low_s = t_data["Low"] if "Low" in t_data else t_data.iloc[:, 0]
    close_s = t_data["Adj Close"] if "Adj Close" in t_data else (t_data["Close"] if "Close" in t_data else t_data.iloc[:, 0])
    volume_s = t_data["Volume"] if "Volume" in t_data else pd.Series(0, index=t_data.index)

    # Technical indicators
    sma_20 = close_s.rolling(20, min_periods=1).mean()
    sma_50 = close_s.rolling(50, min_periods=1).mean()

    # RSI (14-day)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi_s = 100 - (100 / (1 + rs))

    # Triple Subplots: Candlestick + Volume + RSI (TradingView aesthetic)
    fig_stock = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.62, 0.19, 0.19],
        subplot_titles=(
            f"{sel_stock} Price Action & Trend (SMA 20 / SMA 50)",
            "Trading Volume Analysis",
            "Relative Strength Index (RSI 14)",
        ),
    )

    # 1. Candlestick & SMAs
    fig_stock.add_trace(
        go.Candlestick(
            x=t_data.index,
            open=open_s,
            high=high_s,
            low=low_s,
            close=close_s,
            name="OHLC",
            increasing_line_color="#10b981",
            decreasing_line_color="#f43f5e",
        ),
        row=1, col=1,
    )
    fig_stock.add_trace(
        go.Scatter(x=t_data.index, y=sma_20, mode="lines", name="SMA 20", line=dict(color="#fb923c", width=1.5)),
        row=1, col=1,
    )
    fig_stock.add_trace(
        go.Scatter(x=t_data.index, y=sma_50, mode="lines", name="SMA 50", line=dict(color="#818cf8", width=1.5, dash="dash")),
        row=1, col=1,
    )

    # 2. Volume Bars
    vol_colors = np.where(close_s.diff() >= 0, "#10b981", "#f43f5e")
    fig_stock.add_trace(
        go.Bar(x=t_data.index, y=volume_s, name="Volume", marker_color=vol_colors.tolist(), opacity=0.85),
        row=2, col=1,
    )

    # 3. RSI Oscillator
    fig_stock.add_trace(
        go.Scatter(x=t_data.index, y=rsi_s, name="RSI 14", line=dict(color="#38bdf8", width=1.5)),
        row=3, col=1,
    )
    fig_stock.add_hline(y=70, line_dash="dot", line_color="#f43f5e", annotation_text="Overbought (70)", row=3, col=1)
    fig_stock.add_hline(y=30, line_dash="dot", line_color="#10b981", annotation_text="Oversold (30)", row=3, col=1)

    fig_stock.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#94a3b8", size=11),
        height=650,
        margin=dict(l=45, r=20, t=35, b=20),
        xaxis_rangeslider_visible=False,
        showlegend=True,
    )
    fig_stock.update_yaxes(gridcolor=GRID_COLOR)
    fig_stock.update_xaxes(gridcolor=GRID_COLOR)
    st.plotly_chart(fig_stock, use_container_width=True)

    # Correlation Matrix
    if len(available_tickers) >= 2:
        st.markdown('<p class="term-header">Cross-Asset Return Correlation Heatmap</p>', unsafe_allow_html=True)
        ret_df = close_matrix.pct_change().dropna()
        corr_matrix = ret_df.corr()

        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            aspect="auto",
            template=PLOTLY_TEMPLATE,
        )
        fig_corr.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=340,
            margin=dict(l=40, r=40, t=30, b=30),
        )
        st.plotly_chart(fig_corr, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE C: BIG DATA PROCESSING (ACADEMIC HIGHLIGHT)
# ══════════════════════════════════════════════════════════════
elif page == "Big Data Processing":
    st.markdown("### Big Data Processing Engine")
    st.caption("Distributed ETL Pipeline with Apache Spark (PySpark), Partitioning, and Parquet Storage")
    st.markdown("---")

    # High-End Visual Stepper
    st.markdown(
        """
        <div class="workflow-container">
            <div class="step-item">
                <div class="step-badge active">01</div>
                <div class="step-title">Ingestion</div>
                <div class="step-desc">Yahoo / NSE / Parquet</div>
            </div>
            <div class="step-arrow">→</div>
            <div class="step-item">
                <div class="step-badge active">02</div>
                <div class="step-title">Validation</div>
                <div class="step-desc">Strict Schema Typing</div>
            </div>
            <div class="step-arrow">→</div>
            <div class="step-item">
                <div class="step-badge active">03</div>
                <div class="step-title">Hygiene</div>
                <div class="step-desc">IQR Outliers & Null Fills</div>
            </div>
            <div class="step-arrow">→</div>
            <div class="step-item">
                <div class="step-badge active">04</div>
                <div class="step-title">Window ETL</div>
                <div class="step-desc">Distributed Rolling SMAs</div>
            </div>
            <div class="step-arrow">→</div>
            <div class="step-item">
                <div class="step-badge active">05</div>
                <div class="step-title">Lake Storage</div>
                <div class="step-desc">Partitioned Parquet</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_ctrl1, col_ctrl2 = st.columns([2, 1])
    with col_ctrl1:
        target_p_ticker = st.selectbox("Select Target Equity for Processing Pipeline", options=available_tickers, index=0)
    with col_ctrl2:
        st.write("")
        st.write("")
        execute_button = st.button("Execute Processing Pipeline", use_container_width=True)

    proc_dir = str(Path(__file__).resolve().parent.parent / "data" / "processed")
    spark_res = process_stock_data_spark(df, ticker=target_p_ticker, output_dir=proc_dir)

    # Engine Status Indicators
    engine_badge = (
        '<span class="chip-blue">Apache PySpark 3.5 (Distributed Local)</span>'
        if "PySpark" in spark_res["engine"]
        else '<span class="chip-gold">Pandas Processing Engine</span>'
    )
    st.markdown(f"**Active Execution Engine:** {engine_badge} &nbsp;|&nbsp; **Latency:** `{spark_res['duration_seconds']}s`", unsafe_allow_html=True)
    st.markdown("")

    # Audit Metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Raw Ingested Records", f"{spark_res['raw_records']:,}")
    c2.metric("Clean Output Records", f"{spark_res['processed_records']:,}")
    c3.metric("Missing Values Handled", f"{spark_res['missing_values_handled']:,}")
    c4.metric("Duplicates Deduplicated", f"{spark_res['duplicates_removed']:,}")
    c5.metric("Outliers Detected (IQR)", f"{spark_res['outliers_detected']:,}")

    st.markdown("---")

    # Before and After Previews
    tab_raw, tab_clean, tab_schema = st.tabs(["Raw Ingestion Preview", "Cleaned & Transformed (Spark)", "DataFrame Schema"])

    with tab_raw:
        st.markdown("##### Unprocessed Ingested Records")
        st.dataframe(df.head(8), use_container_width=True)

    with tab_clean:
        st.markdown("##### Processed Feature-Engineered Output")
        st.dataframe(spark_res["preview_df"], use_container_width=True)
        if spark_res["output_parquet"]:
            st.info(f"Columnar artifact persisted: `{Path(spark_res['output_parquet']).name}`")

    with tab_schema:
        st.markdown("##### Verified Data Types & Partitions")
        schema_df = pd.DataFrame(spark_res["schema"], columns=["Column Identifier", "Spark Data Type"])
        st.dataframe(schema_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# PAGE D: BIG DATA ANALYTICS
# ══════════════════════════════════════════════════════════════
elif page == "Big Data Analytics":
    st.markdown("### Big Data Analytics")
    st.caption("Four Analytical Pillars: Descriptive, Diagnostic, Predictive, and Prescriptive")
    st.markdown("---")

    sel_an_ticker = st.selectbox("Select Equity to Analyze", options=available_tickers, index=0)

    tab_desc, tab_diag, tab_pred, tab_pres = st.tabs([
        "Descriptive Analytics",
        "Diagnostic Analytics",
        "Predictive Analytics",
        "Prescriptive Analytics",
    ])

    # 1. Descriptive
    with tab_desc:
        st.markdown('<p class="term-header">Distribution Moments & Statistical Summaries</p>', unsafe_allow_html=True)
        desc_data = compute_descriptive_stats(df, sel_an_ticker)

        d1, d2, d3, d4 = st.columns(4)
        curr_sym = "₹" if ".NS" in sel_an_ticker else "$"
        d1.metric("Mean Price", f"{curr_sym}{desc_data['mean_price']:,.2f}")
        d2.metric("Median Price", f"{curr_sym}{desc_data['median_price']:,.2f}")
        d3.metric("Annualized Return", f"{desc_data['annualized_return_pct']:+.2f}%")
        d4.metric("Annualized Volatility", f"{desc_data['annualized_volatility_pct']:.2f}%")

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.markdown("##### Statistical Summary Table")
            summary_table = pd.DataFrame([
                {"Statistic": "Price Standard Deviation", "Value": f"{curr_sym}{desc_data['price_std']}"},
                {"Statistic": "52-Week Minimum", "Value": f"{curr_sym}{desc_data['min_price']}"},
                {"Statistic": "52-Week Maximum", "Value": f"{curr_sym}{desc_data['max_price']}"},
                {"Statistic": "Daily Mean Return", "Value": f"{desc_data['daily_mean_return_pct']:+.3f}%"},
                {"Statistic": "Skewness (Tail Asymmetry)", "Value": desc_data["skewness"]},
                {"Statistic": "Kurtosis (Tail Fatness)", "Value": desc_data["kurtosis"]},
            ])
            st.dataframe(summary_table, use_container_width=True, hide_index=True)

        with col_d2:
            st.markdown("##### Daily Returns Probability Density")
            dr_series = desc_data["daily_returns_series"] * 100.0
            fig_dist = px.histogram(
                dr_series,
                nbins=35,
                template=PLOTLY_TEMPLATE,
                color_discrete_sequence=["#38bdf8"],
            )
            fig_dist.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=260,
                margin=dict(l=40, r=20, t=20, b=30),
                showlegend=False,
                xaxis_title="Daily Return (%)",
            )
            st.plotly_chart(fig_dist, use_container_width=True)

    # 2. Diagnostic
    with tab_diag:
        st.markdown('<p class="term-header">Risk Diagnostics, Drawdowns, and Anomaly Detection</p>', unsafe_allow_html=True)
        diag_data = compute_diagnostic_analytics(df, sel_an_ticker)

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Maximum Drawdown", f"{diag_data['max_drawdown_pct']:.2f}%")
        r2.metric("Value-at-Risk (95% Daily)", f"{diag_data['var_95_pct']:.2f}%")
        r3.metric("Conditional VaR (CVaR)", f"{diag_data['cvar_95_pct']:.2f}%")
        r4.metric("Sortino Ratio", f"{diag_data['sortino_ratio']:.2f}")

        # Drawdown Curve
        dd_s = diag_data["drawdown_series"] * 100.0
        fig_dd = go.Figure()
        fig_dd.add_trace(
            go.Scatter(
                x=dd_s.index,
                y=dd_s,
                mode="lines",
                fill="tozeroy",
                line=dict(color="#f43f5e", width=1.5),
                name="Drawdown %",
            )
        )
        fig_dd.update_layout(**_plotly_layout(f"{sel_an_ticker} Historical Drawdown Curve (%)", height=260))
        st.plotly_chart(fig_dd, use_container_width=True)

        # Anomaly log
        col_an1, col_an2 = st.columns(2)
        with col_an1:
            st.markdown(f"##### Volume Anomalies ($Z > 2.5$): `{diag_data['volume_anomalies_count']} days`")
            if diag_data["volume_anomalies"]:
                st.dataframe(pd.DataFrame(diag_data["volume_anomalies"]), use_container_width=True, hide_index=True)
            else:
                st.caption("No volume spikes exceeding 2.5 standard deviations.")
        with col_an2:
            st.markdown(f"##### Price Shocks ($|Z| > 3.0$): `{diag_data['return_anomalies_count']} days`")
            if diag_data["return_anomalies"]:
                st.dataframe(pd.DataFrame(diag_data["return_anomalies"]), use_container_width=True, hide_index=True)
            else:
                st.caption("No extreme price return anomalies exceeding 3.0 standard deviations.")

    # 3. Predictive (Summary link to dedicated module)
    with tab_pred:
        st.markdown('<p class="term-header">Machine Learning Predictive Diagnostics</p>', unsafe_allow_html=True)
        st.markdown(
            """
            Predictive modeling leverages supervised regression trained with chronological splitting.
            For full model tuning, residual distributions, and feature importance analysis, 
            navigate to the **Stock Prediction** section.
            """
        )
        t_ohlcv = df.xs(sel_an_ticker, axis=1, level=1).copy() if isinstance(df.columns, pd.MultiIndex) and sel_an_ticker in df.columns.get_level_values(1) else df.copy()
        try:
            quick_comp = compare_all_models(t_ohlcv, horizon=1, test_size=0.20)
            st.dataframe(quick_comp, use_container_width=True, hide_index=True)
        except Exception as exc:
            st.info(f"Model initialization: {exc}")

    # 4. Prescriptive
    with tab_pres:
        st.markdown('<p class="term-header">Rule-Based Decision Support Signals & Risk Classification</p>', unsafe_allow_html=True)
        pres_data = generate_prescriptive_signals(df, sel_an_ticker)

        if "status" in pres_data:
            st.warning(pres_data["status"])
        else:
            p1, p2, p3 = st.columns(3)
            p1.metric("Educational Posture", pres_data["action"])
            p2.metric("RSI (14-day)", f"{pres_data['rsi_value']} ({pres_data['rsi_signal']})")
            p3.metric("Moving Average Cross", pres_data["trend_signal"])

            st.markdown(
                f"""
                <div class="action-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                        <span style="font-size: 1.15rem; font-weight: 700; color: {pres_data['action_color']};">
                            STRATEGIC POSTURE: {pres_data['action']}
                        </span>
                        <span class="chip-blue">{pres_data['risk_tier']}</span>
                    </div>
                    <div style="font-weight: 600; color: #cbd5e1; font-size: 0.90rem; margin-bottom: 8px;">
                        Quantitative Signal Rationales:
                    </div>
                    <ul style="color: #94a3b8; font-size: 0.88rem; margin: 0; padding-left: 20px; line-height: 1.6;">
                        {"".join([f"<li>{r}</li>" for r in pres_data['supporting_reasons']])}
                    </ul>
                    <hr style="margin: 14px 0;">
                    <small style="color: #64748b;">
                        <b>Academic Note:</b> These rule-based signals are derived deterministically from classical technical indicators 
                        (SMA Cross, RSI Overbought/Oversold thresholds, and Bollinger Band boundary conditions) for academic demonstration. 
                        They do not constitute financial advisory recommendations.
                    </small>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════
# PAGE E: STOCK PREDICTION
# ══════════════════════════════════════════════════════════════
elif page == "Stock Prediction":
    st.markdown("### Stock Prediction")
    st.caption("Chronological ML Splits, Feature Importance, Directional Accuracy, and Naive Baseline Benchmarks")
    st.markdown("---")

    sel_p_stock = st.selectbox("Select Target Equity", options=available_tickers, index=0)

    # Extract OHLCV
    if isinstance(df.columns, pd.MultiIndex):
        t_ohlcv = df.xs(sel_p_stock, axis=1, level=1).copy() if sel_p_stock in df.columns.get_level_values(1) else df.copy()
    else:
        t_ohlcv = df.copy()

    col_cfg1, col_cfg2, col_cfg3 = st.columns([1, 1, 1])
    with col_cfg1:
        chosen_algo = st.selectbox(
            "Prediction Algorithm",
            ["Random Forest", "Linear Regression", "Gradient Boosting"],
            index=0,
        )
    with col_cfg2:
        pred_horizon = st.selectbox("Forecast Horizon", [1, 5], format_func=lambda x: f"{x} Trading Day Ahead", index=0)
    with col_cfg3:
        test_pct = st.slider("Chronological Test Split (%)", min_value=15, max_value=30, value=20, step=5) / 100.0

    with st.spinner("Training predictive machine learning model..."):
        try:
            ml_output = train_and_evaluate_model(t_ohlcv, model_name=chosen_algo, horizon=pred_horizon, test_size=test_pct)
        except Exception as err:
            st.error(f"Machine learning execution error: {err}")
            st.stop()

    # Evaluation KPI Bar
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("MAE (Mean Absolute Error)", f"${ml_output['mae']:.2f}")
    k2.metric("RMSE (Root Mean Square)", f"${ml_output['rmse']:.2f}")
    k3.metric("R² Score (Goodness of Fit)", f"{ml_output['r2_score']:.4f}")
    k4.metric("Directional Accuracy", f"{ml_output['directional_accuracy_pct']:.1f}%")

    curr_sym = "₹" if ".NS" in sel_p_stock else "$"
    st.markdown(
        f"""
        <div style="background: linear-gradient(145deg, #121827 0%, #0d131f 100%); border: 1px solid var(--border-subtle); border-left: 4px solid var(--accent-cyan); border-radius: 6px; padding: 14px 18px; margin: 12px 0;">
            <div style="color: #94a3b8; font-size: 0.74rem; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em;">{pred_horizon}-Day Out-of-Sample Price Forecast</div>
            <div style="display: flex; align-items: baseline; gap: 12px; margin: 4px 0;">
                <span style="font-size: 1.6rem; font-weight: 800; color: #f8fafc; font-family: monospace;">{curr_sym}{ml_output['next_day_forecast']:,.2f}</span>
                <span class="{'chip-green' if ml_output['predicted_change'] >= 0 else 'chip-red'}">
                    {ml_output['predicted_change_pct']:+.2f}% expected move
                </span>
            </div>
            <small style="color: #64748b;">Current Close: {curr_sym}{ml_output['current_price']:,.2f} | Expected Delta: {curr_sym}{ml_output['predicted_change']:+.2f} | Test Window: {ml_output['test_period']}</small>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Actual vs Predicted Plotly Chart
    st.markdown('<p class="term-header">Actual vs. Predicted Price (Out-of-Sample Test Window)</p>', unsafe_allow_html=True)
    viz = ml_output["viz_df"]

    fig_test = go.Figure()
    fig_test.add_trace(
        go.Scatter(
            x=viz.index,
            y=viz["Actual_Close"],
            mode="lines",
            name="Actual Close",
            line=dict(color="#38bdf8", width=2.0),
        )
    )
    fig_test.add_trace(
        go.Scatter(
            x=viz.index,
            y=viz["Predicted_Close"],
            mode="lines",
            name=f"Predicted ({chosen_algo})",
            line=dict(color="#fb923c", width=1.8, dash="dash"),
        )
    )
    fig_test.add_trace(
        go.Scatter(
            x=viz.index,
            y=viz["Naive_Baseline"],
            mode="lines",
            name="Naive Baseline (Previous-Day Price)",
            line=dict(color="#64748b", width=1.2, dash="dot"),
        )
    )
    fig_test.update_layout(**_plotly_layout(height=400))
    st.plotly_chart(fig_test, use_container_width=True)

    # Feature Importance & Error Distribution
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        st.markdown("##### Feature Importance / Coefficients")
        fi_df = pd.DataFrame(list(ml_output["feature_importance"].items()), columns=["Feature", "Weight"]).head(8)
        fig_fi = px.bar(
            fi_df,
            x="Weight",
            y="Feature",
            orientation="h",
            color="Weight",
            color_continuous_scale="Blues",
            template=PLOTLY_TEMPLATE,
        )
        fig_fi.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=270, coloraxis_showscale=False)
        fig_fi.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_fi, use_container_width=True)

    with col_f2:
        st.markdown("##### Residual Error Distribution (Actual - Predicted)")
        fig_err = px.histogram(
            viz["Error"],
            nbins=28,
            color_discrete_sequence=["#a78bfa"],
            template=PLOTLY_TEMPLATE,
        )
        fig_err.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=270, showlegend=False)
        st.plotly_chart(fig_err, use_container_width=True)

    # Multi-Model Benchmark Comparison Table
    st.markdown('<p class="term-header">Comprehensive Model Benchmark Comparison</p>', unsafe_allow_html=True)
    comp_results = compare_all_models(t_ohlcv, horizon=pred_horizon, test_size=test_pct)
    st.dataframe(comp_results, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# PAGE F: PERFORMANCE BENCHMARK
# ══════════════════════════════════════════════════════════════
elif page == "Performance Benchmark":
    st.markdown("### Performance Benchmark")
    st.caption("Empirical Runtime Profiling: In-Memory Pandas vs. Distributed Apache PySpark")
    st.markdown("---")

    st.markdown(
        """
        <div style="background: linear-gradient(145deg, #121827 0%, #0d131f 100%); border: 1px solid var(--border-subtle); border-radius: 6px; padding: 14px 18px; margin-bottom: 14px;">
            <div style="color: #38bdf8; font-size: 0.76rem; font-weight: 700; text-transform: uppercase;">Academic Benchmark Methodology</div>
            <p style="color: #cbd5e1; font-size: 0.88rem; margin: 4px 0 0 0; line-height: 1.5;">
            This module benchmarks identical operations across varying row counts scaled from real market data.
            <b>Notice:</b> On modest record sizes (< 50,000 rows), Pandas executed natively in C is typically faster than PySpark 
            due to PySpark's fixed JVM startup, Py4J serialization, and DAG task scheduling overhead. 
            Spark's advantage emerges when datasets exceed single-node RAM where Pandas crashes with out-of-memory errors.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    scale_choices = st.multiselect(
        "Select Dataset Scales to Profile (Records)",
        [10000, 50000, 100000],
        default=[10000, 50000],
    )

    if st.button("Run Live Benchmark Suite", use_container_width=True):
        if not scale_choices:
            st.warning("Select at least one record count scale.")
        else:
            with st.spinner("Executing real Pandas vs. PySpark benchmark operations..."):
                b_res = run_full_benchmark(df, record_counts=scale_choices)
                st.session_state["active_bench_df"] = b_res["benchmark_df"]
                st.session_state["active_bench_analysis"] = b_res["academic_analysis"]

    if "active_bench_df" in st.session_state:
        b_df = st.session_state["active_bench_df"]

        st.markdown('<p class="term-header">Execution Latency Comparison (Lower is Faster)</p>', unsafe_allow_html=True)
        valid_p = b_df[b_df["PySpark (s)"] != "N/A"].copy()
        if not valid_p.empty:
            valid_p["PySpark (s)"] = valid_p["PySpark (s)"].astype(float)
            valid_p["Pandas (s)"] = valid_p["Pandas (s)"].astype(float)

            fig_b = go.Figure()
            fig_b.add_trace(
                go.Bar(
                    x=valid_p["Operation"] + " (" + valid_p["Records"].astype(str) + " rows)",
                    y=valid_p["Pandas (s)"],
                    name="Pandas (In-Memory)",
                    marker_color="#38bdf8",
                )
            )
            fig_b.add_trace(
                go.Bar(
                    x=valid_p["Operation"] + " (" + valid_p["Records"].astype(str) + " rows)",
                    y=valid_p["PySpark (s)"],
                    name="Apache PySpark (Local Distributed)",
                    marker_color="#fb923c",
                )
            )
            fig_b.update_layout(**_plotly_layout(height=390), barmode="group", yaxis_title="Seconds")
            st.plotly_chart(fig_b, use_container_width=True)

        st.markdown("##### Detailed Measurement Ledger")
        st.dataframe(b_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# PAGE G: PROJECT INFORMATION (ACADEMIC ABL SPECIFICATION)
# ══════════════════════════════════════════════════════════════
elif page == "Project Information":
    st.markdown("### Project Information")
    st.caption("Activity-Based Learning (ABL) Academic Project Documentation")
    st.markdown("---")

    st.markdown(
        """
        ### StockSense: Big Data Analytics in Stock Market Prediction

        #### 1. Problem Statement
        Financial equity markets produce millions of structured and semi-structured time-series records daily. 
        Traditional in-memory processing frameworks (such as standalone single-threaded Python scripts) face severe scalability 
        bottlenecks, high execution latency, and out-of-memory (OOM) failures when performing historical feature engineering, 
        rolling volatility windowing, and risk modeling across broad multi-asset portfolios. 

        **StockSense** addresses this challenge by establishing a distributed Big Data pipeline utilizing **Apache Spark (PySpark)** 
        for parallelized transformations, combined with supervised machine learning algorithms for empirical directional prediction.

        ---

        #### 2. Project Architecture Diagram

        ```
        Stock Data Sources (Yahoo Finance / NSE / Parquet / CSV)
                               │
                               ▼
                         Data Ingestion
                               │
                               ▼
                          PySpark ETL
             (Data Hygiene, Imputation, Window Transforms)
                               │
                               ▼
                        Parquet Storage
                     (Columnar Binary Files)
                               │
                               ▼
                       Analytics Engine
          (Descriptive, Diagnostic & Prescriptive Metrics)
                               │
                               ▼
                           ML Models
         (Random Forest, Linear Regression, Gradient Boosting)
                               │
                               ▼
                      Streamlit Dashboard
                 (Interactive Financial Terminal)
        ```

        ---

        #### 3. Demonstration of the Five Vs of Big Data

        | Big Data Characteristic | Implementation in StockSense |
        | :--- | :--- |
        | **Volume** | Multi-year historical datasets spanning hundreds of thousands of OHLCV records; partitioned PySpark execution handles datasets exceeding single-node RAM. |
        | **Velocity** | Automated ingestion cadence via Yahoo Finance API with dynamic caching (`@st.cache_data`) and on-demand live market refresh. |
        | **Variety** | Multi-modal data ingestion: binary Apache Parquet files, structured tabular CSV, nested MultiIndex market arrays, and company metadata. |
        | **Veracity** | Robust data hygiene pipeline: duplicate timestamp elimination, backward/forward holiday imputation, schema type enforcement, and IQR return outlier detection. |
        | **Value** | Actionable quantitative output: probabilistic next-day price forecasts, risk tier classifications, and empirical distributed performance benchmarks. |

        ---

        #### 4. Technology Stack
        - **Distributed Big Data Engine:** Apache Spark 3.5.x via PySpark (`local[2]` master, 4 shuffle partitions)
        - **Primary Language:** Python 3.12
        - **Frontend & Visualizations:** Streamlit, Plotly Graph Objects & Express
        - **Data Manipulation:** Apache PySpark DataFrames, Pandas, NumPy, PyArrow
        - **Machine Learning & Statistics:** Scikit-Learn (Random Forest, Gradient Boosting, Linear Regression), SciPy
        - **Storage Artifacts:** Apache Parquet, CSV

        ---

        #### 5. Project Limitations & Future Scope
        - **Distributed Infrastructure:** For local academic reproducibility, PySpark operates in `local[*]` mode rather than an external multi-node Hadoop/YARN cluster.
        - **Market Microstructure:** Daily OHLCV data abstracts away tick-by-tick order-book order flow dynamics and slippage.
        - **Future Scope:** Integration of Apache Kafka for real-time streaming tick ingestion, deep learning architectures (LSTM / Temporal Fusion Transformers), and sentiment analysis from financial news feeds.
        """
    )

# ─────────────────────────────────────────────────────────────
# INSTITUTIONAL FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    """
    <div style="display: flex; justify-content: space-between; align-items: center; padding: 4px 0; color: #64748b; font-size: 0.75rem;">
        <span>StockSense Enterprise Terminal · Activity-Based Learning (ABL) Demonstration</span>
        <span>Apache PySpark 3.5 · Python 3.12 · Machine Learning</span>
    </div>
    """,
    unsafe_allow_html=True,
)
