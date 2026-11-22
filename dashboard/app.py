"""StockSense: Big Data Analytics in Stock Market Prediction.

Focused Academic Stock Market Prediction System:
- Big Data processing with Apache PySpark 3.5 & Parquet Storage
- Feature Engineering (PySpark Window Functions, Rolling Averages, Lags, Volatility)
- Chronological ML Modeling (Linear Regression, Random Forest, Gradient Boosting)
- Out-of-sample Test Benchmarking against Naive Persistence Baseline
- Executive Financial UI/UX Architecture
"""

from __future__ import annotations

import io
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Optional

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

import importlib
import src.data_fetcher
import src.spark_engine
import src.ml_predictor

importlib.reload(src.data_fetcher)
importlib.reload(src.spark_engine)
importlib.reload(src.ml_predictor)

from src.data_fetcher import (
    NSE_PRESETS,
    US_PRESETS,
    get_processed_data_dir,
    get_sample_data_dir,
    get_stock_data,
    list_sample_datasets,
    load_local_dataset,
    parse_uploaded_file,
)
from src.ml_predictor import (
    compare_all_models,
    evaluate_models_suite,
    train_and_evaluate_model,
)
from src.spark_engine import get_spark_status, process_stock_data_spark

# ─────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StockSense | Stock Market Prediction System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CUSTOM FINANCIAL TERMINAL STYLING (CSS)
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
        --pos-green: #10b981;
        --pos-green-bg: rgba(16, 185, 129, 0.12);
        --neg-red: #ef4444;
        --neg-red-bg: rgba(239, 68, 68, 0.12);
    }

    /* Global Typography & Canvas */
    html, body, [data-testid="stAppViewContainer"],
    [data-testid="stMain"], section.main {
        background-color: var(--bg-main) !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        letter-spacing: -0.01em;
    }

    /* Monospace for Numbers & Codes */
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
        background: #0b1019;
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        padding: 12px 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    .terminal-logo-box {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .terminal-badge-pulse {
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 0.74rem;
        font-weight: 600;
        background: rgba(14, 165, 233, 0.12);
        color: #38bdf8;
        border: 1px solid rgba(14, 165, 233, 0.3);
        padding: 4px 10px;
        border-radius: 20px;
        letter-spacing: 0.04em;
    }
    .pulse-dot {
        width: 7px;
        height: 7px;
        background-color: #38bdf8;
        border-radius: 50%;
        box-shadow: 0 0 8px #38bdf8;
    }

    /* KPI and Prediction Metric Cards */
    .kpi-card {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        padding: 18px 20px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        transition: border-color 0.2s ease, transform 0.2s ease;
    }
    .kpi-card:hover {
        border-color: var(--border-strong);
        transform: translateY(-2px);
    }
    .kpi-label {
        font-size: 0.76rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--text-secondary);
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 1.65rem;
        font-weight: 700;
        color: #f8fafc;
        font-family: 'JetBrains Mono', monospace;
    }
    .kpi-sub {
        font-size: 0.75rem;
        color: var(--text-muted);
        margin-top: 4px;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    /* Dominant Prediction Card */
    .pred-card-container {
        background: linear-gradient(145deg, #101625 0%, #0d121c 100%);
        border: 1px solid #2b3954;
        border-radius: 10px;
        padding: 24px 28px;
        margin-bottom: 22px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }

    /* Badges & Chips */
    .chip-green {
        display: inline-block;
        background: var(--pos-green-bg);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: var(--pos-green) !important;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }
    .chip-red {
        display: inline-block;
        background: var(--neg-red-bg);
        border: 1px solid rgba(239, 68, 68, 0.3);
        color: var(--neg-red) !important;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }
    .chip-blue {
        display: inline-block;
        background: rgba(14, 165, 233, 0.12);
        border: 1px solid rgba(14, 165, 233, 0.3);
        color: #38bdf8 !important;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }
    .chip-gold {
        display: inline-block;
        background: rgba(245, 158, 11, 0.12);
        border: 1px solid rgba(245, 158, 11, 0.3);
        color: #fbbf24 !important;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Section Headers */
    .term-header {
        font-size: 0.85rem;
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
        min-width: 100px;
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

    /* Streamlit Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        padding: 9px 18px !important;
        box-shadow: 0 2px 10px rgba(2, 132, 199, 0.3) !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        box-shadow: 0 4px 18px rgba(2, 132, 199, 0.5) !important;
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
# PLOTLY UNIFIED THEME
# ─────────────────────────────────────────────────────────────
GRID_COLOR = "#182234"

def _plotly_layout(title: str = "", height: int = 400, **extra) -> dict:
    """Returns a unified, high-contrast Plotly layout matching the dark terminal theme."""
    return dict(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title=dict(text=title, font=dict(family="Inter, sans-serif", size=13, color="#e2e8f0")),
        font=dict(family="Inter, sans-serif", color="#94a3b8", size=11),
        height=height,
        margin=dict(l=45, r=20, t=45, b=35),
        legend=dict(
            bgcolor="rgba(13, 18, 29, 0.7)",
            bordercolor="#1e293b",
            borderwidth=1,
            font=dict(color="#cbd5e1", size=11),
        ),
        xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor="#223147", showline=False),
        yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor="#223147", showline=False),
        **extra,
    )

# ─────────────────────────────────────────────────────────────
# CACHED DATA INGESTION HELPERS
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _cached_api_load(ticker: str, period: str = "1y") -> pd.DataFrame:
    return get_stock_data(tickers=[ticker], period=period)

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
                <span style="font-size: 0.72rem; color: #64748b; margin-left: 8px; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600;">Big Data Stock Market Prediction System</span>
            </div>
        </div>
        <div style="display: flex; align-items: center; gap: 10px;">
            <div class="terminal-badge-pulse">
                <div class="pulse-dot"></div>
                Apache PySpark 3.5 Active
            </div>
            <div style="background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.3); color: #34d399; padding: 4px 10px; border-radius: 20px; font-size: 0.74rem; font-weight: 600; letter-spacing: 0.04em;">
                US (NASDAQ/NYSE) & Indian (NSE) Equities
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────
# SIDEBAR CONTROLS: STOCK SELECTION & INGESTION
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 4px 0 10px 0;">
            <div style="font-weight: 700; font-size: 1.05rem; color: #f8fafc;">System Navigation</div>
            <div style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em;">Big Data Prediction Architecture</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 4 Main Pages Only
    page = st.radio(
        "Navigation Menu",
        [
            "Prediction Dashboard",
            "Data Processing",
            "Model Evaluation",
            "Project Information",
        ],
        index=0,
        label_visibility="collapsed",
    )
    st.markdown("---")

    st.markdown('<p class="term-header">Stock Selection</p>', unsafe_allow_html=True)
    CORE_STOCK_PRESETS = ["NVDA", "AAPL", "MSFT", "RELIANCE.NS", "TCS.NS"]
    
    selected_stock = st.selectbox(
        "Target Equity",
        CORE_STOCK_PRESETS,
        index=0,
        help="Select stock for Big Data processing and ML prediction",
    )

    st.markdown('<p class="term-header">Data Source Mode</p>', unsafe_allow_html=True)
    data_mode = st.radio(
        "Ingestion Source",
        ["Live (Yahoo Finance)", "Offline Sample Dataset", "Upload File (CSV/Parquet)"],
        index=0,
        label_visibility="collapsed",
    )

    df_raw: pd.DataFrame = pd.DataFrame()
    data_source_badge = ""
    data_source_type = ""

    # Load Data Based on Ingestion Mode
    if data_mode == "Live (Yahoo Finance)":
        with st.spinner(f"Ingesting live market feed for {selected_stock}..."):
            try:
                df_raw = _cached_api_load(selected_stock, period="1y")
                data_source_badge = "Live Market Data (Yahoo Finance API)"
                data_source_type = "LIVE"
            except Exception as exc:
                st.warning(f"Live API unavailable ({exc}). Falling back to cached offline sample.")
                # Fallback to local sample dataset
                normalized_name = selected_stock.replace(".", "_")
                sample_p = get_sample_data_dir() / f"{normalized_name}.parquet"
                if not sample_p.exists():
                    sample_p = get_sample_data_dir() / f"{selected_stock}.parquet"
                if sample_p.exists():
                    df_raw = load_local_dataset(sample_p)
                    data_source_badge = "Offline Sample Dataset (Auto-Fallback)"
                    data_source_type = "SAMPLE"
                else:
                    st.error("No offline sample found. Please check data/sample/ directory.")
                    st.stop()

    elif data_mode == "Offline Sample Dataset":
        sample_files = list_sample_datasets()
        matching_sample = None
        for s in sample_files:
            if s["ticker"] == selected_stock or s["ticker"] == selected_stock.replace(".", "_"):
                matching_sample = s
                break
        if matching_sample is None and sample_files:
            matching_sample = sample_files[0]

        if matching_sample:
            try:
                df_raw = load_local_dataset(matching_sample["path"])
                data_source_badge = f"Offline Sample Dataset ({matching_sample['filename']})"
                data_source_type = "SAMPLE"
            except Exception as exc:
                st.error(f"Error loading offline file: {exc}")
                st.stop()
        else:
            st.error("No sample datasets discovered in data/sample/.")
            st.stop()

    elif data_mode == "Upload File (CSV/Parquet)":
        uploaded = st.file_uploader("Upload Historical Stock File", type=["csv", "parquet"])
        if uploaded is not None:
            try:
                df_raw, up_ticker = parse_uploaded_file(uploaded)
                selected_stock = up_ticker
                data_source_badge = f"Uploaded Dataset ({uploaded.name})"
                data_source_type = "UPLOAD"
            except Exception as exc:
                st.error(f"Error parsing uploaded file: {exc}")
                st.stop()
        else:
            st.info("Upload a historical stock CSV or Parquet file to proceed.")
            st.stop()

    st.markdown("---")
    st.markdown('<p class="term-header">Cluster Telemetry</p>', unsafe_allow_html=True)
    spark_info = get_spark_status()
    if spark_info["pyspark_installed"]:
        st.markdown('<span class="chip-blue">Engine: Apache Spark 3.5</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="chip-gold">Engine: Pandas Fallback</span>', unsafe_allow_html=True)

    if data_source_type == "LIVE":
        st.markdown('<span class="chip-green">Feed: Real-Time Active</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="chip-gold">Feed: Offline / Cached</span>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# DATA SANITY & ETL EXECUTION OR RETRIEVAL
# ─────────────────────────────────────────────────────────────
if df_raw.empty:
    st.warning("No data available for the selected stock. Please select another stock or source.")
    st.stop()

# Flatten and extract single-ticker close series
close_col = "Adj Close" if "Adj Close" in df_raw.columns else ("Close" if "Close" in df_raw.columns else df_raw.columns[0])
if isinstance(df_raw.columns, pd.MultiIndex):
    if selected_stock in df_raw.columns.get_level_values(1):
        df_flat = df_raw.xs(selected_stock, axis=1, level=1).copy()
    else:
        df_flat = df_raw.iloc[:, :6].copy()
else:
    df_flat = df_raw.copy()

# Currency resolution
currency_sym = "₹" if selected_stock.endswith(".NS") else "$"

# Cache processed dataset in session state so ETL and Prediction share the exact same dataset
cache_key = f"spark_processed_{selected_stock}"
proc_dir = str(get_processed_data_dir())

if cache_key not in st.session_state:
    with st.spinner(f"Running Spark distributed ETL pipeline for {selected_stock}..."):
        st.session_state[cache_key] = process_stock_data_spark(
            df_flat,
            ticker=selected_stock,
            output_dir=proc_dir,
        )

spark_results = st.session_state[cache_key]
processed_df = spark_results.get("processed_df", df_flat)


# ══════════════════════════════════════════════════════════════
# PAGE 1: PREDICTION DASHBOARD (PRIMARY LANDING PAGE)
# ══════════════════════════════════════════════════════════════
if page == "Prediction Dashboard":
    st.markdown(f"### Stock Prediction Dashboard · {selected_stock}")
    st.caption(f"Big Data Feature Pipeline & Chronological Machine Learning Forecast · {data_source_badge}")
    st.markdown("---")

    # Latest Available Closing Price
    latest_close = float(df_flat[close_col].iloc[-1])
    latest_date = df_flat.index[-1].strftime("%b %d, %Y") if hasattr(df_flat.index[-1], "strftime") else str(df_flat.index[-1])
    prev_close = float(df_flat[close_col].iloc[-2]) if len(df_flat) > 1 else latest_close
    day_change = latest_close - prev_close
    day_pct = (day_change / prev_close) * 100.0 if prev_close != 0 else 0.0

    # Top KPI Row: Stock, Latest Price, Data Source
    kpi_col1, kpi_col2, kpi_col3 = st.columns([1.2, 1.2, 1.6])
    with kpi_col1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Active Stock Symbol</div>
                <div class="kpi-value">{selected_stock}</div>
                <div class="kpi-sub">Market: {'National Stock Exchange of India' if selected_stock.endswith('.NS') else 'US Equities (NASDAQ/NYSE)'}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi_col2:
        change_class = "chip-green" if day_change >= 0 else "chip-red"
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Latest Closing Price ({latest_date})</div>
                <div class="kpi-value">{currency_sym}{latest_close:,.2f}</div>
                <div class="kpi-sub"><span class="{change_class}">{day_change:+.2f} ({day_pct:+.2f}%)</span> vs prev day</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi_col3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Data Ingestion Origin</div>
                <div class="kpi-value" style="font-size: 1.25rem;">{data_source_badge.split('(')[0].strip()}</div>
                <div class="kpi-sub"><span class="chip-blue">{len(df_flat)} Records Available</span> · Cleaned via PySpark</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Historical Price Chart
    st.markdown('<p class="term-header">Historical Stock Price Action & Moving Averages</p>', unsafe_allow_html=True)
    
    fig_hist = go.Figure()
    fig_hist.add_trace(
        go.Scatter(
            x=df_flat.index,
            y=df_flat[close_col],
            name="Close Price",
            mode="lines",
            line=dict(color="#38bdf8", width=2),
            fill="tozeroy",
            fillcolor="rgba(14, 165, 233, 0.08)",
        )
    )
    # Add 20-day SMA if in processed_df
    if "SMA_20" in processed_df.columns:
        fig_hist.add_trace(
            go.Scatter(
                x=df_flat.index,
                y=processed_df["SMA_20"].values,
                name="20-Day SMA (PySpark)",
                mode="lines",
                line=dict(color="#f59e0b", width=1.5, dash="dot"),
            )
        )
    if "SMA_50" in processed_df.columns:
        fig_hist.add_trace(
            go.Scatter(
                x=df_flat.index,
                y=processed_df["SMA_50"].values,
                name="50-Day SMA (PySpark)",
                mode="lines",
                line=dict(color="#a78bfa", width=1.5, dash="dash"),
            )
        )
    fig_hist.update_layout(
        **_plotly_layout(f"{selected_stock} Historical Closing Price & Moving Average Trends", height=380),
        xaxis_title="Date",
        yaxis_title=f"Price ({currency_sym})",
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # Prediction Controls & Execution
    st.markdown('<p class="term-header">Prediction Configuration</p>', unsafe_allow_html=True)
    
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.5, 1.5, 1.0])
    with ctrl_col1:
        horizon_days = st.select_slider(
            "Prediction Horizon (Trading Days)",
            options=[1, 5, 10, 15, 30],
            value=1,
            format_func=lambda x: f"{x} Day{'s' if x > 1 else ''} Forward (T+{x})",
        )
    with ctrl_col2:
        model_choice = st.selectbox(
            "Machine Learning Model",
            ["Linear Regression", "Random Forest", "Gradient Boosting"],
            index=1,
        )
    with ctrl_col3:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        predict_btn = st.button("Generate Prediction", use_container_width=True)

    # Train and evaluate selected prediction model
    with st.spinner(f"Training {model_choice} on chronological features..."):
        try:
            pred_res = train_and_evaluate_model(
                ohlcv_df=processed_df,
                model_name=model_choice,
                horizon=horizon_days,
                test_size=0.20,
            )
        except Exception as exc:
            st.error(f"Prediction modeling error: {exc}")
            st.stop()

    forecast_price = pred_res["next_day_forecast"]
    delta_val = pred_res["predicted_change"]
    delta_pct = pred_res["predicted_change_pct"]
    is_positive = delta_val >= 0

    # DOMINANT PREDICTION CARD
    move_color = "#10b981" if is_positive else "#ef4444"
    move_symbol = "+" if is_positive else ""
    move_badge = "chip-green" if is_positive else "chip-red"

    st.markdown(
        f"""
        <div class="pred-card-container">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 16px;">
                <div>
                    <div style="font-size: 0.8rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px;">
                        Model Forecast · {pred_res['engine_name']}
                    </div>
                    <div style="font-size: 2.8rem; font-weight: 800; color: #ffffff; font-family: 'JetBrains Mono', monospace; line-height: 1.1;">
                        {currency_sym}{forecast_price:,.2f}
                    </div>
                    <div style="margin-top: 10px; display: flex; align-items: center; gap: 10px;">
                        <span class="{move_badge}" style="font-size: 1rem; padding: 4px 12px;">
                            Expected Move: {move_symbol}{currency_sym}{abs(delta_val):.2f} ({move_symbol}{delta_pct:.2f}%)
                        </span>
                        <span style="font-size: 0.8rem; color: #94a3b8;">
                            Horizon: <b>T+{horizon_days} Trading Day{'s' if horizon_days > 1 else ''}</b>
                        </span>
                    </div>
                </div>
                <div style="background: rgba(15, 23, 42, 0.8); border: 1px solid #1e293b; border-radius: 8px; padding: 14px 20px; min-width: 260px;">
                    <div style="font-size: 0.74rem; font-weight: 700; color: #38bdf8; text-transform: uppercase; margin-bottom: 8px;">
                        Out-of-Sample Test Reliability
                    </div>
                    <div style="font-size: 0.85rem; color: #cbd5e1; display: flex; justify-content: space-between; margin-bottom: 4px;">
                        <span>Test MAE:</span> <b class="mono-num">{currency_sym}{pred_res['mae']:.2f}</b>
                    </div>
                    <div style="font-size: 0.85rem; color: #cbd5e1; display: flex; justify-content: space-between; margin-bottom: 4px;">
                        <span>Test RMSE:</span> <b class="mono-num">{currency_sym}{pred_res['rmse']:.2f}</b>
                    </div>
                    <div style="font-size: 0.85rem; color: #cbd5e1; display: flex; justify-content: space-between; margin-bottom: 4px;">
                        <span>Directional Accuracy:</span> <b class="mono-num" style="color: {'#10b981' if pred_res['directional_accuracy_pct'] >= 50 else '#ef4444'};">{pred_res['directional_accuracy_pct']:.1f}%</b>
                    </div>
                    <div style="font-size: 0.85rem; color: #cbd5e1; display: flex; justify-content: space-between;">
                        <span>Naive Baseline MAE:</span> <b class="mono-num">{currency_sym}{pred_res['baseline_mae']:.2f}</b>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Actual versus Predicted Historical Chart
    st.markdown('<p class="term-header">Actual vs. Predicted Historical Performance (Out-of-Sample Test Window)</p>', unsafe_allow_html=True)
    
    viz_df = pred_res["viz_df"]
    fig_pred = go.Figure()
    fig_pred.add_trace(
        go.Scatter(
            x=viz_df.index,
            y=viz_df["Actual_Close"],
            name="Actual Close Price",
            mode="lines+markers",
            line=dict(color="#38bdf8", width=2.5),
            marker=dict(size=4),
        )
    )
    fig_pred.add_trace(
        go.Scatter(
            x=viz_df.index,
            y=viz_df["Predicted_Close"],
            name=f"Predicted ({pred_res['engine_name']})",
            mode="lines",
            line=dict(color="#f59e0b", width=2, dash="dash"),
        )
    )
    fig_pred.add_trace(
        go.Scatter(
            x=viz_df.index,
            y=viz_df["Naive_Baseline"],
            name="Naive Baseline (Close[t-1])",
            mode="lines",
            line=dict(color="#64748b", width=1.5, dash="dot"),
        )
    )
    fig_pred.update_layout(
        **_plotly_layout(
            f"{selected_stock}: Ground Truth vs. {pred_res['engine_name']} Forecasts ({pred_res['test_period']})",
            height=400,
        ),
        xaxis_title="Date",
        yaxis_title=f"Price ({currency_sym})",
    )
    st.plotly_chart(fig_pred, use_container_width=True)

    # Prediction Summary Box
    st.markdown('<p class="term-header">Prediction Summary & Feature Contribution</p>', unsafe_allow_html=True)
    
    sum_col1, sum_col2 = st.columns([1.6, 1.4])
    with sum_col1:
        st.markdown(
            f"""
            <div style="background: #0d121d; border: 1px solid var(--border-subtle); border-radius: 8px; padding: 18px 22px;">
                <div style="font-weight: 700; font-size: 0.95rem; color: #f8fafc; margin-bottom: 8px;">
                    Model Interpretation & Educational Disclosure
                </div>
                <p style="font-size: 0.84rem; color: #94a3b8; line-height: 1.6; margin-bottom: 8px;">
                    The <b>{pred_res['engine_name']}</b> forecasts an estimated future price of 
                    <b style="color: #ffffff;">{currency_sym}{forecast_price:,.2f}</b> over a horizon of <b>{horizon_days} trading day{'s' if horizon_days > 1 else ''}</b>, 
                    representing an expected movement of <b style="color: {move_color};">{move_symbol}{delta_pct:.2f}%</b> from the latest close of {currency_sym}{latest_close:,.2f}.
                </p>
                <p style="font-size: 0.84rem; color: #94a3b8; line-height: 1.6; margin-bottom: 8px;">
                    • <b>Data Split:</b> Evaluated on {pred_res['test_samples']} out-of-sample trading days ({pred_res['test_period']}) strictly partitioned chronologically without lookahead data leakage.<br>
                    • <b>Error Tolerance:</b> Test Mean Absolute Error is ±{currency_sym}{pred_res['mae']:.2f} ({pred_res['mae']/latest_close*100:.2f}% of current price).<br>
                    • <b>Benchmark Context:</b> Naive Previous-Day baseline achieved an MAE of ±{currency_sym}{pred_res['baseline_mae']:.2f}.
                </p>
                <div style="font-size: 0.74rem; color: #64748b; font-style: italic;">
                    Academic Disclosure: This prediction is generated solely for academic demonstration in Big Data analytics. Financial markets exhibit non-stationary stochastic behavior; predictions do not constitute investment advice.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with sum_col2:
        top_features = list(pred_res["feature_importance"].items())[:6]
        if top_features:
            feat_df = pd.DataFrame(top_features, columns=["Feature", "Contribution / Weight"])
            fig_feat = px.bar(
                feat_df,
                x="Contribution / Weight",
                y="Feature",
                orientation="h",
                color="Contribution / Weight",
                color_continuous_scale="Blues",
            )
            fig_feat.update_layout(
                **_plotly_layout("Top Contributing Engineered Features", height=240),
                coloraxis_showscale=False,
                margin=dict(l=70, r=20, t=35, b=25),
            )
            st.plotly_chart(fig_feat, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE 2: DATA PROCESSING (BIG DATA PYSPARK PIPELINE)
# ══════════════════════════════════════════════════════════════
elif page == "Data Processing":
    st.markdown("### Big Data Processing Pipeline")
    st.caption(f"Distributed Ingestion, Validation, Feature Engineering & Parquet Serialization · {selected_stock}")
    st.markdown("---")

    # Workflow Stepper Diagram
    st.markdown(
        """
        <div class="workflow-container">
            <div class="step-item">
                <div class="step-badge active">1</div>
                <div class="step-title">Ingestion</div>
                <div class="step-desc">Historical OHLCV</div>
            </div>
            <div style="color: #334155; font-size: 1.2rem;">→</div>
            <div class="step-item">
                <div class="step-badge active">2</div>
                <div class="step-title">Validation</div>
                <div class="step-desc">Schema & Nulls</div>
            </div>
            <div style="color: #334155; font-size: 1.2rem;">→</div>
            <div class="step-item">
                <div class="step-badge active">3</div>
                <div class="step-title">IQR Outliers</div>
                <div class="step-desc">Treatment Flag</div>
            </div>
            <div style="color: #334155; font-size: 1.2rem;">→</div>
            <div class="step-item">
                <div class="step-badge active">4</div>
                <div class="step-title">Spark Window</div>
                <div class="step-desc">Lags & Rolling SMA</div>
            </div>
            <div style="color: #334155; font-size: 1.2rem;">→</div>
            <div class="step-item">
                <div class="step-badge active">5</div>
                <div class="step-title">Parquet Storage</div>
                <div class="step-desc">Snappy Compression</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Re-run button for presentation demonstration
    run_col1, run_col2 = st.columns([2.5, 1.5])
    with run_col1:
        st.write("Click below to trigger a live distributed execution through Apache Spark DataFrame APIs:")
    with run_col2:
        if st.button("⚡ Execute PySpark Pipeline", use_container_width=True):
            with st.spinner("Executing PySpark distributed transformations..."):
                st.session_state[cache_key] = process_stock_data_spark(
                    df_flat,
                    ticker=selected_stock,
                    output_dir=proc_dir,
                )
            st.success("PySpark execution completed successfully!")

    spark_res = st.session_state[cache_key]

    # Processing Metrics KPI Cards
    st.markdown('<p class="term-header">Execution Metrics & Cluster Telemetry</p>', unsafe_allow_html=True)
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Input / Output Records</div>
                <div class="kpi-value">{spark_res['raw_records']} / {spark_res['processed_records']}</div>
                <div class="kpi-sub">Cleaned records: <b>{spark_res['processed_records']}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m_col2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Duplicates & Missing Handled</div>
                <div class="kpi-value">{spark_res['duplicates_removed']} / {spark_res['missing_values_handled']}</div>
                <div class="kpi-sub">Forward/Backward Imputation</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m_col3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Outliers Flagged (IQR)</div>
                <div class="kpi-value">{spark_res['outliers_detected']}</div>
                <div class="kpi-sub">Flagged in <code>Outlier_Flag</code></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m_col4:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Execution Duration</div>
                <div class="kpi-value">{spark_res['duration_seconds']}s</div>
                <div class="kpi-sub">Partitions: <b>{spark_res['partitions']}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Engine Transparency Badge
    st.markdown("<br>", unsafe_allow_html=True)
    eng_name = spark_res["engine"]
    eng_chip = "chip-blue" if "PySpark" in eng_name else "chip-gold"
    st.markdown(
        f"""
        <div style="background: #0f1523; border: 1px solid var(--border-subtle); border-radius: 8px; padding: 14px 20px; display: flex; align-items: center; justify-content: space-between;">
            <div>
                <span style="font-size: 0.8rem; color: #94a3b8; font-weight: 600; text-transform: uppercase;">Processing Engine:</span>
                <span class="{eng_chip}" style="margin-left: 10px;">{eng_name}</span>
            </div>
            <div style="font-size: 0.78rem; color: #94a3b8;">
                Storage Sink: <code style="color: #38bdf8;">{spark_res['output_parquet']}</code>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Outlier Treatment Explanation Box
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="background: rgba(14, 165, 233, 0.06); border: 1px solid rgba(14, 165, 233, 0.2); border-radius: 8px; padding: 14px 18px;">
            <div style="font-size: 0.82rem; font-weight: 700; color: #38bdf8; margin-bottom: 4px;">
                Outlier Treatment & Academic Integrity Protocol
            </div>
            <div style="font-size: 0.8rem; color: #94a3b8; line-height: 1.5;">
                {spark_res.get('outlier_treatment', 'Outlier detection via Interquartile Range (IQR). Genuine extreme market events are preserved.')}
                To prevent distortion in downstream machine learning estimators while honoring financial reality, extreme price moves are identified via the IQR threshold <code>[Q1 - 1.5*IQR, Q3 + 1.5*IQR]</code> and annotated in the <code>Outlier_Flag</code> column without altering historical prices.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Generated Features Specification
    st.markdown('<p class="term-header">Generated Features in PySpark Processing</p>', unsafe_allow_html=True)
    
    features_list = spark_res.get("generated_features", [])
    feat_chips = " ".join([f'<span class="chip-blue" style="margin: 3px 4px;">{f}</span>' for f in features_list])
    st.markdown(
        f"""
        <div style="background: #0d121d; border: 1px solid var(--border-subtle); border-radius: 8px; padding: 16px 20px; margin-bottom: 20px;">
            <div style="font-size: 0.82rem; font-weight: 600; color: #f8fafc; margin-bottom: 8px;">
                Engineered Features Produced for Machine Learning ({len(features_list)} Features):
            </div>
            <div>{feat_chips}</div>
            <div style="font-size: 0.75rem; color: #64748b; margin-top: 10px;">
                Calculated using distributed PySpark Window functions over chronological partitions: <code>Window.partitionBy(F.lit("GLOBAL")).orderBy("Date")</code>.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Before and After Data Previews
    st.markdown('<p class="term-header">Before & After Data Previews</p>', unsafe_allow_html=True)
    prev_tab1, prev_tab2 = st.tabs(["Processed Dataset Preview (Features Engineered)", "Raw Input Ingestion Sample"])
    
    with prev_tab1:
        st.write("Cleaned and transformed dataset loaded from Parquet sink:")
        if "preview_df" in spark_res and isinstance(spark_res["preview_df"], pd.DataFrame):
            st.dataframe(spark_res["preview_df"], use_container_width=True)
        else:
            st.dataframe(processed_df.tail(10), use_container_width=True)

    with prev_tab2:
        st.write("Original raw OHLCV dataset as ingested:")
        st.dataframe(df_flat.head(10), use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE 3: MODEL EVALUATION (ACADEMIC BENCHMARKING)
# ══════════════════════════════════════════════════════════════
elif page == "Model Evaluation":
    st.markdown(f"### Machine Learning Model Evaluation · {selected_stock}")
    st.caption("Rigorous Out-of-Sample Chronological Validation & Baseline Comparison")
    st.markdown("---")

    # Horizon Selector for Comparative Evaluation
    eval_col1, eval_col2 = st.columns([2.0, 2.0])
    with eval_col1:
        eval_horizon = st.select_slider(
            "Evaluation Horizon",
            options=[1, 5, 10, 15, 30],
            value=1,
            format_func=lambda x: f"{x} Day{'s' if x > 1 else ''} Horizon (T+{x})",
        )
    with eval_col2:
        st.info("Validation Method: Chronological 80% Train / 20% Test Split. Zero lookahead leakage.")

    # Run Multi-Model Evaluation Suite
    with st.spinner(f"Training and benchmarking all ML models for {selected_stock}..."):
        try:
            suite = evaluate_models_suite(
                ohlcv_df=processed_df,
                horizon=eval_horizon,
                test_size=0.20,
            )
        except Exception as exc:
            st.error(f"Error evaluating model suite: {exc}")
            st.stop()

    comp_df = suite["comparison_df"]
    best_name = suite["best_model_name"]
    best_res = suite["best_model_res"]
    beats_baseline = suite["beats_baseline"]
    imp_pct = suite["improvement_pct"]

    # BEST PERFORMING MODEL SHOWCASE CARD
    best_chip = "chip-green" if beats_baseline else "chip-gold"
    status_text = f"Outperforms Naive Baseline by {imp_pct}% MAE reduction" if beats_baseline else "Competes with Random Walk Naive Baseline"

    st.markdown(
        f"""
        <div style="background: linear-gradient(145deg, #101726 0%, #0d121c 100%); border: 1px solid #23344d; border-radius: 10px; padding: 22px 26px; margin-bottom: 22px;">
            <div style="font-size: 0.76rem; font-weight: 700; color: #38bdf8; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px;">
                Top Out-of-Sample Test Performer (Chronological Validation)
            </div>
            <div style="font-size: 1.85rem; font-weight: 800; color: #ffffff;">
                {best_res['engine_name']}
            </div>
            <div style="margin-top: 8px; display: flex; align-items: center; gap: 12px; flex-wrap: wrap;">
                <span class="{best_chip}">{status_text}</span>
                <span style="font-size: 0.82rem; color: #94a3b8;">
                    Test MAE: <b class="mono-num" style="color: #ffffff;">{currency_sym}{best_res['mae']:.2f}</b> · 
                    RMSE: <b class="mono-num" style="color: #ffffff;">{currency_sym}{best_res['rmse']:.2f}</b> · 
                    Directional Accuracy: <b class="mono-num" style="color: #ffffff;">{best_res['directional_accuracy_pct']:.1f}%</b>
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Comparative Model Scorecard Table
    st.markdown('<p class="term-header">Comprehensive Model Comparison Table</p>', unsafe_allow_html=True)
    st.dataframe(comp_df, use_container_width=True)

    # Actual vs Predicted Multi-Model Overlay Chart
    st.markdown('<p class="term-header">Actual vs. Multi-Model Predictions (Test Window)</p>', unsafe_allow_html=True)
    
    comb_viz = suite["combined_viz_df"]
    fig_comp = go.Figure()
    
    # Ground truth
    fig_comp.add_trace(
        go.Scatter(
            x=comb_viz.index,
            y=comb_viz["Actual_Close"],
            name="Actual Close",
            mode="lines+markers",
            line=dict(color="#38bdf8", width=2.5),
            marker=dict(size=4),
        )
    )
    # Naive Baseline
    fig_comp.add_trace(
        go.Scatter(
            x=comb_viz.index,
            y=comb_viz["Naive_Baseline"],
            name="Naive Baseline (Close[t-1])",
            mode="lines",
            line=dict(color="#64748b", width=1.5, dash="dot"),
        )
    )
    # Models
    colors = {"Linear Regression": "#f59e0b", "Random Forest": "#10b981", "Gradient Boosting": "#a78bfa"}
    for m in ["Linear Regression", "Random Forest", "Gradient Boosting"]:
        col = f"Pred_{m}"
        if col in comb_viz.columns:
            fig_comp.add_trace(
                go.Scatter(
                    x=comb_viz.index,
                    y=comb_viz[col],
                    name=m,
                    mode="lines",
                    line=dict(color=colors.get(m, "#cbd5e1"), width=1.8, dash="dash"),
                )
            )

    fig_comp.update_layout(
        **_plotly_layout(
            f"{selected_stock}: Ground Truth vs. All Models over Out-of-Sample Test Window",
            height=420,
        ),
        xaxis_title="Date",
        yaxis_title=f"Price ({currency_sym})",
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    # Academic Evaluation Insights
    st.markdown('<p class="term-header">Academic Methodology & Metric Definitions</p>', unsafe_allow_html=True)
    e_col1, e_col2 = st.columns(2)
    with e_col1:
        st.markdown(
            f"""
            <div style="background: #0d121d; border: 1px solid var(--border-subtle); border-radius: 8px; padding: 18px 20px;">
                <div style="font-weight: 700; font-size: 0.9rem; color: #f8fafc; margin-bottom: 8px;">
                    Chronological Train/Test Partitioning
                </div>
                <p style="font-size: 0.82rem; color: #94a3b8; line-height: 1.5; margin: 0;">
                    Unlike standard machine learning datasets where random $k$-fold cross-validation is permitted, time-series forecasting requires strict chronological splitting. Shuffling rows causes <b>lookahead data leakage</b>, where past forecasts use future price information. StockSense uses the initial 80% chronological observations for training and strictly reserves the trailing 20% for testing.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with e_col2:
        st.markdown(
            f"""
            <div style="background: #0d121d; border: 1px solid var(--border-subtle); border-radius: 8px; padding: 18px 20px;">
                <div style="font-weight: 700; font-size: 0.9rem; color: #f8fafc; margin-bottom: 8px;">
                    Role of the Persistence Baseline
                </div>
                <p style="font-size: 0.82rem; color: #94a3b8; line-height: 1.5; margin: 0;">
                    In financial econometrics (Fama's Efficient Market Hypothesis), daily stock prices closely resemble a martingale process ($Price_t \\approx Price_{t-1} + \\epsilon$). Comparing machine learning models against a <b>Naive Persistence Baseline</b> ensures that models are evaluated for genuine statistical edge rather than merely tracking autocorrelation.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════
# PAGE 4: PROJECT INFORMATION (ACADEMIC ABL DOSSIER)
# ══════════════════════════════════════════════════════════════
elif page == "Project Information":
    st.markdown("### Project Information: Academic ABL Dossier")
    st.caption("StockSense – Big Data Analytics in Stock Market Prediction")
    st.markdown("---")

    # Problem Statement & Solution
    p_col1, p_col2 = st.columns(2)
    with p_col1:
        st.markdown(
            """
            <div class="kpi-card" style="height: 100%;">
                <div class="kpi-label">Problem Statement</div>
                <p style="font-size: 0.84rem; color: #cbd5e1; line-height: 1.6; margin-top: 6px;">
                    Financial markets generate massive volumes of continuous high-frequency tick data. Conventional single-threaded in-memory tools like Pandas fail to scale due to memory limits (OOM errors) and lack distributed windowing capabilities. Moreover, financial time-series exhibit extreme non-stationarity, noise, and autocorrelation, requiring strict data preparation, leakage prevention, and robust machine learning modeling.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with p_col2:
        st.markdown(
            """
            <div class="kpi-card" style="height: 100%;">
                <div class="kpi-label">Proposed Solution</div>
                <p style="font-size: 0.84rem; color: #cbd5e1; line-height: 1.6; margin-top: 6px;">
                    StockSense implements an end-to-end distributed Big Data architecture using <b>Apache PySpark 3.5</b> and <b>Apache Parquet</b> columnar storage. It executes distributed schema validation, IQR outlier detection, and Window-based feature engineering (lagged prices, rolling moving averages, volatility). It connects seamlessly to chronologically validated machine learning models evaluated against persistence benchmarks.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 5Vs of Big Data
    st.markdown('<p class="term-header">5Vs of Big Data in Stock Market Prediction</p>', unsafe_allow_html=True)
    v1, v2, v3, v4, v5 = st.columns(5)
    with v1:
        st.markdown(
            """
            <div class="kpi-card">
                <div style="font-size: 0.78rem; font-weight: 700; color: #38bdf8;">VOLUME</div>
                <div style="font-size: 0.76rem; color: #94a3b8; margin-top: 4px;">Millions of market transactions and trade order books.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with v2:
        st.markdown(
            """
            <div class="kpi-card">
                <div style="font-size: 0.78rem; font-weight: 700; color: #38bdf8;">VELOCITY</div>
                <div style="font-size: 0.76rem; color: #94a3b8; margin-top: 4px;">Millisecond price updates and streaming ticker ticks.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with v3:
        st.markdown(
            """
            <div class="kpi-card">
                <div style="font-size: 0.78rem; font-weight: 700; color: #38bdf8;">VARIETY</div>
                <div style="font-size: 0.76rem; color: #94a3b8; margin-top: 4px;">Structured OHLCV, trade volume, and technical indicators.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with v4:
        st.markdown(
            """
            <div class="kpi-card">
                <div style="font-size: 0.78rem; font-weight: 700; color: #38bdf8;">VERACITY</div>
                <div style="font-size: 0.76rem; color: #94a3b8; margin-top: 4px;">Handling missing values, market anomalies, and stock splits.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with v5:
        st.markdown(
            """
            <div class="kpi-card">
                <div style="font-size: 0.78rem; font-weight: 700; color: #38bdf8;">VALUE</div>
                <div style="font-size: 0.76rem; color: #94a3b8; margin-top: 4px;">Actionable directional price forecasts and risk mitigation.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # System Architecture Diagram
    st.markdown('<p class="term-header">End-to-End System Architecture</p>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="background: #0f1523; border: 1px solid var(--border-subtle); border-radius: 8px; padding: 20px; font-family: monospace; font-size: 0.85rem; color: #cbd5e1; line-height: 1.7;">
            [Stock Data Sources: Yahoo Finance API / Offline Parquet]
                                      ↓
            [Data Ingestion Manager: data_fetcher.py]
                                      ↓
            [Apache PySpark 3.5 Distributed ETL: spark_engine.py]
              ├─ Schema Validation & Missing Value Imputation
              ├─ IQR Outlier Detection (Flagged, Original Preserved)
              └─ PySpark Window Aggregations (Lags 1-5, SMA 5-50, Volatility)
                                      ↓
            [Columnar Storage: Snappy Compressed Parquet (data/processed/)]
                                      ↓
            [Machine Learning Engine: ml_predictor.py]
              ├─ Strict Chronological 80/20 Train/Test Split
              ├─ Estimators: Linear Regression, Random Forest, Gradient Boosting
              └─ Evaluation Metrics vs. Naive Persistence Baseline (MAE, RMSE, R², Dir. Acc.)
                                      ↓
            [Streamlit Prediction Terminal: dashboard/app.py]
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Technologies Used & Roles
    st.markdown('<p class="term-header">Technologies & Architectural Roles</p>', unsafe_allow_html=True)
    t1, t2 = st.columns(2)
    with t1:
        st.markdown(
            """
            <div class="kpi-card" style="height: 100%;">
                <div style="font-weight: 700; font-size: 0.88rem; color: #38bdf8; margin-bottom: 6px;">Role of Apache Spark</div>
                <p style="font-size: 0.82rem; color: #94a3b8; line-height: 1.5; margin: 0;">
                    • <b>Catalyst Query Optimizer:</b> Translates high-level DataFrame transformations into optimized physical execution plans.<br>
                    • <b>Distributed Window Operations:</b> Computes rolling averages and lags across chronological partitions without cross-node network bottlenecking.<br>
                    • <b>Out-of-Core Processing:</b> Scales computation seamlessly beyond single-node physical RAM boundaries.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with t2:
        st.markdown(
            """
            <div class="kpi-card" style="height: 100%;">
                <div style="font-weight: 700; font-size: 0.88rem; color: #38bdf8; margin-bottom: 6px;">Role of Machine Learning</div>
                <p style="font-size: 0.82rem; color: #94a3b8; line-height: 1.5; margin: 0;">
                    • <b>Nonlinear Pattern Recognition:</b> Tree ensembles capture non-linear interactions between momentum, moving averages, and volatility.<br>
                    • <b>Directional Conviction:</b> Identifies probabilistic price movement direction over selected trading horizons.<br>
                    • <b>Empirical Benchmarking:</b> Grounds model performance against persistence baselines to avoid inflated accuracy claims.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Limitations & Future Scope
    st.markdown('<p class="term-header">Academic Limitations & Future Scope</p>', unsafe_allow_html=True)
    lim1, lim2 = st.columns(2)
    with lim1:
        st.markdown(
            """
            <div class="kpi-card">
                <div style="font-weight: 700; font-size: 0.88rem; color: #f43f5e; margin-bottom: 6px;">System Limitations</div>
                <div style="font-size: 0.82rem; color: #94a3b8; line-height: 1.5;">
                    • Models predict based purely on historical technical price action; unpredictable macroeconomic shocks or earnings announcements are exogenous.<br>
                    • Local Spark instance operates in multi-threaded local mode (`local[2]`) rather than a multi-node distributed cluster.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with lim2:
        st.markdown(
            """
            <div class="kpi-card">
                <div style="font-weight: 700; font-size: 0.88rem; color: #10b981; margin-bottom: 6px;">Future Scope</div>
                <div style="font-size: 0.82rem; color: #94a3b8; line-height: 1.5;">
                    • Ingestion of real-time streaming market tick feeds via <b>Apache Kafka</b> and <b>Spark Structured Streaming</b>.<br>
                    • Integration of natural language processing (FinBERT) for sentiment analysis from financial news feeds.<br>
                    • Deployment on cloud-managed distributed clusters (AWS EMR or Databricks).
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
