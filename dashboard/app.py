"""StockSense: Big Data Analytics in Stock Market Prediction.

Focused Stock Market Prediction Application:
- Big Data processing with Apache PySpark 3.5 & Parquet Storage
- Feature Engineering (PySpark Window Functions, Moving Averages, Lags, Volatility)
- Chronological Machine Learning (Linear Regression, Random Forest, Gradient Boosting)
- Out-of-sample Test Benchmarking against Naive Persistence Baseline
- Professional Financial Analytics UI/UX Architecture
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
# PAGE CONFIGURATION (CLEAN, NO EMOJIS)
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StockSense | Stock Market Prediction",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# PROFESSIONAL FINANCIAL TERMINAL STYLING (CSS)
# Clean, solid dark theme without AI gradients, glowing dots, or cartoons
# ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    :root {
        --bg-main: #000000;
        --bg-panel: #050505;
        --bg-card: #0a0a0a;
        --border-color: #222222;
        --border-subtle: #171717;
        --text-primary: #f8fafc;
        --text-secondary: #a3a3a3;
        --text-muted: #737373;
        --pos-green: #10b981;
        --pos-green-bg: rgba(16, 185, 129, 0.1);
        --neg-red: #ef4444;
        --neg-red-bg: rgba(239, 68, 68, 0.1);
    }

    /* Global Typography & True Black Canvas */
    html, body, [data-testid="stAppViewContainer"],
    [data-testid="stMain"], section.main,
    [data-testid="stHeader"], [data-testid="stSidebar"],
    [data-testid="stSidebarContent"], [data-testid="stToolbar"] {
        background-color: #000000 !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    }

    /* Monospace for Numbers */
    .mono-num {
        font-family: 'JetBrains Mono', monospace !important;
        font-feature-settings: "tnum" 1;
    }

    /* Sidebar Clean Styling */
    [data-testid="stSidebar"] {
        background-color: #050505 !important;
        border-right: 1px solid var(--border-color) !important;
    }
    [data-testid="stSidebar"] * {
        color: #cbd5e1 !important;
    }

    /* Clean Financial Header */
    .header-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #080808;
        border: 1px solid var(--border-color);
        border-radius: 6px;
        padding: 14px 20px;
        margin-bottom: 20px;
    }
    .header-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #f8fafc;
        letter-spacing: -0.01em;
    }
    .header-subtitle {
        font-size: 0.8rem;
        color: #737373;
        margin-left: 10px;
        font-weight: 500;
    }
    .status-tag {
        font-size: 0.74rem;
        font-weight: 600;
        color: #a3a3a3;
        background: #111111;
        border: 1px solid var(--border-color);
        padding: 4px 10px;
        border-radius: 4px;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Standardized Metric Card */
    .metric-card {
        background: #0a0a0a;
        border: 1px solid var(--border-color);
        border-radius: 6px;
        padding: 16px 18px;
    }
    .metric-title {
        font-size: 0.74rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-secondary);
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.55rem;
        font-weight: 700;
        color: #f8fafc;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-sub {
        font-size: 0.75rem;
        color: var(--text-muted);
        margin-top: 5px;
    }

    /* Prediction Forecast Panel */
    .forecast-panel {
        background: #0a0a0a;
        border: 1px solid #262626;
        border-radius: 6px;
        padding: 22px 24px;
        margin-bottom: 20px;
    }

    /* Badges */
    .chip-green {
        display: inline-block;
        background: var(--pos-green-bg);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: var(--pos-green) !important;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.78rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }
    .chip-red {
        display: inline-block;
        background: var(--neg-red-bg);
        border: 1px solid rgba(239, 68, 68, 0.3);
        color: var(--neg-red) !important;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.78rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }
    .chip-neutral {
        display: inline-block;
        background: #111111;
        border: 1px solid var(--border-color);
        color: #a3a3a3 !important;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.78rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Section Subheadings */
    .section-label {
        font-size: 0.82rem;
        font-weight: 700;
        color: #a3a3a3;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin: 18px 0 10px 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .section-label::after {
        content: "";
        flex: 1;
        height: 1px;
        background: var(--border-color);
    }

    /* Clean Pipeline Stage Row */
    .pipeline-strip {
        display: flex;
        align-items: center;
        background: #080808;
        border: 1px solid var(--border-color);
        border-radius: 6px;
        padding: 12px 18px;
        margin-bottom: 18px;
        overflow-x: auto;
    }
    .stage-item {
        display: flex;
        flex-direction: column;
        flex: 1;
        text-align: left;
        padding: 4px 10px;
    }
    .stage-num {
        font-size: 0.7rem;
        font-weight: 700;
        color: #e5e5e5;
        font-family: 'JetBrains Mono', monospace;
    }
    .stage-name {
        font-size: 0.8rem;
        font-weight: 600;
        color: #f1f5f9;
        margin-top: 2px;
    }
    .stage-desc {
        font-size: 0.7rem;
        color: #737373;
    }
    .stage-sep {
        color: #262626;
        font-size: 0.9rem;
        padding: 0 8px;
    }

    /* Clean Solid Button */
    .stButton > button {
        background: #171717 !important;
        color: #ffffff !important;
        border: 1px solid #333333 !important;
        border-radius: 4px !important;
        font-weight: 600 !important;
        font-size: 0.86rem !important;
        padding: 8px 16px !important;
        transition: all 0.15s ease !important;
    }
    .stButton > button:hover {
        background: #262626 !important;
        border-color: #525252 !important;
    }
    hr {
        border-color: var(--border-color) !important;
        margin: 14px 0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────
# PLOTLY TRUE BLACK FINANCIAL THEME
# ─────────────────────────────────────────────────────────────
GRID_COLOR = "#1a1a1a"

def _plotly_layout(title: str = "", height: int = 380, **extra) -> dict:
    """Returns a clean, true black Plotly layout matching institutional financial tools."""
    base = dict(
        template="plotly_dark",
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        title=dict(text=title, font=dict(family="Inter, sans-serif", size=12, color="#e5e5e5")),
        font=dict(family="Inter, sans-serif", color="#a3a3a3", size=11),
        height=height,
        margin=dict(l=45, r=20, t=40, b=35),
        legend=dict(
            bgcolor="#0a0a0a",
            bordercolor="#222222",
            borderwidth=1,
            font=dict(color="#d4d4d4", size=11),
        ),
        xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor="#222222", showline=False),
        yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor="#222222", showline=False),
    )
    base.update(extra)
    return base

# ─────────────────────────────────────────────────────────────
# CACHED DATA INGESTION HELPER
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _cached_api_load(ticker: str, period: str = "1y") -> pd.DataFrame:
    return get_stock_data(tickers=[ticker], period=period)

# ─────────────────────────────────────────────────────────────
# CLEAN HEADER BAR
# ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="header-bar">
        <div>
            <span class="header-title">StockSense</span>
            <span class="header-subtitle">Big Data Analytics in Stock Market Prediction</span>
        </div>
        <div style="display: flex; gap: 8px;">
            <span class="status-tag">Apache PySpark 3.5</span>
            <span class="status-tag">Dual-Market (US / NSE)</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION & DATA INGESTION
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 2px 0 10px 0;">
            <div style="font-weight: 700; font-size: 0.95rem; color: #f8fafc;">Navigation</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        [
            "Prediction Dashboard",
            "Data Processing",
            "Model Evaluation",
        ],
        index=0,
        label_visibility="collapsed",
    )
    st.markdown("---")

    st.markdown('<p class="section-label">Stock Selection</p>', unsafe_allow_html=True)
    CORE_STOCK_PRESETS = ["NVDA", "AAPL", "MSFT", "RELIANCE.NS", "TCS.NS"]
    
    selected_stock = st.selectbox(
        "Stock Ticker",
        CORE_STOCK_PRESETS,
        index=0,
    )

    st.markdown('<p class="section-label">Data Ingestion Source</p>', unsafe_allow_html=True)
    data_mode = st.radio(
        "Ingestion Source",
        ["Live (Yahoo Finance)", "Offline Sample Dataset", "Upload File (CSV/Parquet)"],
        index=0,
        label_visibility="collapsed",
    )

    df_raw: pd.DataFrame = pd.DataFrame()
    data_source_badge = ""
    data_source_type = ""

    if data_mode == "Live (Yahoo Finance)":
        with st.spinner(f"Fetching market data for {selected_stock}..."):
            try:
                df_raw = _cached_api_load(selected_stock, period="1y")
                data_source_badge = "Live Market Data (Yahoo Finance)"
                data_source_type = "LIVE"
            except Exception as exc:
                st.warning(f"Live API unavailable ({exc}). Using offline sample dataset.")
                normalized_name = selected_stock.replace(".", "_")
                sample_p = get_sample_data_dir() / f"{normalized_name}.parquet"
                if not sample_p.exists():
                    sample_p = get_sample_data_dir() / f"{selected_stock}.parquet"
                if sample_p.exists():
                    df_raw = load_local_dataset(sample_p)
                    data_source_badge = "Offline Sample Dataset"
                    data_source_type = "SAMPLE"
                else:
                    st.error("Offline dataset not found.")
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
                data_source_badge = f"Uploaded File ({uploaded.name})"
                data_source_type = "UPLOAD"
            except Exception as exc:
                st.error(f"Error parsing uploaded file: {exc}")
                st.stop()
        else:
            st.info("Upload a historical stock CSV or Parquet file to proceed.")
            st.stop()

    st.markdown("---")
    st.markdown('<p class="section-label">System Status</p>', unsafe_allow_html=True)
    spark_info = get_spark_status()
    if spark_info["pyspark_installed"]:
        st.markdown('<span class="status-tag">Engine: Apache Spark 3.5</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-tag">Engine: Pandas Fallback</span>', unsafe_allow_html=True)

    feed_class = "chip-green" if data_source_type == "LIVE" else "chip-neutral"
    st.markdown(f'<div style="margin-top: 6px;"><span class="{feed_class}">Feed: {data_source_type}</span></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# DATA SANITY & ETL EXECUTION OR RETRIEVAL
# ─────────────────────────────────────────────────────────────
if df_raw.empty:
    st.warning("No data available for the selected stock. Please select another stock or source.")
    st.stop()

close_col = "Adj Close" if "Adj Close" in df_raw.columns else ("Close" if "Close" in df_raw.columns else df_raw.columns[0])
if isinstance(df_raw.columns, pd.MultiIndex):
    if selected_stock in df_raw.columns.get_level_values(1):
        df_flat = df_raw.xs(selected_stock, axis=1, level=1).copy()
    else:
        df_flat = df_raw.iloc[:, :6].copy()
else:
    df_flat = df_raw.copy()

currency_sym = "₹" if selected_stock.endswith(".NS") else "$"

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
    st.markdown(f"### Stock Prediction Dashboard: {selected_stock}")
    st.caption(f"Big Data Feature Pipeline & Chronological Machine Learning Forecast | {data_source_badge}")
    st.markdown("---")

    latest_close = float(df_flat[close_col].iloc[-1])
    latest_date = df_flat.index[-1].strftime("%b %d, %Y") if hasattr(df_flat.index[-1], "strftime") else str(df_flat.index[-1])
    prev_close = float(df_flat[close_col].iloc[-2]) if len(df_flat) > 1 else latest_close
    day_change = latest_close - prev_close
    day_pct = (day_change / prev_close) * 100.0 if prev_close != 0 else 0.0

    # Top KPI Row
    kpi_col1, kpi_col2, kpi_col3 = st.columns([1.2, 1.2, 1.6])
    with kpi_col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Selected Stock</div>
                <div class="metric-value">{selected_stock}</div>
                <div class="metric-sub">Exchange: {'National Stock Exchange (NSE)' if selected_stock.endswith('.NS') else 'US Equities (NASDAQ/NYSE)'}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi_col2:
        change_class = "chip-green" if day_change >= 0 else "chip-red"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Latest Close ({latest_date})</div>
                <div class="metric-value">{currency_sym}{latest_close:,.2f}</div>
                <div class="metric-sub"><span class="{change_class}">{day_change:+.2f} ({day_pct:+.2f}%)</span> vs prev day</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi_col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Data Ingestion Origin</div>
                <div class="metric-value" style="font-size: 1.2rem;">{data_source_badge.split('(')[0].strip()}</div>
                <div class="metric-sub">{len(df_flat)} Records Available | Processed with PySpark</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Historical Stock Price Chart
    st.markdown('<p class="section-label">Historical Stock Price Action & Moving Averages</p>', unsafe_allow_html=True)
    
    fig_hist = go.Figure()
    fig_hist.add_trace(
        go.Scatter(
            x=df_flat.index,
            y=df_flat[close_col],
            name="Close Price",
            mode="lines",
            line=dict(color="#ffffff", width=1.8),
        )
    )
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
        **_plotly_layout(f"{selected_stock} Historical Closing Price & Moving Averages", height=360),
        xaxis_title="Date",
        yaxis_title=f"Price ({currency_sym})",
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # Prediction Controls
    st.markdown('<p class="section-label">Prediction Configuration</p>', unsafe_allow_html=True)
    
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

    # Train and Evaluate Model
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

    # FORECAST PANEL (Clean, Refined, High Visual Hierarchy)
    move_color = "#10b981" if is_positive else "#ef4444"
    move_symbol = "+" if is_positive else ""
    move_badge = "chip-green" if is_positive else "chip-red"

    st.markdown(
        f"""
        <div class="forecast-panel">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 16px;">
                <div>
                    <div style="font-size: 0.76rem; font-weight: 600; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 4px;">
                        Model Forecast | {pred_res['engine_name']}
                    </div>
                    <div style="font-size: 2.6rem; font-weight: 700; color: #ffffff; font-family: 'JetBrains Mono', monospace; line-height: 1.1;">
                        {currency_sym}{forecast_price:,.2f}
                    </div>
                    <div style="margin-top: 10px; display: flex; align-items: center; gap: 10px;">
                        <span class="{move_badge}" style="font-size: 0.95rem; padding: 4px 10px;">
                            Expected Move: {move_symbol}{currency_sym}{abs(delta_val):.2f} ({move_symbol}{delta_pct:.2f}%)
                        </span>
                        <span style="font-size: 0.8rem; color: #94a3b8;">
                            Horizon: <b>T+{horizon_days} Trading Day{'s' if horizon_days > 1 else ''}</b>
                        </span>
                    </div>
                </div>
                <div style="background: #090d14; border: 1px solid var(--border-color); border-radius: 6px; padding: 14px 18px; min-width: 250px;">
                    <div style="font-size: 0.72rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; margin-bottom: 8px;">
                        Out-of-Sample Test Reliability
                    </div>
                    <div style="font-size: 0.82rem; color: #cbd5e1; display: flex; justify-content: space-between; margin-bottom: 4px;">
                        <span>Test MAE:</span> <b class="mono-num">{currency_sym}{pred_res['mae']:.2f}</b>
                    </div>
                    <div style="font-size: 0.82rem; color: #cbd5e1; display: flex; justify-content: space-between; margin-bottom: 4px;">
                        <span>Test RMSE:</span> <b class="mono-num">{currency_sym}{pred_res['rmse']:.2f}</b>
                    </div>
                    <div style="font-size: 0.82rem; color: #cbd5e1; display: flex; justify-content: space-between; margin-bottom: 4px;">
                        <span>Directional Accuracy:</span> <b class="mono-num" style="color: {'#10b981' if pred_res['directional_accuracy_pct'] >= 50 else '#ef4444'};">{pred_res['directional_accuracy_pct']:.1f}%</b>
                    </div>
                    <div style="font-size: 0.82rem; color: #cbd5e1; display: flex; justify-content: space-between;">
                        <span>Naive Baseline MAE:</span> <b class="mono-num">{currency_sym}{pred_res['baseline_mae']:.2f}</b>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Actual versus Predicted Historical Chart
    st.markdown('<p class="section-label">Actual vs. Predicted Historical Performance (Test Window)</p>', unsafe_allow_html=True)
    
    viz_df = pred_res["viz_df"]
    fig_pred = go.Figure()
    fig_pred.add_trace(
        go.Scatter(
            x=viz_df.index,
            y=viz_df["Actual_Close"],
            name="Actual Close",
            mode="lines+markers",
            line=dict(color="#ffffff", width=2),
            marker=dict(size=4),
        )
    )
    fig_pred.add_trace(
        go.Scatter(
            x=viz_df.index,
            y=viz_df["Predicted_Close"],
            name=f"Predicted ({pred_res['engine_name']})",
            mode="lines",
            line=dict(color="#f59e0b", width=1.8, dash="dash"),
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
            height=380,
        ),
        xaxis_title="Date",
        yaxis_title=f"Price ({currency_sym})",
    )
    st.plotly_chart(fig_pred, use_container_width=True)

    # Prediction Summary
    st.markdown('<p class="section-label">Prediction Summary & Feature Contribution</p>', unsafe_allow_html=True)
    
    sum_col1, sum_col2 = st.columns([1.6, 1.4])
    with sum_col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div style="font-weight: 600; font-size: 0.9rem; color: #f8fafc; margin-bottom: 6px;">
                    Model Interpretation & Methodology
                </div>
                <p style="font-size: 0.82rem; color: #94a3b8; line-height: 1.5; margin-bottom: 8px;">
                    The <b>{pred_res['engine_name']}</b> projects an estimated price of 
                    <b style="color: #ffffff;">{currency_sym}{forecast_price:,.2f}</b> over a horizon of <b>{horizon_days} trading day{'s' if horizon_days > 1 else ''}</b>, 
                    representing an expected move of <b style="color: {move_color};">{move_symbol}{delta_pct:.2f}%</b> from the latest close of {currency_sym}{latest_close:,.2f}.
                </p>
                <p style="font-size: 0.82rem; color: #94a3b8; line-height: 1.5; margin-bottom: 8px;">
                    • <b>Data Partition:</b> Evaluated across {pred_res['test_samples']} out-of-sample trading days ({pred_res['test_period']}) with strict chronological order.<br>
                    • <b>Error Tolerance:</b> Test Mean Absolute Error is ±{currency_sym}{pred_res['mae']:.2f}.<br>
                    • <b>Baseline Comparison:</b> Naive Previous-Day baseline achieved an MAE of ±{currency_sym}{pred_res['baseline_mae']:.2f}.
                </p>
                <div style="font-size: 0.74rem; color: #64748b;">
                    Note: Predictions are generated for academic demonstration in Big Data analytics. Stock price movements are non-stationary and carry inherent market risk.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with sum_col2:
        top_features = list(pred_res["feature_importance"].items())[:6]
        if top_features:
            feat_df = pd.DataFrame(top_features, columns=["Feature", "Contribution"])
            fig_feat = px.bar(
                feat_df,
                x="Contribution",
                y="Feature",
                orientation="h",
                color_discrete_sequence=["#e5e5e5"],
            )
            fig_feat.update_layout(
                **_plotly_layout("Top Contributing Features", height=230, margin=dict(l=70, r=20, t=35, b=25))
            )
            st.plotly_chart(fig_feat, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE 2: DATA PROCESSING (BIG DATA PYSPARK PIPELINE)
# ══════════════════════════════════════════════════════════════
elif page == "Data Processing":
    st.markdown("### Big Data Processing Pipeline")
    st.caption(f"Distributed Ingestion, Validation, Feature Engineering & Parquet Serialization | {selected_stock}")
    st.markdown("---")

    # Clean Pipeline Stages Strip (No cartoons or glowing circles)
    st.markdown(
        """
        <div class="pipeline-strip">
            <div class="stage-item">
                <span class="stage-num">01 INGESTION</span>
                <span class="stage-name">Historical OHLCV</span>
                <span class="stage-desc">US & NSE Equities</span>
            </div>
            <div class="stage-sep">/</div>
            <div class="stage-item">
                <span class="stage-num">02 VALIDATION</span>
                <span class="stage-name">Schema & Nulls</span>
                <span class="stage-desc">Forward/Backward Imputation</span>
            </div>
            <div class="stage-sep">/</div>
            <div class="stage-item">
                <span class="stage-num">03 OUTLIERS</span>
                <span class="stage-name">IQR Detection</span>
                <span class="stage-desc">Tagged in Outlier_Flag</span>
            </div>
            <div class="stage-sep">/</div>
            <div class="stage-item">
                <span class="stage-num">04 SPARK WINDOW</span>
                <span class="stage-name">Feature Engineering</span>
                <span class="stage-desc">Lags & Rolling SMA</span>
            </div>
            <div class="stage-sep">/</div>
            <div class="stage-item">
                <span class="stage-num">05 STORAGE</span>
                <span class="stage-name">Snappy Parquet</span>
                <span class="stage-desc">data/processed/</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Re-run execution control
    run_col1, run_col2 = st.columns([2.5, 1.5])
    with run_col1:
        st.write("Execute distributed processing job through Apache PySpark DataFrame APIs:")
    with run_col2:
        if st.button("Run PySpark Pipeline", use_container_width=True):
            with st.spinner("Executing PySpark distributed transformations..."):
                st.session_state[cache_key] = process_stock_data_spark(
                    df_flat,
                    ticker=selected_stock,
                    output_dir=proc_dir,
                )
            st.success("PySpark execution completed successfully.")

    spark_res = st.session_state[cache_key]

    # Processing Metrics
    st.markdown('<p class="section-label">Pipeline Execution Metrics</p>', unsafe_allow_html=True)
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Input / Output Records</div>
                <div class="metric-value">{spark_res['raw_records']} / {spark_res['processed_records']}</div>
                <div class="metric-sub">Cleaned records: <b>{spark_res['processed_records']}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m_col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Duplicates & Missing Handled</div>
                <div class="metric-value">{spark_res['duplicates_removed']} / {spark_res['missing_values_handled']}</div>
                <div class="metric-sub">Imputed cleanly</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m_col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Outliers Flagged (IQR)</div>
                <div class="metric-value">{spark_res['outliers_detected']}</div>
                <div class="metric-sub">Marked in <code>Outlier_Flag</code></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m_col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Execution Duration</div>
                <div class="metric-value">{spark_res['duration_seconds']}s</div>
                <div class="metric-sub">Partitions: <b>{spark_res['partitions']}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Engine Status Row
    st.markdown("<br>", unsafe_allow_html=True)
    eng_name = spark_res["engine"]
    st.markdown(
        f"""
        <div style="background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 6px; padding: 12px 18px; display: flex; align-items: center; justify-content: space-between;">
            <div>
                <span style="font-size: 0.8rem; color: #94a3b8; font-weight: 600;">Processing Engine:</span>
                <span class="status-tag" style="margin-left: 8px;">{eng_name}</span>
            </div>
            <div style="font-size: 0.78rem; color: #94a3b8;">
                Storage Sink: <code style="color: #e5e5e5;">{spark_res['output_parquet']}</code>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Outlier Treatment Explanation Box
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 6px; padding: 14px 18px;">
            <div style="font-size: 0.82rem; font-weight: 600; color: #cbd5e1; margin-bottom: 4px;">
                Outlier Handling Methodology
            </div>
            <div style="font-size: 0.8rem; color: #94a3b8; line-height: 1.5;">
                {spark_res.get('outlier_treatment', 'Outlier detection via Interquartile Range (IQR). Genuine extreme market events are preserved.')}
                Extreme price movements are identified via the IQR threshold <code>[Q1 - 1.5*IQR, Q3 + 1.5*IQR]</code> and annotated in the <code>Outlier_Flag</code> column without dropping historical rows, preserving market reality for downstream models.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Generated Features Specification
    st.markdown('<p class="section-label">Generated Features</p>', unsafe_allow_html=True)
    
    features_list = spark_res.get("generated_features", [])
    feat_chips = " ".join([f'<span class="status-tag" style="margin: 3px 4px;">{f}</span>' for f in features_list])
    st.markdown(
        f"""
        <div class="metric-card" style="margin-bottom: 18px;">
            <div style="font-size: 0.8rem; font-weight: 600; color: #f8fafc; margin-bottom: 8px;">
                Features Produced for Machine Learning ({len(features_list)} Features):
            </div>
            <div>{feat_chips}</div>
            <div style="font-size: 0.74rem; color: #64748b; margin-top: 10px;">
                Calculated using distributed PySpark Window functions over chronological partitions: <code>Window.partitionBy(F.lit("GLOBAL")).orderBy("Date")</code>.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Before and After Data Previews
    st.markdown('<p class="section-label">Before & After Data Previews</p>', unsafe_allow_html=True)
    prev_tab1, prev_tab2 = st.tabs(["Processed Parquet Dataset (Features Included)", "Raw Input Ingestion Sample"])
    
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
    st.markdown(f"### Model Evaluation & Baseline Benchmark: {selected_stock}")
    st.caption("Out-of-Sample Chronological Validation & Baseline Comparison")
    st.markdown("---")

    eval_col1, eval_col2 = st.columns([2.0, 2.0])
    with eval_col1:
        eval_horizon = st.select_slider(
            "Evaluation Horizon",
            options=[1, 5, 10, 15, 30],
            value=1,
            format_func=lambda x: f"{x} Day{'s' if x > 1 else ''} Horizon (T+{x})",
        )
    with eval_col2:
        st.markdown(
            """
            <div style="background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 6px; padding: 10px 14px; font-size: 0.78rem; color: #94a3b8; margin-top: 12px;">
                Partition Method: Strict Chronological 80% Train / 20% Test Split. Zero lookahead leakage.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.spinner(f"Evaluating models for {selected_stock}..."):
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

    # TOP PERFORMER CARD (Clean & Sharp)
    best_chip = "chip-green" if beats_baseline else "chip-neutral"
    status_text = f"Beats Naive Baseline by {imp_pct}% MAE reduction" if beats_baseline else "Competes with Random Walk Naive Baseline"

    st.markdown(
        f"""
        <div class="forecast-panel">
            <div style="font-size: 0.74rem; font-weight: 600; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 4px;">
                Best Performing Model (Test Window)
            </div>
            <div style="font-size: 1.75rem; font-weight: 700; color: #ffffff;">
                {best_res['engine_name']}
            </div>
            <div style="margin-top: 8px; display: flex; align-items: center; gap: 12px; flex-wrap: wrap;">
                <span class="{best_chip}">{status_text}</span>
                <span style="font-size: 0.82rem; color: #94a3b8;">
                    Test MAE: <b class="mono-num" style="color: #ffffff;">{currency_sym}{best_res['mae']:.2f}</b> | 
                    RMSE: <b class="mono-num" style="color: #ffffff;">{currency_sym}{best_res['rmse']:.2f}</b> | 
                    Directional Accuracy: <b class="mono-num" style="color: #ffffff;">{best_res['directional_accuracy_pct']:.1f}%</b>
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Comparative Model Scorecard Table
    st.markdown('<p class="section-label">Model Comparison Scorecard</p>', unsafe_allow_html=True)
    st.dataframe(comp_df, use_container_width=True)

    # Actual vs Predicted Multi-Model Overlay Chart
    st.markdown('<p class="section-label">Actual vs. Multi-Model Predictions (Test Window)</p>', unsafe_allow_html=True)
    
    comb_viz = suite["combined_viz_df"]
    fig_comp = go.Figure()
    
    fig_comp.add_trace(
        go.Scatter(
            x=comb_viz.index,
            y=comb_viz["Actual_Close"],
            name="Actual Close",
            mode="lines+markers",
            line=dict(color="#ffffff", width=2),
            marker=dict(size=4),
        )
    )
    fig_comp.add_trace(
        go.Scatter(
            x=comb_viz.index,
            y=comb_viz["Naive_Baseline"],
            name="Naive Baseline (Close[t-1])",
            mode="lines",
            line=dict(color="#64748b", width=1.5, dash="dot"),
        )
    )
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
                    line=dict(color=colors.get(m, "#cbd5e1"), width=1.6, dash="dash"),
                )
            )

    fig_comp.update_layout(
        **_plotly_layout(
            f"{selected_stock}: Ground Truth vs. All Models over Out-of-Sample Test Window",
            height=400,
        ),
        xaxis_title="Date",
        yaxis_title=f"Price ({currency_sym})",
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    # Academic Methodology
    st.markdown('<p class="section-label">Methodology & Baseline Notes</p>', unsafe_allow_html=True)
    e_col1, e_col2 = st.columns(2)
    with e_col1:
        st.markdown(
            """
            <div class="metric-card">
                <div style="font-weight: 600; font-size: 0.88rem; color: #f8fafc; margin-bottom: 6px;">
                    Chronological Train/Test Partitioning
                </div>
                <p style="font-size: 0.8rem; color: #94a3b8; line-height: 1.5; margin: 0;">
                    Unlike standard cross-sectional ML tasks where random cross-validation is standard, financial time-series strictly requires chronological ordering. Shuffling rows causes <b>lookahead data leakage</b>, where past predictions use future information. StockSense uses the first 80% chronological observations for training and reserves the trailing 20% strictly for testing.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with e_col2:
        st.markdown(
            """
            <div class="metric-card">
                <div style="font-weight: 600; font-size: 0.88rem; color: #f8fafc; margin-bottom: 6px;">
                    Role of the Persistence Baseline
                </div>
                <p style="font-size: 0.8rem; color: #94a3b8; line-height: 1.5; margin: 0;">
                    Under Eugene Fama's Efficient Market Hypothesis, asset prices closely resemble a martingale process: Price(t) ≈ Price(t-1) + error. Comparing models against a <b>Naive Persistence Baseline</b> ensures that models are evaluated for genuine statistical edge rather than merely tracking autocorrelation.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )



