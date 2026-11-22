"""StockSense: Big Data Analytics in Stock Market Prediction.

Focused Stock Market Prediction Application:
- User-Controlled Workflow:
  1. Import Data -> 2. Preview Raw Data -> 3. Process Data -> 4. User Clicks Predict -> 5. Display Results
- 14 Pre-Loaded Global & Indian Company Datasets (NVIDIA, Apple, Microsoft, Google, Amazon, Tesla, Meta, Reliance, TCS, Infosys, HDFC Bank, ICICI Bank, SBI, Wipro)
- Big Data processing with Apache PySpark 3.5 & Parquet Columnar Storage
- Feature Engineering (PySpark Window Functions, Moving Averages, Lags, Volatility)
- Chronological Machine Learning (Linear Regression, Random Forest, Gradient Boosting)
- Out-of-sample Test Benchmarking against Naive Persistence Baseline
- Professional Institutional Financial Terminal UI
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
    get_processed_data_dir,
    get_raw_metadata,
    get_sample_data_dir,
    list_sample_datasets,
    load_raw_dataset,
    parse_raw_uploaded_file,
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
    page_title="StockSense | Stock Market Prediction",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# PROFESSIONAL FINANCIAL TERMINAL STYLING (CSS)
# Clean, solid true-black dark theme without AI gradients, glowing dots, or cartoons
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
        color: #737373;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #f8fafc;
        font-family: 'JetBrains Mono', monospace;
        line-height: 1.2;
    }
    .metric-sub {
        font-size: 0.75rem;
        color: #737373;
        margin-top: 6px;
    }

    /* Status Chips */
    .chip-green {
        color: #10b981;
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.2);
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }
    .chip-red {
        color: #ef4444;
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.2);
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }
    .chip-neutral {
        color: #a3a3a3;
        background: #171717;
        border: 1px solid var(--border-color);
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Forecast Highlight Hero Box */
    .forecast-panel {
        background: #080808;
        border: 1px solid var(--border-color);
        border-left: 3px solid #f59e0b;
        border-radius: 6px;
        padding: 20px 24px;
        margin-bottom: 24px;
    }

    /* Workflow Stepper Bar */
    .workflow-tracker {
        display: flex;
        align-items: center;
        background: #080808;
        border: 1px solid var(--border-color);
        border-radius: 6px;
        padding: 12px 18px;
        margin-bottom: 20px;
        gap: 12px;
        overflow-x: auto;
    }
    .wf-step {
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .wf-badge-done {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }
    .wf-badge-pending {
        background: #171717;
        color: #737373;
        border: 1px solid #262626;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }
    .wf-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: #f8fafc;
    }
    .wf-arrow {
        color: #404040;
        font-size: 0.85rem;
    }

    /* Buttons */
    .stButton>button {
        background-color: #171717 !important;
        color: #f8fafc !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 4px !important;
        font-weight: 600 !important;
        font-size: 0.84rem !important;
        padding: 6px 16px !important;
        transition: all 0.15s ease-in-out !important;
    }
    .stButton>button:hover {
        background-color: #262626 !important;
        border-color: #404040 !important;
    }

    /* Section Subheadings */
    .section-label {
        font-size: 0.74rem;
        font-weight: 700;
        color: #737373;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 18px;
        margin-bottom: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────
# PLOTLY TRUE BLACK THEME
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
# STREAMLINED SIDEBAR: COMPANY DATASET SELECTION ONLY
# Cleaned per user request: No data prediction, no model evaluation,
# no live option, no system status tags in the side panel.
# ─────────────────────────────────────────────────────────────
COMPANY_DATASETS = [
    # US Tech & Global Leaders
    {"ticker": "NVDA", "name": "NVIDIA Corporation (NVDA) - US", "file": "NVDA.parquet"},
    {"ticker": "AAPL", "name": "Apple Inc. (AAPL) - US", "file": "AAPL.parquet"},
    {"ticker": "MSFT", "name": "Microsoft Corporation (MSFT) - US", "file": "MSFT.parquet"},
    {"ticker": "GOOGL", "name": "Alphabet / Google (GOOGL) - US", "file": "GOOGL.parquet"},
    {"ticker": "AMZN", "name": "Amazon.com Inc. (AMZN) - US", "file": "AMZN.parquet"},
    {"ticker": "TSLA", "name": "Tesla Inc. (TSLA) - US", "file": "TSLA.parquet"},
    {"ticker": "META", "name": "Meta Platforms Inc. (META) - US", "file": "META.parquet"},
    # Indian NSE Blue-Chips
    {"ticker": "RELIANCE.NS", "name": "Reliance Industries (RELIANCE.NS) - NSE", "file": "RELIANCE_NS.parquet"},
    {"ticker": "TCS.NS", "name": "Tata Consultancy Services (TCS.NS) - NSE", "file": "TCS_NS.parquet"},
    {"ticker": "INFY.NS", "name": "Infosys Limited (INFY.NS) - NSE", "file": "INFY_NS.parquet"},
    {"ticker": "HDFCBANK.NS", "name": "HDFC Bank (HDFCBANK.NS) - NSE", "file": "HDFCBANK_NS.parquet"},
    {"ticker": "ICICIBANK.NS", "name": "ICICI Bank (ICICIBANK.NS) - NSE", "file": "ICICIBANK_NS.parquet"},
    {"ticker": "SBIN.NS", "name": "State Bank of India (SBIN.NS) - NSE", "file": "SBIN_NS.parquet"},
    {"ticker": "WIPRO.NS", "name": "Wipro Limited (WIPRO.NS) - NSE", "file": "WIPRO_NS.parquet"},
]

with st.sidebar:
    st.markdown(
        """
        <div style="padding: 2px 0 10px 0;">
            <div style="font-weight: 700; font-size: 0.95rem; color: #f8fafc;">Dataset Selection</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    company_names = [c["name"] for c in COMPANY_DATASETS]
    selected_company_name = st.selectbox(
        "Choose Company Dataset",
        company_names,
        index=0,
    )
    # Find matching company
    selected_company = next(c for c in COMPANY_DATASETS if c["name"] == selected_company_name)
    selected_stock = selected_company["ticker"]

    st.markdown("---")
    st.markdown('<p class="section-label">Or Upload Custom File</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Custom Dataset (CSV / Parquet)", type=["csv", "parquet"], label_visibility="collapsed")

# ─────────────────────────────────────────────────────────────
# SESSION STATE & WORKFLOW GATE MANAGEMENT
# ─────────────────────────────────────────────────────────────
if uploaded is not None:
    dataset_key = f"UPLOAD_{uploaded.name}_{uploaded.size}"
else:
    dataset_key = f"SAMPLE_{selected_stock}"

# Initialize session state keys
if "active_dataset_key" not in st.session_state:
    st.session_state["active_dataset_key"] = None
if "processing_completed" not in st.session_state:
    st.session_state["processing_completed"] = False
if "prediction_completed" not in st.session_state:
    st.session_state["prediction_completed"] = False
if "raw_df" not in st.session_state:
    st.session_state["raw_df"] = pd.DataFrame()
if "processed_df" not in st.session_state:
    st.session_state["processed_df"] = None
if "spark_results" not in st.session_state:
    st.session_state["spark_results"] = None
if "prediction_results" not in st.session_state:
    st.session_state["prediction_results"] = None
if "raw_metadata" not in st.session_state:
    st.session_state["raw_metadata"] = {}

# Reset workflow when dataset changes
if st.session_state["active_dataset_key"] != dataset_key:
    st.session_state["active_dataset_key"] = dataset_key
    st.session_state["processing_completed"] = False
    st.session_state["prediction_completed"] = False
    st.session_state["processed_df"] = None
    st.session_state["spark_results"] = None
    st.session_state["prediction_results"] = None
    st.session_state["raw_df"] = pd.DataFrame()
    st.session_state["raw_metadata"] = {}

    df_raw = pd.DataFrame()
    meta = {}
    data_source_badge = ""

    if uploaded is not None:
        try:
            df_raw, up_ticker, meta = parse_raw_uploaded_file(uploaded)
            selected_stock = up_ticker
            data_source_badge = f"Uploaded File ({uploaded.name})"
        except Exception as exc:
            st.error(f"Error parsing uploaded file: {exc}")
    else:
        sample_path = get_sample_data_dir() / selected_company["file"]
        if not sample_path.exists():
            sample_path = get_sample_data_dir() / selected_company["file"].replace(".parquet", ".csv")

        if sample_path.exists():
            try:
                df_raw, meta = load_raw_dataset(sample_path)
                data_source_badge = f"Offline Dataset ({sample_path.name})"
            except Exception as exc:
                st.error(f"Error loading offline file: {exc}")
        else:
            st.error(f"Sample dataset file {selected_company['file']} not found.")

    if not df_raw.empty:
        if isinstance(df_raw.columns, pd.MultiIndex):
            if selected_stock in df_raw.columns.get_level_values(1):
                df_flat = df_raw.xs(selected_stock, axis=1, level=1).copy()
            else:
                df_flat = df_raw.iloc[:, :6].copy()
        else:
            df_flat = df_raw.copy()

        meta["rows"] = len(df_flat)
        meta["columns"] = len(df_flat.columns)
        meta["column_names"] = list(df_flat.columns)
        meta["dtypes"] = {str(c): str(t) for c, t in df_flat.dtypes.items()}
        if isinstance(df_flat.index, pd.DatetimeIndex) and len(df_flat.index) > 0:
            meta["date_range"] = f"{df_flat.index[0].strftime('%b %d, %Y')} to {df_flat.index[-1].strftime('%b %d, %Y')}"
            meta["start_date"] = df_flat.index[0].strftime("%b %d, %Y")
            meta["end_date"] = df_flat.index[-1].strftime("%b %d, %Y")

        st.session_state["raw_df"] = df_flat
        st.session_state["raw_metadata"] = meta
        st.session_state["data_source_badge"] = data_source_badge
        st.session_state["selected_stock"] = selected_stock

# Retrieve current state
df_flat = st.session_state["raw_df"]
raw_metadata = st.session_state.get("raw_metadata", {})
data_source_badge = st.session_state.get("data_source_badge", "")
selected_stock = st.session_state.get("selected_stock", selected_stock)
currency_sym = "₹" if selected_stock.endswith(".NS") else "$"
proc_dir = str(get_processed_data_dir())

if df_flat.empty:
    st.warning("No data loaded. Please select a valid company dataset from the sidebar.")
    st.stop()

close_col = "Adj Close" if "Adj Close" in df_flat.columns else ("Close" if "Close" in df_flat.columns else df_flat.columns[0])


# ─────────────────────────────────────────────────────────────
# MAIN CONTENT TABS (PRESERVES ALL 4 CAPABILITIES WITHOUT CLUTTERING SIDE PANEL)
# ─────────────────────────────────────────────────────────────
tab_dash, tab_eval, tab_info = st.tabs([
    "📈 Prediction Dashboard (User-Controlled Pipeline)",
    "📊 Model Evaluation & Persistence Baseline",
    "📚 Project Information & Big Data Architecture",
])


# ══════════════════════════════════════════════════════════════
# TAB 1: PREDICTION DASHBOARD (4-STAGE PIPELINE)
# ══════════════════════════════════════════════════════════════
with tab_dash:
    st.markdown(f"### Stock Prediction Dashboard: {selected_stock}")
    st.caption(f"Big Data Analytics & Machine Learning Pipeline | {data_source_badge}")
    st.markdown("---")

    # Workflow Status Tracker
    proc_badge = "wf-badge-done" if st.session_state["processing_completed"] else "wf-badge-pending"
    proc_status = "03 PROCESSED (PySpark) ✓" if st.session_state["processing_completed"] else "03 PROCESS DATA (Pending)"

    pred_badge = "wf-badge-done" if st.session_state["prediction_completed"] else "wf-badge-pending"
    pred_status = "04 PREDICTION GENERATED ✓" if st.session_state["prediction_completed"] else "04 PREDICTION (Pending)"

    st.markdown(
        f"""
        <div class="workflow-tracker">
            <div class="wf-step">
                <span class="wf-badge-done">01 IMPORTED ✓</span>
                <span class="wf-label">{selected_stock}</span>
            </div>
            <span class="wf-arrow">→</span>
            <div class="wf-step">
                <span class="wf-badge-done">02 RAW PREVIEW ✓</span>
                <span class="wf-label">{len(df_flat)} Records</span>
            </div>
            <span class="wf-arrow">→</span>
            <div class="wf-step">
                <span class="{proc_badge}">{proc_status}</span>
            </div>
            <span class="wf-arrow">→</span>
            <div class="wf-step">
                <span class="{pred_badge}">{pred_status}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ─────────────────────────────────────────────────────────
    # STEP 1: RAW DATA IMPORT & PREVIEW
    # ─────────────────────────────────────────────────────────
    st.markdown('<p class="section-label">1. Imported Raw Data Preview & Schema Inspection</p>', unsafe_allow_html=True)

    raw_kpi1, raw_kpi2, raw_kpi3, raw_kpi4 = st.columns([1.3, 1.2, 1.5, 1.0])
    with raw_kpi1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Source File / Origin</div>
                <div class="metric-value" style="font-size: 1.15rem;">{raw_metadata.get('source_name', data_source_badge)}</div>
                <div class="metric-sub">Exchange: {'National Stock Exchange (NSE)' if selected_stock.endswith('.NS') else 'US Equities (NASDAQ/NYSE)'}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with raw_kpi2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Raw Dimensions</div>
                <div class="metric-value">{len(df_flat)} <span style="font-size: 0.9rem; color: #737373;">Rows</span></div>
                <div class="metric-sub">{len(df_flat.columns)} Columns Ingested</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with raw_kpi3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Historical Date Range</div>
                <div class="metric-value" style="font-size: 1.1rem;">{raw_metadata.get('start_date', 'N/A')}</div>
                <div class="metric-sub">Through {raw_metadata.get('end_date', 'N/A')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with raw_kpi4:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-title">Pipeline Status</div>
                <div class="metric-value" style="font-size: 1.1rem; color: #f59e0b;">RAW</div>
                <div class="metric-sub">No ETL Applied Yet</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    tbl_col1, tbl_col2, tbl_col3 = st.columns([1.5, 2.0, 1.2])
    with tbl_col1:
        row_limit = st.selectbox(
            "Records to display:",
            options=[10, 25, 50, 100, "All Records"],
            index=0,
            key="raw_row_limit",
        )
    with tbl_col2:
        with st.expander("Inspect Column Schema & Data Types"):
            schema_records = [{"Column Name": c, "Data Type": str(t)} for c, t in df_flat.dtypes.items()]
            st.dataframe(pd.DataFrame(schema_records), use_container_width=True)
    with tbl_col3:
        st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
        csv_data = df_flat.to_csv(index=True)
        st.download_button(
            label="Download Raw CSV",
            data=csv_data,
            file_name=f"{selected_stock}_raw_data.csv",
            mime="text/csv",
            use_container_width=True,
        )

    display_rows = len(df_flat) if row_limit == "All Records" else int(row_limit)
    st.dataframe(df_flat.head(display_rows), use_container_width=True)

    # Raw Closing Price Chart
    fig_raw = go.Figure()
    fig_raw.add_trace(
        go.Scatter(
            x=df_flat.index,
            y=df_flat[close_col],
            name="Raw Close Price",
            mode="lines",
            line=dict(color="#ffffff", width=1.8),
        )
    )
    fig_raw.update_layout(
        **_plotly_layout(f"{selected_stock} Raw Closing Price Action (Unprocessed)", height=280),
        xaxis_title="Date",
        yaxis_title=f"Price ({currency_sym})",
    )
    st.plotly_chart(fig_raw, use_container_width=True)

    st.markdown("---")

    # ─────────────────────────────────────────────────────────
    # STEP 2: EXPLICIT DATA PROCESSING (APACHE PYSPARK)
    # ─────────────────────────────────────────────────────────
    st.markdown('<p class="section-label">2. Big Data Processing & Feature Engineering (Apache PySpark)</p>', unsafe_allow_html=True)

    p_btn_col1, p_btn_col2 = st.columns([2.8, 1.2])
    with p_btn_col1:
        if not st.session_state["processing_completed"]:
            st.markdown(
                """
                <div style="background: #080808; border: 1px solid var(--border-color); border-left: 3px solid #38bdf8; border-radius: 6px; padding: 12px 16px;">
                    <span style="font-weight: 600; font-size: 0.85rem; color: #f8fafc;">Step 2: Processing Required</span>
                    <p style="font-size: 0.78rem; color: #94a3b8; margin: 4px 0 0 0;">
                        Click <b>"Process Data"</b> to validate geometric OHLC bounds, flag IQR outliers, and execute distributed PySpark Window functions (Lags 1-5, Moving Averages 5-50, and 20-Day Volatility).
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            sp_res = st.session_state["spark_results"]
            st.markdown(
                f"""
                <div style="background: #080808; border: 1px solid var(--border-color); border-left: 3px solid #10b981; border-radius: 6px; padding: 12px 16px;">
                    <span style="font-weight: 600; font-size: 0.85rem; color: #10b981;">Data Processing Completed ✓</span>
                    <span style="font-size: 0.78rem; color: #94a3b8; margin-left: 8px;">
                        Engine: <b>{sp_res.get('engine')}</b> | {sp_res.get('processed_records')} Records Cleaned | {len(sp_res.get('generated_features', []))} Features Generated
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
    with p_btn_col2:
        process_btn = st.button("Process Data (PySpark)", use_container_width=True, key="dashboard_process_btn")

    if process_btn:
        with st.spinner(f"Running Apache PySpark distributed ETL pipeline for {selected_stock}..."):
            spark_results = process_stock_data_spark(
                df_flat,
                ticker=selected_stock,
                output_dir=proc_dir,
            )
            st.session_state["spark_results"] = spark_results
            st.session_state["processed_df"] = spark_results.get("processed_df", df_flat)
            st.session_state["processing_completed"] = True
            st.session_state["prediction_completed"] = False
            st.session_state["prediction_results"] = None
            st.success("Apache PySpark processing completed successfully.")
            st.rerun()

    if st.session_state["processing_completed"]:
        sp_res = st.session_state["spark_results"]
        pt1, pt2, pt3, pt4 = st.columns(4)
        with pt1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Cleaned Records</div>
                    <div class="metric-value">{sp_res['processed_records']}</div>
                    <div class="metric-sub">Input: {sp_res['raw_records']} rows</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with pt2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Outliers Flagged (IQR)</div>
                    <div class="metric-value">{sp_res['outliers_detected']}</div>
                    <div class="metric-sub">Preserved in <code>Outlier_Flag</code></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with pt3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">ETL Duration & Partitions</div>
                    <div class="metric-value">{sp_res['duration_seconds']}s</div>
                    <div class="metric-sub">Partitions: {sp_res['partitions']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with pt4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Parquet Storage Sink</div>
                    <div class="metric-value" style="font-size: 1.05rem;">PERSISTED</div>
                    <div class="metric-sub"><code>data/processed/</code></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.expander("Preview Processed Parquet Dataset (11 Engineered Features)"):
            st.dataframe(st.session_state["processed_df"].tail(10), use_container_width=True)

    st.markdown("---")

    # ─────────────────────────────────────────────────────────
    # STEP 3: USER-CONTROLLED PREDICTION (PREDICT BUTTON)
    # ─────────────────────────────────────────────────────────
    st.markdown('<p class="section-label">3. Machine Learning Stock Prediction</p>', unsafe_allow_html=True)

    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.5, 1.5, 1.0])
    with ctrl_col1:
        horizon_days = st.select_slider(
            "Prediction Horizon (Trading Sessions)",
            options=[1, 5, 10, 15, 30],
            value=1,
            format_func=lambda x: f"{x} Session{'s' if x > 1 else ''} Forward (T+{x})",
        )
    with ctrl_col2:
        model_choice = st.selectbox(
            "Machine Learning Model",
            ["Linear Regression", "Random Forest", "Gradient Boosting"],
            index=1,
        )
    with ctrl_col3:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        predict_btn = st.button(
            "Generate Prediction",
            disabled=not st.session_state["processing_completed"],
            use_container_width=True,
            key="dashboard_predict_btn",
        )

    if (
        st.session_state.get("active_horizon") != horizon_days
        or st.session_state.get("active_model") != model_choice
    ):
        st.session_state["prediction_completed"] = False
        st.session_state["prediction_results"] = None

    if not st.session_state["processing_completed"]:
        st.markdown(
            """
            <div style="background: #080808; border: 1px solid var(--border-color); border-left: 3px solid #737373; border-radius: 6px; padding: 14px 18px; margin-top: 10px;">
                <div style="font-weight: 600; font-size: 0.85rem; color: #a3a3a3;">Prediction Disabled (Processing Required)</div>
                <div style="font-size: 0.78rem; color: #737373; margin-top: 4px;">
                    Machine learning models require engineered window features. Please click <b>"Process Data"</b> above to run the PySpark pipeline before initiating prediction.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif not st.session_state["prediction_completed"] and not predict_btn:
        st.markdown(
            f"""
            <div style="background: #080808; border: 1px solid var(--border-color); border-left: 3px solid #f59e0b; border-radius: 6px; padding: 14px 18px; margin-top: 10px;">
                <div style="font-weight: 600; font-size: 0.85rem; color: #f8fafc;">Ready for Prediction</div>
                <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 4px;">
                    Dataset is processed and ready. Select your horizon (<b>T+{horizon_days}</b>) and model (<b>{model_choice}</b>), then click <b>"Generate Prediction"</b> to train models and forecast prices.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if predict_btn and st.session_state["processing_completed"]:
        with st.spinner(f"Training {model_choice} on chronological features for {selected_stock}..."):
            try:
                pred_res = train_and_evaluate_model(
                    ohlcv_df=st.session_state["processed_df"],
                    model_name=model_choice,
                    horizon=horizon_days,
                    test_size=0.20,
                )
                st.session_state["prediction_results"] = pred_res
                st.session_state["active_horizon"] = horizon_days
                st.session_state["active_model"] = model_choice
                st.session_state["prediction_completed"] = True
                st.rerun()
            except Exception as exc:
                st.error(f"Prediction modeling error: {exc}")

    if st.session_state["prediction_completed"] and st.session_state["prediction_results"]:
        pred_res = st.session_state["prediction_results"]
        forecast_price = pred_res["next_day_forecast"]
        delta_val = pred_res["predicted_change"]
        delta_pct = pred_res["predicted_change_pct"]
        is_positive = delta_val >= 0

        move_color = "#10b981" if is_positive else "#ef4444"
        move_symbol = "+" if is_positive else ""
        move_badge = "chip-green" if is_positive else "chip-red"

        st.markdown(
            f"""
            <div class="forecast-panel" style="margin-top: 16px;">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 16px;">
                    <div>
                        <div style="font-size: 0.76rem; font-weight: 600; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 4px;">
                            Model Forecast | {pred_res['engine_name']}
                        </div>
                        <div style="font-size: 2.6rem; font-weight: 700; color: #ffffff; font-family: 'JetBrains Mono', monospace; line-height: 1.1;">
                            {currency_sym}{forecast_price:,.2f}
                        </div>
                        <div style="margin-top: 10px; display: flex; align-items: center; gap: 10px; flex-wrap: wrap;">
                            <span class="{move_badge}" style="font-size: 0.95rem; padding: 4px 10px;">
                                Expected Move: {move_symbol}{currency_sym}{abs(delta_val):.2f} ({move_symbol}{delta_pct:.2f}%)
                            </span>
                            <span style="font-size: 0.8rem; color: #94a3b8;">
                                Origin: <b>{pred_res['forecast_origin_date']}</b> → Target Session: <b style="color: #ffffff;">{pred_res['target_trading_date']} (T+{horizon_days})</b>
                            </span>
                        </div>
                    </div>
                    <div style="background: #090d14; border: 1px solid var(--border-color); border-radius: 6px; padding: 14px 18px; min-width: 260px;">
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
                            <span>Model Dir. Accuracy:</span> <b class="mono-num" style="color: {'#10b981' if pred_res['directional_accuracy_pct'] >= 50 else '#ef4444'};">{pred_res['directional_accuracy_pct']:.1f}%</b>
                        </div>
                        <div style="font-size: 0.82rem; color: #cbd5e1; display: flex; justify-content: space-between;">
                            <span>Baseline Dir. Accuracy:</span> <b class="mono-num">{pred_res['baseline_directional_acc']:.1f}%</b>
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Actual vs Predicted Historical Chart
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

        # Prediction Interpretation & Top Feature Weights
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
                        <b style="color: #ffffff;">{currency_sym}{forecast_price:,.2f}</b> over a horizon of <b>{horizon_days} trading session{'s' if horizon_days > 1 else ''}</b>, 
                        representing an expected move of <b style="color: {move_color};">{move_symbol}{delta_pct:.2f}%</b> from the latest close of {currency_sym}{pred_res['current_price']:,.2f}.
                    </p>
                    <p style="font-size: 0.82rem; color: #94a3b8; line-height: 1.5; margin-bottom: 8px;">
                        • <b>Origin to Target:</b> Forecast origin <b>{pred_res['forecast_origin_date']}</b> targeting trading session <b>{pred_res['target_trading_date']}</b>.<br>
                        • <b>Data Partition:</b> Evaluated across {pred_res['test_samples']} out-of-sample trading days ({pred_res['test_period']}) with strict chronological ordering.<br>
                        • <b>Error Tolerance:</b> Test MAE is ±{currency_sym}{pred_res['mae']:.2f} (RMSE: ±{currency_sym}{pred_res['rmse']:.2f}).<br>
                        • <b>Baseline Comparison:</b> Naive Persistence baseline achieved an MAE of ±{currency_sym}{pred_res['baseline_mae']:.2f}.
                    </p>
                    <div style="font-size: 0.74rem; color: #64748b;">
                        Note: Forecasts are quantitative statistical estimates for academic demonstration in Big Data analytics. Financial markets involve non-stationarity and inherent capital risk; this is not financial investment advice.
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
# TAB 2: MODEL EVALUATION & PERSISTENCE BASELINE
# ══════════════════════════════════════════════════════════════
with tab_eval:
    st.markdown(f"### Model Evaluation & Persistence Baseline: {selected_stock}")
    st.caption("Out-of-Sample Chronological Validation & Baseline Comparison")
    st.markdown("---")

    if not st.session_state["processing_completed"]:
        st.warning("Data processing has not yet been executed. Please click 'Process Data' on the Prediction Dashboard tab first.")
    else:
        eval_col1, eval_col2 = st.columns([2.0, 2.0])
        with eval_col1:
            eval_horizon = st.select_slider(
                "Evaluation Horizon (Trading Sessions)",
                options=[1, 5, 10, 15, 30],
                value=1,
                format_func=lambda x: f"{x} Session{'s' if x > 1 else ''} Horizon (T+{x})",
                key="eval_tab_horizon",
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

        eval_cache_key = f"eval_suite_{selected_stock}_{eval_horizon}"
        if eval_cache_key not in st.session_state:
            with st.spinner(f"Evaluating models for {selected_stock}..."):
                try:
                    st.session_state[eval_cache_key] = evaluate_models_suite(
                        ohlcv_df=st.session_state["processed_df"],
                        horizon=eval_horizon,
                        test_size=0.20,
                    )
                except Exception as exc:
                    st.error(f"Error evaluating model suite: {exc}")

        if eval_cache_key in st.session_state:
            suite = st.session_state[eval_cache_key]
            comp_df = suite["comparison_df"]
            best_res = suite["best_model_res"]
            beats_baseline = suite["beats_baseline"]
            imp_pct = suite["improvement_pct"]

            best_chip = "chip-green" if beats_baseline else "chip-neutral"
            status_text = f"Beats Naive Baseline by {imp_pct}% MAE reduction" if beats_baseline else "Evaluated against Random Walk Persistence Baseline (EMH active)"

            st.markdown(
                f"""
                <div class="forecast-panel">
                    <div style="font-size: 0.74rem; font-weight: 600; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 4px;">
                        Best Performing Model (Out-of-Sample Test Window)
                    </div>
                    <div style="font-size: 1.75rem; font-weight: 700; color: #ffffff;">
                        {best_res['engine_name']}
                    </div>
                    <div style="margin-top: 8px; display: flex; align-items: center; gap: 12px; flex-wrap: wrap;">
                        <span class="{best_chip}">{status_text}</span>
                        <span style="font-size: 0.82rem; color: #94a3b8;">
                            Test MAE: <b class="mono-num" style="color: #ffffff;">{currency_sym}{best_res['mae']:.2f}</b> | 
                            RMSE: <b class="mono-num" style="color: #ffffff;">{currency_sym}{best_res['rmse']:.2f}</b> | 
                            R²: <b class="mono-num" style="color: #ffffff;">{best_res['r2_score']:.4f}</b> | 
                            Dir. Accuracy: <b class="mono-num" style="color: #ffffff;">{best_res['directional_accuracy_pct']:.1f}%</b>
                        </span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown('<p class="section-label">Model Comparison Scorecard</p>', unsafe_allow_html=True)
            st.dataframe(comp_df, use_container_width=True)

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


# ══════════════════════════════════════════════════════════════
# TAB 3: PROJECT INFORMATION & ARCHITECTURE
# ══════════════════════════════════════════════════════════════
with tab_info:
    st.markdown("### Project Architecture & Academic Methodology")
    st.caption("StockSense | Big Data Analytics in Stock Market Prediction")
    st.markdown("---")

    p_col1, p_col2 = st.columns(2)
    with p_col1:
        st.markdown(
            """
            <div class="metric-card" style="height: 100%;">
                <div class="metric-title">Problem Statement</div>
                <p style="font-size: 0.82rem; color: #cbd5e1; line-height: 1.5; margin-top: 6px;">
                    Financial markets generate massive volumes of noisy, high-frequency, non-stationary time-series data. Standard single-threaded in-memory tools (e.g. basic Pandas) fail to scale due to RAM saturation and lack distributed windowing primitives. Furthermore, financial time-series forecasting is plagued by subtle <b>lookahead data leakage</b> (such as backward filling, shuffling train/test splits, or computing rolling features across boundaries), resulting in spuriously optimistic models that collapse in live deployment.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with p_col2:
        st.markdown(
            """
            <div class="metric-card" style="height: 100%;">
                <div class="metric-title">Proposed Solution</div>
                <p style="font-size: 0.82rem; color: #cbd5e1; line-height: 1.5; margin-top: 6px;">
                    StockSense implements an end-to-end distributed Big Data analytics system utilizing <b>Apache PySpark 3.5</b> and <b>Apache Parquet</b> columnar storage. It executes distributed schema validation, geometric OHLC boundary checks, IQR outlier flagging, and PySpark Window-based feature engineering (multi-period lags, moving averages, rolling volatility). It couples distributed ETL directly to chronologically split Machine Learning models evaluated rigorously against an empirical <b>Naive Persistence Baseline</b>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ASCII Architecture Diagram
    st.markdown('<p class="section-label">End-to-End System Architecture</p>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="background: #050505; border: 1px solid var(--border-color); border-radius: 6px; padding: 16px 20px; font-family: 'JetBrains Mono', monospace; font-size: 0.74rem; color: #38bdf8; line-height: 1.45; overflow-x: auto; white-space: pre;">
[Data Ingestion: data_fetcher.py]
  |-- Pre-Cached Offline Parquet & CSV Datasets (14 Global & NSE Blue-Chips in data/sample/)
  `-- File Upload Interface (User-provided CSV/Parquet)
       |
       v
[User-Controlled Preview: dashboard/app.py]
  `-- Raw Unprocessed Table Preview, Column Dtypes, Date Range, CSV Download
       |
       v (User Clicks "Process Data")
[Distributed Processing Engine: spark_engine.py]
  |-- Engine: Apache PySpark 3.5 (local[2], Catalyst Optimizer) with Pandas Fallback
  |-- Data Cleaning: Deduplication, Forward-fill Imputation (Strictly NO backward fill)
  |-- Geometric OHLC Validation: High >= Open/Close, Low <= Open/Close, Volume >= 0
  |-- Outlier Handling: IQR on Daily Return annotated in Outlier_Flag (Original data preserved)
  `-- Window Operations: Chronological Lags (1,2,3,5), Rolling SMA (5,10,20,50), Volatility (20d)
       |
       v
[Columnar Storage Sink: Parquet Format (data/processed/)]
  `-- Direct PySpark Parquet Writer (DataFrame.write.mode("overwrite").parquet(...))
       |
       v (User Clicks "Generate Prediction")
[Machine Learning Engine: ml_predictor.py]
  |-- Target Alignment: T+h trading sessions (excludes calendar non-trading sessions)
  |-- Chronological Partitioning: 80% Train / 20% Test (Strictly NO random shuffling)
  |-- Preprocessing: StandardScaler fit strictly on Training partition
  |-- Estimators: Linear Regression, Random Forest Regressor, Gradient Boosting
  `-- Evaluation Suite: Dynamic benchmarking vs. Naive Persistence Baseline (MAE, RMSE, R², Dir. Acc.)
       |
       v
[User Interface: dashboard/app.py]
  |-- Prediction Dashboard (User-Controlled 4-Step Pipeline: Import -> Preview -> Process -> Predict)
  |-- Model Evaluation Suite (Multi-Model Comparison Scorecard & Baseline Validation)
  `-- Project Information (Academic Dossier, 5Vs, Spark & ML Roles)
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 5Vs of Big Data
    st.markdown('<p class="section-label">5Vs of Big Data in Stock Market Prediction</p>', unsafe_allow_html=True)
    v1, v2, v3, v4, v5 = st.columns(5)
    with v1:
        st.markdown(
            """
            <div class="metric-card">
                <div style="font-size: 0.74rem; font-weight: 700; color: #ffffff;">VOLUME</div>
                <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 4px;">Multi-year historical tick series, high-frequency quotes, and trade volumes across global exchanges.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with v2:
        st.markdown(
            """
            <div class="metric-card">
                <div style="font-size: 0.74rem; font-weight: 700; color: #ffffff;">VELOCITY</div>
                <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 4px;">Rapid session-by-session price updates requiring cached ingestion pipelines and stream readiness.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with v3:
        st.markdown(
            """
            <div class="metric-card">
                <div style="font-size: 0.74rem; font-weight: 700; color: #ffffff;">VARIETY</div>
                <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 4px;">Structured Parquet, tabular CSV, multi-index API responses, and engineered technical matrices.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with v4:
        st.markdown(
            """
            <div class="metric-card">
                <div style="font-size: 0.74rem; font-weight: 700; color: #ffffff;">VERACITY</div>
                <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 4px;">Geometric OHLC validation, missing data isolation, and IQR outlier detection preserving true market action.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with v5:
        st.markdown(
            """
            <div class="metric-card">
                <div style="font-size: 0.74rem; font-weight: 700; color: #ffffff;">VALUE</div>
                <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 4px;">Actionable directional price forecasts and risk-managed horizon estimations for market participants.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Technologies Used & Roles
    st.markdown('<p class="section-label">Technologies & Architectural Roles</p>', unsafe_allow_html=True)
    t1, t2 = st.columns(2)
    with t1:
        st.markdown(
            """
            <div class="metric-card" style="height: 100%;">
                <div style="font-weight: 600; font-size: 0.85rem; color: #cbd5e1; margin-bottom: 6px;">Role of Apache Spark</div>
                <p style="font-size: 0.8rem; color: #94a3b8; line-height: 1.5; margin: 0;">
                    • <b>Catalyst Optimizer:</b> Compiles high-level DataFrame transformations into optimized physical execution plans.<br>
                    • <b>Windowed Feature Engineering:</b> Calculates chronological rolling moving averages, lags, and rolling volatility across dataset partitions without cross-node network bottlenecks.<br>
                    • <b>Scalable Columnar Persistence:</b> Directly persists cleaned partitions to Snappy-compressed Apache Parquet format for fast subsequent reads.<br>
                    • <b>Out-of-Core Processing:</b> Seamlessly handles dataset sizes exceeding physical driver RAM limits.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with t2:
        st.markdown(
            """
            <div class="metric-card" style="height: 100%;">
                <div style="font-weight: 600; font-size: 0.85rem; color: #cbd5e1; margin-bottom: 6px;">Role of Machine Learning</div>
                <p style="font-size: 0.8rem; color: #94a3b8; line-height: 1.5; margin: 0;">
                    • <b>Nonlinear Pattern Recognition:</b> Ensemble architectures (Random Forest, Gradient Boosting) capture non-linear relationships among technical momentum, moving averages, and volatility.<br>
                    • <b>Multi-Horizon Forecasting:</b> Projects price levels and directional deltas across configurable trading horizons (1 to 30 sessions).<br>
                    • <b>Empirical Benchmarking:</b> Evaluates models against a dynamic Naive Persistence Baseline to ensure genuine predictive alpha rather than simple autoregressive tracking.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
