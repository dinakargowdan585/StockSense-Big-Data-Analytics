"""StockSense: Big Data Analytics in Stock Market Prediction.

Academic Activity-Based Learning (ABL) Platform:
- Big Data ingestion & cleaning with Apache PySpark
- Descriptive, Diagnostic, Predictive, and Prescriptive Analytics
- Chronological Machine Learning predictions with Naive Baseline Benchmarks
- Empirical PySpark vs. Pandas Performance Benchmarking
- Institutional Financial Terminal UI
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
    page_title="StockSense | Financial Analytics",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# INSTITUTIONAL FINANCIAL TERMINAL THEME (CSS)
# ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Institutional Dark Theme ── */
    :root {
        --bg-main: #0b0f17;
        --bg-card: #121824;
        --bg-hover: #192233;
        --border-color: #1e293b;
        --border-subtle: #2d3b55;
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --accent-blue: #38bdf8;
        --accent-violet: #818cf8;
        --pos-green: #22c55e;
        --neg-red: #ef4444;
    }

    html, body, [data-testid="stAppViewContainer"],
    [data-testid="stMain"], section.main {
        background-color: var(--bg-main) !important;
        color: var(--text-primary) !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0e131d !important;
        border-right: 1px solid var(--border-color);
    }
    [data-testid="stSidebar"] * {
        color: #cbd5e1 !important;
    }

    /* Metric Cards */
    [data-testid="stMetric"] {
        background-color: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 6px;
        padding: 12px 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.3);
    }
    [data-testid="stMetricLabel"] {
        color: var(--text-secondary) !important;
        font-size: 0.76rem !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
        font-size: 1.35rem !important;
        font-weight: 600 !important;
    }
    [data-testid="stMetricDelta"] {
        font-size: 0.8rem !important;
    }

    /* Clean Badges */
    .badge-spark {
        background-color: #c2410c;
        color: #fff !important;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.75rem;
        display: inline-block;
        letter-spacing: 0.02em;
    }
    .badge-pandas {
        background-color: #1d4ed8;
        color: #fff !important;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.75rem;
        display: inline-block;
        letter-spacing: 0.02em;
    }
    .badge-live {
        background-color: #065f46;
        color: #34d399 !important;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 600;
    }
    .badge-offline {
        background-color: #374151;
        color: #9ca3af !important;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 600;
    }

    /* Section Headers */
    .term-header {
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--accent-blue);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-top: 14px;
        margin-bottom: 8px;
        border-left: 2px solid var(--accent-blue);
        padding-left: 8px;
    }

    /* Financial Data Container */
    .terminal-box {
        background-color: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 6px;
        padding: 16px;
        margin: 10px 0 16px 0;
    }

    hr {
        border-color: var(--border-color) !important;
        margin: 14px 0 !important;
    }

    /* Tables */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--border-color);
        border-radius: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────
# PLOTLY THEME & COLOR PALETTE
# ─────────────────────────────────────────────────────────────
PLOTLY_TEMPLATE = "plotly_dark"
PLOTLY_BG = "#0b0f17"
PLOTLY_CARD_BG = "#121824"
GRID_COLOR = "#1e293b"

PALETTE = [
    "#38bdf8", "#34d399", "#f87171", "#a78bfa",
    "#fb923c", "#f472b6", "#facc15", "#4ade80",
]

def _plotly_layout(title: str = "", height: int = 400, **extra) -> dict:
    return dict(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=PLOTLY_CARD_BG,
        plot_bgcolor=PLOTLY_BG,
        title=dict(text=title, font=dict(size=13, color="#e2e8f0")),
        height=height,
        margin=dict(l=45, r=20, t=40, b=35),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#cbd5e1", size=11)),
        xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor="#334155"),
        yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor="#334155"),
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
# SIDEBAR NAVIGATION & DATA CONTROLS
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### **StockSense**")
    st.caption("Big Data Analytics & Prediction Platform")
    st.markdown("---")

    # Clean, non-emoji navigation
    page = st.radio(
        "Navigation",
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
    )
    st.markdown("---")

    st.markdown("#### Data Ingestion Mode")
    ingestion_mode = st.selectbox(
        "Data Source",
        ["Live / Yahoo Finance API", "Local Offline Datasets", "Upload File (CSV/Parquet)"],
        index=0,
    )

    df: pd.DataFrame = pd.DataFrame()
    data_origin_tag = "LIVE_API"
    data_origin_label = "Yahoo Finance API"
    selected_tickers: list[str] = []

    # Benchmark stock universe required by user specification:
    CORE_STOCK_PRESETS = ["NVDA", "AAPL", "MSFT", "RELIANCE.NS", "TCS.NS"]

    if ingestion_mode == "Live / Yahoo Finance API":
        market_universe = st.selectbox(
            "Equities Watchlist",
            [
                "ABL Core Set (NVDA, AAPL, MSFT, RELIANCE, TCS)",
                "US Tech & Indices (NVDA, AAPL, MSFT, VOO, SPY)",
                "Indian NSE Universe (RELIANCE.NS, TCS.NS, INFY.NS)",
                "Custom Selection",
            ],
            index=0,
        )

        if "ABL Core Set" in market_universe:
            default_selection = CORE_STOCK_PRESETS
            options_pool = CORE_STOCK_PRESETS + ["GOOGL", "AMZN", "INFY.NS"]
        elif "US Tech" in market_universe:
            default_selection = ["NVDA", "AAPL", "MSFT", "VOO"]
            options_pool = US_PRESETS
        elif "Indian NSE" in market_universe:
            default_selection = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
            options_pool = NSE_PRESETS
        else:
            default_selection = ["NVDA", "AAPL", "RELIANCE.NS"]
            options_pool = CORE_STOCK_PRESETS + US_PRESETS + NSE_PRESETS

        chosen_tickers = st.multiselect("Active Tickers", options=options_pool, default=default_selection)
        custom_input = st.text_input("Add Ticker (e.g. TSLA, HDFCBANK.NS)", "")
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
        start_date = st.date_input("Start Date", value=default_start, max_value=today - timedelta(days=7))
        end_date = st.date_input("End Date", value=today, min_value=start_date + timedelta(days=7), max_value=today)

        days_diff = max((end_date - start_date).days, 1)
        period = "1y" if days_diff <= 370 else ("2y" if days_diff <= 740 else "5y")

        if st.button("Refresh Ingestion Cache", use_container_width=True):
            st.cache_data.clear()
            st.success("Cache cleared — pulling latest quotes.")

        with st.spinner("Ingesting market data..."):
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
    spark_status = get_spark_status()
    if spark_status["pyspark_installed"]:
        st.markdown('<span class="badge-spark">Apache PySpark Active</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-pandas">Engine: Pandas Fallback</span>', unsafe_allow_html=True)

    if data_origin_tag == "LIVE_API":
        st.markdown('<span class="badge-live">Source: Live Yahoo Finance</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-offline">Source: Local Cached/Sample</span>', unsafe_allow_html=True)

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
    st.title("Market Overview")
    st.caption(f"Watchlist Telemetry · {data_origin_label} · Active Assets: {len(available_tickers)}")
    st.markdown("---")

    # 1. Primary KPI Grid
    st.markdown('<p class="term-header">Asset Snapshot & Technical Benchmarks</p>', unsafe_allow_html=True)
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
                delta=f"{daily_chg:+.2f}%",
            )
            st.caption(f"52W Range: {curr_sym}{low_52w:,.1f} - {curr_sym}{high_52w:,.1f} | Cap: {mcap_str}")

    st.markdown("---")

    # 2. Normalized Price Performance Chart
    st.markdown('<p class="term-header">Comparative Price Performance (Base = 100)</p>', unsafe_allow_html=True)
    normalized = (close_matrix / close_matrix.iloc[0]) * 100.0

    fig_norm = go.Figure()
    for ticker in available_tickers:
        fig_norm.add_trace(
            go.Scatter(
                x=normalized.index,
                y=normalized[ticker].round(2),
                mode="lines",
                name=ticker,
                line=dict(width=1.8, color=_color_for(ticker, available_tickers)),
                hovertemplate=f"<b>{ticker}</b><br>Date: %{{x|%b %d, %Y}}<br>Value: %{{y:.2f}}<extra></extra>",
            )
        )
    fig_norm.add_hline(y=100, line_dash="dot", line_color="#475569", annotation_text="Base 100")
    fig_norm.update_layout(**_plotly_layout(height=410))
    st.plotly_chart(fig_norm, use_container_width=True)

    # 3. Top Gainers & Losers + Performance Rankings
    st.markdown('<p class="term-header">Watchlist Performance & Volatility Rankings</p>', unsafe_allow_html=True)
    rank_df = compute_market_performance_rankings(df, available_tickers)
    if not rank_df.empty:
        col_t1, col_t2 = st.columns([3, 1])
        with col_t1:
            st.dataframe(rank_df, use_container_width=True, hide_index=True)
        with col_t2:
            st.markdown("#### Performance Extremes")
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
                <div style="background-color: #121824; border: 1px solid #1e293b; border-radius: 6px; padding: 12px; margin-bottom: 8px;">
                    <small style="color: #94a3b8;">TOP PERFORMER</small><br>
                    <b style="color: #22c55e; font-size: 1.1rem;">{best_asset['Ticker']}</b>: +{best_ret:.2f}%
                </div>
                <div style="background-color: #121824; border: 1px solid #1e293b; border-radius: 6px; padding: 12px;">
                    <small style="color: #94a3b8;">LAGGING ASSET</small><br>
                    <b style="color: #ef4444; font-size: 1.1rem;">{worst_asset['Ticker']}</b>: {worst_ret:.2f}%
                </div>
                """,
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════
# PAGE B: STOCK ANALYSIS
# ══════════════════════════════════════════════════════════════
elif page == "Stock Analysis":
    st.title("Stock Analysis")
    st.caption("Technical OHLCV Indicators, Trend Smoothing, Momentum, and Cross-Asset Correlation")
    st.markdown("---")

    sel_stock = st.selectbox("Select Target Equity", options=available_tickers, index=0)

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

    # Compute technicals
    sma_20 = close_s.rolling(20, min_periods=1).mean()
    sma_50 = close_s.rolling(50, min_periods=1).mean()

    # RSI (14-day)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi_s = 100 - (100 / (1 + rs))

    # Subplots: Candlestick + Volume + RSI
    fig_stock = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.60, 0.20, 0.20],
        subplot_titles=(f"{sel_stock} Price & Moving Averages", "Trading Volume", "Relative Strength Index (RSI 14)"),
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
            increasing_line_color="#22c55e",
            decreasing_line_color="#ef4444",
        ),
        row=1, col=1,
    )
    fig_stock.add_trace(
        go.Scatter(x=t_data.index, y=sma_20, mode="lines", name="SMA 20", line=dict(color="#fb923c", width=1.4)),
        row=1, col=1,
    )
    fig_stock.add_trace(
        go.Scatter(x=t_data.index, y=sma_50, mode="lines", name="SMA 50", line=dict(color="#818cf8", width=1.4, dash="dash")),
        row=1, col=1,
    )

    # 2. Volume
    vol_colors = np.where(close_s.diff() >= 0, "#22c55e", "#ef4444")
    fig_stock.add_trace(
        go.Bar(x=t_data.index, y=volume_s, name="Volume", marker_color=vol_colors.tolist(), opacity=0.8),
        row=2, col=1,
    )

    # 3. RSI
    fig_stock.add_trace(
        go.Scatter(x=t_data.index, y=rsi_s, name="RSI (14)", line=dict(color="#38bdf8", width=1.4)),
        row=3, col=1,
    )
    fig_stock.add_hline(y=70, line_dash="dot", line_color="#f87171", row=3, col=1)
    fig_stock.add_hline(y=30, line_dash="dot", line_color="#34d399", row=3, col=1)

    fig_stock.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=PLOTLY_CARD_BG,
        plot_bgcolor=PLOTLY_BG,
        height=620,
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
            paper_bgcolor=PLOTLY_CARD_BG,
            plot_bgcolor=PLOTLY_BG,
            height=340,
            margin=dict(l=40, r=40, t=30, b=30),
        )
        st.plotly_chart(fig_corr, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE C: BIG DATA PROCESSING (ACADEMIC HIGHLIGHT)
# ══════════════════════════════════════════════════════════════
elif page == "Big Data Processing":
    st.title("Big Data Processing Engine")
    st.caption("Distributed ETL Pipeline with Apache Spark (PySpark), Partitioning, and Parquet Storage")
    st.markdown("---")

    # Workflow Visualizer
    st.markdown(
        """
        <div style="background-color: #121824; border: 1px solid #1e293b; border-radius: 6px; padding: 14px 18px; margin-bottom: 16px;">
            <small style="color: #94a3b8; font-weight: 600; text-transform: uppercase;">ETL Architecture Flow</small>
            <p style="font-family: monospace; font-size: 0.95rem; color: #38bdf8; margin: 6px 0 0 0;">
            Raw Ingestion &nbsp;→&nbsp; Schema Validation &nbsp;→&nbsp; Data Hygiene &nbsp;→&nbsp; 
            Distributed Window Transforms &nbsp;→&nbsp; Parquet Columnar Storage &nbsp;→&nbsp; Downstream Analytics
            </p>
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
        '<span class="badge-spark">Apache PySpark 3.5 (Distributed Local)</span>'
        if "PySpark" in spark_res["engine"]
        else '<span class="badge-pandas">Pandas Processing Engine</span>'
    )
    st.markdown(f"**Execution Runtime:** {engine_badge} &nbsp;|&nbsp; **Latency:** `{spark_res['duration_seconds']}s`", unsafe_allow_html=True)
    st.markdown("")

    # Audit Metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Input Records", f"{spark_res['raw_records']:,}")
    c2.metric("Output Records", f"{spark_res['processed_records']:,}")
    c3.metric("Missing Handled", f"{spark_res['missing_values_handled']:,}")
    c4.metric("Duplicates Removed", f"{spark_res['duplicates_removed']:,}")
    c5.metric("Outliers (IQR)", f"{spark_res['outliers_detected']:,}")

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
    st.title("Big Data Analytics")
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
                paper_bgcolor=PLOTLY_CARD_BG,
                plot_bgcolor=PLOTLY_BG,
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
                line=dict(color="#ef4444", width=1.5),
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
        # Quick comparative glance
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
                <div style="background-color: #121824; border: 1px solid #1e293b; border-radius: 6px; padding: 16px; margin: 12px 0;">
                    <h4 style="color: {pres_data['action_color']}; margin: 0 0 10px 0;">Signal: {pres_data['action']} · {pres_data['risk_tier']}</h4>
                    <p style="font-weight: 500; color: #cbd5e1; margin-bottom: 6px;">Quantitative Rationales:</p>
                    <ul style="color: #94a3b8; font-size: 0.92rem; margin: 0; padding-left: 20px;">
                        {"".join([f"<li>{r}</li>" for r in pres_data['supporting_reasons']])}
                    </ul>
                    <hr style="margin: 12px 0;">
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
    st.title("Stock Prediction")
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
        <div style="background-color: #121824; border: 1px solid #1e293b; border-left: 4px solid #38bdf8; border-radius: 6px; padding: 14px 18px; margin: 12px 0;">
            <span style="color: #94a3b8; font-size: 0.82rem; text-transform: uppercase; font-weight: 600;">{pred_horizon}-Day Ahead Out-of-Sample Price Forecast</span>
            <h3 style="margin: 4px 0 2px 0; color: #f8fafc;">{curr_sym}{ml_output['next_day_forecast']:,.2f} 
            <span style="font-size: 0.95rem; color: {'#22c55e' if ml_output['predicted_change'] >= 0 else '#ef4444'};">
            ({ml_output['predicted_change_pct']:+.2f}%)</span></h3>
            <small style="color: #64748b;">Current Close: {curr_sym}{ml_output['current_price']:,.2f} | Expected Delta: {curr_sym}{ml_output['predicted_change']:+.2f} | Window: {ml_output['test_period']}</small>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Actual vs Predicted Plotly Chart
    st.markdown('<p class="term-header">Actual vs. Predicted Price (Out-of-Sample Test Evaluation)</p>', unsafe_allow_html=True)
    viz = ml_output["viz_df"]

    fig_test = go.Figure()
    fig_test.add_trace(
        go.Scatter(
            x=viz.index,
            y=viz["Actual_Close"],
            mode="lines",
            name="Actual Close",
            line=dict(color="#38bdf8", width=1.8),
        )
    )
    fig_test.add_trace(
        go.Scatter(
            x=viz.index,
            y=viz["Predicted_Close"],
            mode="lines",
            name=f"Predicted ({chosen_algo})",
            line=dict(color="#fb923c", width=1.6, dash="dash"),
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
    fig_test.update_layout(**_plotly_layout(height=390))
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
        fig_fi.update_layout(paper_bgcolor=PLOTLY_CARD_BG, plot_bgcolor=PLOTLY_BG, height=270, coloraxis_showscale=False)
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
        fig_err.update_layout(paper_bgcolor=PLOTLY_CARD_BG, plot_bgcolor=PLOTLY_BG, height=270, showlegend=False)
        st.plotly_chart(fig_err, use_container_width=True)

    # Multi-Model Benchmark Comparison Table
    st.markdown('<p class="term-header">Comprehensive Model Benchmark Comparison</p>', unsafe_allow_html=True)
    comp_results = compare_all_models(t_ohlcv, horizon=pred_horizon, test_size=test_pct)
    st.dataframe(comp_results, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# PAGE F: PERFORMANCE BENCHMARK
# ══════════════════════════════════════════════════════════════
elif page == "Performance Benchmark":
    st.title("Performance Benchmark")
    st.caption("Empirical Runtime Profiling: In-Memory Pandas vs. Distributed Apache PySpark")
    st.markdown("---")

    st.markdown(
        """
        <div style="background-color: #121824; border: 1px solid #1e293b; border-radius: 6px; padding: 14px 18px; margin-bottom: 14px;">
            <small style="color: #38bdf8; font-weight: 600;">ACADEMIC BENCHMARK METHODOLOGY</small>
            <p style="color: #cbd5e1; font-size: 0.9rem; margin: 4px 0 0 0;">
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
            fig_b.update_layout(**_plotly_layout(height=380), barmode="group", yaxis_title="Seconds")
            st.plotly_chart(fig_b, use_container_width=True)

        st.markdown("##### Detailed Measurement Ledger")
        st.dataframe(b_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# PAGE G: PROJECT INFORMATION (ACADEMIC ABL SPECIFICATION)
# ══════════════════════════════════════════════════════════════
elif page == "Project Information":
    st.title("Project Information")
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
st.caption(
    "StockSense Financial Terminal · Academic Activity-Based Learning (ABL) Project · "
    "Apache Spark & Machine Learning Demonstration · For Educational Purposes Only."
)
