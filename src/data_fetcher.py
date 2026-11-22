"""Utilities for fetching, cleaning, and storing stock market data for StockSense.

Supports:
- US Market stocks (VOO, NVDA, AAPL, MSFT, etc.)
- Indian NSE stocks using Yahoo Finance tickers (RELIANCE.NS, TCS.NS, INFY.NS, etc.)
- Historical CSV and Parquet file loading
- Local offline datasets
- Streamlit file upload parsing
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import yfinance as yf

# Market Presets
US_PRESETS = ["VOO", "QQQM", "NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "SPY"]
NSE_PRESETS = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "TATAMOTORS.NS",
    "SBIN.NS",
    "ITC.NS",
]
ALL_DEFAULT_TICKERS = US_PRESETS[:4] + NSE_PRESETS[:2]


def _validate_tickers(tickers: list[str]) -> list[str]:
    """Validate and normalize a list of ticker symbols."""
    if not isinstance(tickers, list) or not tickers:
        raise ValueError("tickers must be a non-empty list of stock symbols.")

    cleaned: list[str] = []
    for ticker in tickers:
        if not isinstance(ticker, str) or not ticker.strip():
            raise ValueError("Each ticker must be a non-empty string.")
        normalized = ticker.strip().upper()
        if normalized not in cleaned:
            cleaned.append(normalized)

    return cleaned


def validate_ohlcv_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict[str, Any]]:
    """Validate standard financial OHLC relationships and data integrity.

    Integrity checks:
    - High >= Open and High >= Close
    - Low <= Open and Low <= Close
    - High >= Low
    - Volume >= 0
    - Chronological sorting

    Returns cleaned DataFrame and a validation audit dictionary.
    """
    if df is None or df.empty:
        raise ValueError("Cannot validate empty DataFrame.")

    data = df.copy()
    
    # Identify price columns defensively
    cols = {str(c).lower().replace(" ", "_"): c for c in data.columns}
    open_c = cols.get("open")
    high_c = cols.get("high")
    low_c = cols.get("low")
    close_c = cols.get("adj_close") or cols.get("close")
    vol_c = cols.get("volume")

    total_records = len(data)
    invalid_mask = pd.Series(False, index=data.index)

    # Validate OHLC geometric boundaries if columns are present
    if open_c and high_c and low_c and close_c:
        h_lt_o = data[high_c] < data[open_c]
        h_lt_c = data[high_c] < data[close_c]
        l_gt_o = data[low_c] > data[open_c]
        l_gt_c = data[low_c] > data[close_c]
        h_lt_l = data[high_c] < data[low_c]
        invalid_mask = h_lt_o | h_lt_c | l_gt_o | l_gt_c | h_lt_l

    if vol_c:
        neg_vol = data[vol_c] < 0
        invalid_mask = invalid_mask | neg_vol

    invalid_count = int(invalid_mask.sum())
    
    # Filter out invalid records if any violation detected
    if invalid_count > 0:
        cleaned_data = data[~invalid_mask].copy()
    else:
        cleaned_data = data

    audit_report = {
        "total_records": total_records,
        "valid_records": len(cleaned_data),
        "invalid_ohlcv_flagged": invalid_count,
        "treatment": "Excluded rows violating geometric OHLC bounds (High >= Open/Close, Low <= Open/Close, Volume >= 0)",
    }

    return cleaned_data, audit_report


def _clean_ohlcv_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Clean OHLCV data without backward filling (prevents lookahead data leakage)."""
    if raw_data is None or raw_data.empty:
        raise ValueError("No data returned from data source.")

    data = raw_data.copy()

    if isinstance(data.columns, pd.MultiIndex):
        data = data.sort_index(axis=1)
        ticker_level = data.columns.nlevels - 1
        tickers = data.columns.get_level_values(ticker_level).unique().tolist()
        failed_tickers: list[str] = []

        for ticker in tickers:
            ticker_slice = data.xs(ticker, axis=1, level=ticker_level, drop_level=False)
            if ticker_slice.isna().all().all():
                failed_tickers.append(str(ticker))

        if failed_tickers:
            data = data.drop(columns=failed_tickers, level=ticker_level)

    # Strictly preserve chronological ordering
    data = data.sort_index()
    data = data.dropna(how="all")
    
    # Forward-fill only within historical sequence; strictly NO backward-fill to prevent future leakage
    data = data.ffill()
    data = data.dropna(how="any")

    if data.empty:
        raise ValueError(
            "Fetched data contains only missing values after cleaning. "
            "Try a different period or ticker set."
        )

    # Validate OHLC bounds
    validated, _ = validate_ohlcv_data(data)
    return validated


def get_stock_data(tickers: list[str], period: str = "1y") -> pd.DataFrame:
    """Download historical OHLCV data for multiple ticker symbols from Yahoo Finance."""
    cleaned_tickers = _validate_tickers(tickers)

    if not isinstance(period, str) or not period.strip():
        raise ValueError("period must be a non-empty string, e.g. '1y' or '5y'.")

    try:
        data = yf.download(
            tickers=cleaned_tickers,
            period=period.strip(),
            auto_adjust=False,
            progress=False,
            group_by="column",
            threads=False,
        )
    except Exception as exc:
        message = (
            f"Failed to download stock data for tickers {cleaned_tickers} "
            f"with period '{period}': {exc}"
        )
        raise RuntimeError(message) from exc

    try:
        return _clean_ohlcv_data(data)
    except ValueError as exc:
        raise ValueError(
            f"Unable to prepare stock data for {cleaned_tickers}: {exc}"
        ) from exc


def get_stock_info(ticker: str) -> dict[str, Any]:
    """Fetch key company metadata for a single ticker."""
    if not isinstance(ticker, str) or not ticker.strip():
        raise ValueError("ticker must be a non-empty string.")

    normalized = ticker.strip().upper()

    try:
        stock = yf.Ticker(normalized)
        info = stock.info or {}
    except Exception as exc:
        raise RuntimeError(
            f"Failed to fetch stock info for ticker '{normalized}': {exc}"
        ) from exc

    return {
        "ticker": normalized,
        "company_name": info.get("longName") or info.get("shortName") or normalized,
        "sector": info.get("sector", "N/A"),
        "market_cap": info.get("marketCap"),
        "pe_ratio": info.get("trailingPE"),
        "currency": info.get("currency", "USD"),
    }


# ─────────────────────────────────────────────────────────────
# OFFLINE & LOCAL DATASET HANDLING
# ─────────────────────────────────────────────────────────────

def get_sample_data_dir() -> Path:
    """Return path to data/sample directory, creating it if necessary."""
    project_root = Path(__file__).resolve().parent.parent
    sample_dir = project_root / "data" / "sample"
    sample_dir.mkdir(parents=True, exist_ok=True)
    return sample_dir


def get_processed_data_dir() -> Path:
    """Return path to data/processed directory, creating it if necessary."""
    project_root = Path(__file__).resolve().parent.parent
    proc_dir = project_root / "data" / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)
    return proc_dir


def load_local_dataset(filepath: str | Path) -> pd.DataFrame:
    """Load and normalize a local CSV or Parquet stock dataset."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}. Must be CSV or Parquet.")

    # Parse date column
    date_candidates = [c for c in df.columns if "date" in str(c).lower()]
    if date_candidates:
        date_col = date_candidates[0]
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
    elif not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass

    return _clean_ohlcv_data(df)


def parse_uploaded_file(uploaded_file: Any) -> Tuple[pd.DataFrame, str]:
    """Parse a user-uploaded CSV or Parquet file in Streamlit."""
    file_name = uploaded_file.name
    if file_name.endswith(".parquet"):
        df = pd.read_parquet(uploaded_file)
    elif file_name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        raise ValueError("Uploaded file must be a .csv or .parquet file.")

    date_candidates = [c for c in df.columns if "date" in str(c).lower()]
    if date_candidates:
        date_col = date_candidates[0]
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)

    ticker = Path(file_name).stem.upper()
    cleaned = _clean_ohlcv_data(df)
    return cleaned, ticker


def list_sample_datasets() -> list[dict[str, str]]:
    """Enumerate pre-installed sample datasets in data/sample/."""
    sample_dir = get_sample_data_dir()
    files = list(sample_dir.glob("*.parquet")) + list(sample_dir.glob("*.csv"))
    datasets = []
    for f in sorted(files):
        datasets.append({
            "filename": f.name,
            "ticker": f.stem.upper(),
            "format": f.suffix.replace(".", "").upper(),
            "path": str(f.resolve()),
            "size_kb": f"{f.stat().st_size / 1024:.1f} KB",
        })
    return datasets
