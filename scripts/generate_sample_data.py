"""Utility script to generate offline sample datasets in data/sample/.

Fetches real historical market data for sample US and Indian NSE tickers
and saves them as both CSV and Parquet files for offline dashboard operation.
"""

from pathlib import Path
import yfinance as yf
import pandas as pd

SAMPLE_TICKERS = ["AAPL", "NVDA", "MSFT", "RELIANCE.NS", "TCS.NS"]

def main():
    root_dir = Path(__file__).resolve().parent.parent
    sample_dir = root_dir / "data" / "sample"
    sample_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating offline sample datasets into {sample_dir}...")

    for ticker in SAMPLE_TICKERS:
        try:
            print(f"Fetching {ticker}...")
            df = yf.download(ticker, period="1y", auto_adjust=False, progress=False)
            if df.empty:
                print(f"Warning: {ticker} returned empty data.")
                continue

            # Flatten MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]

            # Clean and ensure proper Date index
            df = df.dropna(how="all").ffill().bfill().dropna()
            
            clean_ticker = ticker.replace(".", "_")
            csv_path = sample_dir / f"{clean_ticker}.csv"
            parquet_path = sample_dir / f"{clean_ticker}.parquet"

            df.to_csv(csv_path)
            df.to_parquet(parquet_path)
            print(f"Saved {ticker} -> {csv_path.name} ({len(df)} rows)")
        except Exception as exc:
            print(f"Error fetching {ticker}: {exc}")

    print("Sample generation complete.")

if __name__ == "__main__":
    main()
