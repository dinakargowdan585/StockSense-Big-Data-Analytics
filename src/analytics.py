"""Big Data Analytics module for StockSense.

Implements the 4 Pillars of Financial Analytics:
1. Descriptive Analytics: Summaries, distributions, averages, volumes, top performers.
2. Diagnostic Analytics: Volatility, correlation, drawdown, VaR/CVaR, anomaly detection.
3. Predictive Analytics: Coordinated via ml_predictor.py.
4. Prescriptive Analytics: Rule-based technical signals, risk tiers, quantitative postures.
"""

from __future__ import annotations

from math import sqrt
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats


def _extract_ticker_series(df: pd.DataFrame, ticker: str, field: str = "Close") -> pd.Series:
    """Extract a 1D Series for a specific ticker and field from single or multi-index df."""
    if isinstance(df.columns, pd.MultiIndex):
        fields = df.columns.get_level_values(0)
        tickers = df.columns.get_level_values(1)
        
        target_field = field
        if field == "Close" and "Adj Close" in fields:
            target_field = "Adj Close"

        if (target_field, ticker) in df.columns:
            return df[(target_field, ticker)].dropna()
        if (field, ticker) in df.columns:
            return df[(field, ticker)].dropna()
        if (ticker, target_field) in df.columns:
            return df[(ticker, target_field)].dropna()
        if ticker in tickers:
            slice_df = df.xs(ticker, axis=1, level=1)
            f_col = "Adj Close" if "Adj Close" in slice_df.columns else "Close"
            return slice_df[f_col].dropna()

    if field in df.columns:
        return df[field].dropna()
    if "Adj Close" in df.columns:
        return df["Adj Close"].dropna()
    if "Close" in df.columns:
        return df["Close"].dropna()
    
    return df.iloc[:, 0].dropna()


def _extract_ticker_ohlcv(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Extract flat OHLCV DataFrame for a single ticker."""
    if isinstance(df.columns, pd.MultiIndex):
        if ticker in df.columns.get_level_values(1):
            sub = df.xs(ticker, axis=1, level=1).copy()
            return sub
        elif ticker in df.columns.get_level_values(0):
            return df[ticker].copy()
    return df.copy()


# ─────────────────────────────────────────────────────────────
# 1. DESCRIPTIVE ANALYTICS
# ─────────────────────────────────────────────────────────────

def compute_descriptive_stats(df: pd.DataFrame, ticker: str) -> dict[str, Any]:
    """Compute comprehensive descriptive statistical metrics for a stock."""
    ohlcv = _extract_ticker_ohlcv(df, ticker)
    close_col = "Adj Close" if "Adj Close" in ohlcv.columns else "Close"
    prices = ohlcv[close_col].dropna()
    daily_returns = prices.pct_change().dropna()
    volume = ohlcv["Volume"].dropna() if "Volume" in ohlcv.columns else pd.Series(dtype=float)

    # Return series metrics
    mean_return = float(daily_returns.mean()) if not daily_returns.empty else 0.0
    std_return = float(daily_returns.std()) if not daily_returns.empty else 0.0
    skewness = float(stats.skew(daily_returns.values)) if len(daily_returns) > 3 else 0.0
    kurt = float(stats.kurtosis(daily_returns.values)) if len(daily_returns) > 3 else 0.0

    # Monthly return aggregates
    monthly_returns = prices.resample("ME").last().pct_change().dropna()

    # Price milestones
    current_price = float(prices.iloc[-1]) if not prices.empty else 0.0
    min_price = float(prices.min()) if not prices.empty else 0.0
    max_price = float(prices.max()) if not prices.empty else 0.0
    mean_price = float(prices.mean()) if not prices.empty else 0.0
    median_price = float(prices.median()) if not prices.empty else 0.0

    return {
        "ticker": ticker,
        "sample_size": len(prices),
        "current_price": current_price,
        "mean_price": round(mean_price, 2),
        "median_price": round(median_price, 2),
        "min_price": round(min_price, 2),
        "max_price": round(max_price, 2),
        "price_std": round(float(prices.std()), 2) if not prices.empty else 0.0,
        "daily_mean_return_pct": round(mean_return * 100, 3),
        "daily_volatility_pct": round(std_return * 100, 3),
        "annualized_return_pct": round(mean_return * 252 * 100, 2),
        "annualized_volatility_pct": round(std_return * sqrt(252) * 100, 2),
        "skewness": round(skewness, 3),
        "kurtosis": round(kurt, 3),
        "avg_daily_volume": int(volume.mean()) if not volume.empty else 0,
        "max_volume": int(volume.max()) if not volume.empty else 0,
        "daily_returns_series": daily_returns,
        "monthly_returns_series": monthly_returns,
        "prices_series": prices,
    }


def compute_market_performance_rankings(df: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Rank stocks across key financial performance metrics."""
    rows = []
    for t in tickers:
        try:
            s = _extract_ticker_series(df, t)
            if s.empty:
                continue
            dr = s.pct_change().dropna()
            total_ret = ((s.iloc[-1] / s.iloc[0]) - 1.0) * 100
            ann_ret = dr.mean() * 252 * 100
            ann_vol = dr.std() * sqrt(252) * 100
            sharpe = (dr.mean() * 252 - 0.05) / (dr.std() * sqrt(252)) if ann_vol > 0 else 0.0

            rows.append({
                "Ticker": t,
                "Current Price": f"${s.iloc[-1]:,.2f}" if ".NS" not in t else f"₹{s.iloc[-1]:,.2f}",
                "Period Return (%)": round(total_ret, 2),
                "Ann. Return (%)": round(ann_ret, 2),
                "Ann. Volatility (%)": round(ann_vol, 2),
                "Sharpe Ratio": round(sharpe, 2),
            })
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    rank_df = pd.DataFrame(rows)
    rank_df["Return Rank"] = rank_df["Period Return (%)"].rank(ascending=False).astype(int)
    rank_df["Risk Rank"] = rank_df["Ann. Volatility (%)"].rank(ascending=True).astype(int)
    return rank_df.sort_values(by="Period Return (%)", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# 2. DIAGNOSTIC ANALYTICS
# ─────────────────────────────────────────────────────────────

def compute_diagnostic_analytics(
    df: pd.DataFrame,
    ticker: str,
    benchmark_ticker: Optional[str] = "SPY",
) -> dict[str, Any]:
    """Perform diagnostic risk analytics including drawdown, VaR, anomalies, and beta."""
    ohlcv = _extract_ticker_ohlcv(df, ticker)
    close_col = "Adj Close" if "Adj Close" in ohlcv.columns else "Close"
    prices = ohlcv[close_col].dropna()
    daily_returns = prices.pct_change().dropna()
    volume = ohlcv["Volume"].dropna() if "Volume" in ohlcv.columns else pd.Series(dtype=float)

    # 1. Maximum Drawdown Calculation
    cum_returns = (1 + daily_returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    max_drawdown = float(drawdown.min()) * 100 if not drawdown.empty else 0.0

    # 2. Value-at-Risk (VaR) and Conditional VaR (CVaR) - Historical Method
    var_95 = float(np.percentile(daily_returns, 5)) * 100 if len(daily_returns) > 10 else 0.0
    var_99 = float(np.percentile(daily_returns, 1)) * 100 if len(daily_returns) > 10 else 0.0
    tail_losses = daily_returns[daily_returns <= np.percentile(daily_returns, 5)] if len(daily_returns) > 10 else pd.Series()
    cvar_95 = float(tail_losses.mean()) * 100 if not tail_losses.empty else var_95

    # 3. Downside Deviation & Sortino Ratio
    negative_returns = daily_returns[daily_returns < 0]
    downside_vol = float(negative_returns.std() * sqrt(252) * 100) if len(negative_returns) > 0 else 0.0
    excess_return = (daily_returns.mean() * 252 - 0.05) * 100 if not daily_returns.empty else 0.0
    sortino_ratio = (excess_return / downside_vol) if downside_vol > 0 else 0.0

    # 4. Anomaly Detection
    volume_anomalies = []
    if len(volume) > 20 and volume.std() > 0:
        vol_z = (volume - volume.mean()) / volume.std()
        spike_indices = vol_z[vol_z > 2.5].index
        for idx in spike_indices:
            volume_anomalies.append({
                "Date": idx.strftime("%Y-%m-%d"),
                "Volume": int(volume.loc[idx]),
                "Z-Score": round(float(vol_z.loc[idx]), 2),
                "Price": round(float(prices.loc[idx]), 2) if idx in prices.index else None,
            })

    return_anomalies = []
    if len(daily_returns) > 20 and daily_returns.std() > 0:
        ret_z = (daily_returns - daily_returns.mean()) / daily_returns.std()
        shock_indices = ret_z[ret_z.abs() > 3.0].index
        for idx in shock_indices:
            return_anomalies.append({
                "Date": idx.strftime("%Y-%m-%d"),
                "Return (%)": round(float(daily_returns.loc[idx] * 100), 2),
                "Z-Score": round(float(ret_z.loc[idx]), 2),
            })

    # 5. Beta against Benchmark
    beta = None
    if benchmark_ticker and benchmark_ticker in df.columns.get_level_values(1 if isinstance(df.columns, pd.MultiIndex) else 0):
        try:
            bench_prices = _extract_ticker_series(df, benchmark_ticker)
            bench_returns = bench_prices.pct_change().dropna()
            combined = pd.concat([daily_returns, bench_returns], axis=1).dropna()
            cov = np.cov(combined.iloc[:, 0], combined.iloc[:, 1])[0, 1]
            bench_var = np.var(combined.iloc[:, 1])
            if bench_var > 0:
                beta = round(float(cov / bench_var), 2)
        except Exception:
            beta = None

    return {
        "ticker": ticker,
        "max_drawdown_pct": round(max_drawdown, 2),
        "drawdown_series": drawdown,
        "var_95_pct": round(var_95, 2),
        "var_99_pct": round(var_99, 2),
        "cvar_95_pct": round(cvar_95, 2),
        "downside_volatility_pct": round(downside_vol, 2),
        "sortino_ratio": round(sortino_ratio, 2),
        "beta": beta,
        "volume_anomalies": volume_anomalies,
        "return_anomalies": return_anomalies,
        "volume_anomalies_count": len(volume_anomalies),
        "return_anomalies_count": len(return_anomalies),
    }


# ─────────────────────────────────────────────────────────────
# 3. PRESCRIPTIVE ANALYTICS
# ─────────────────────────────────────────────────────────────

def generate_prescriptive_signals(df: pd.DataFrame, ticker: str) -> dict[str, Any]:
    """Generate rule-based technical signals, risk tiers, and educational insights."""
    ohlcv = _extract_ticker_ohlcv(df, ticker)
    close_col = "Adj Close" if "Adj Close" in ohlcv.columns else "Close"
    prices = ohlcv[close_col].dropna()

    if len(prices) < 50:
        return {
            "ticker": ticker,
            "status": "Insufficient historical data (minimum 50 trading days required for technical rules).",
        }

    sma_20 = prices.rolling(20).mean()
    sma_50 = prices.rolling(50).mean()
    current_price = float(prices.iloc[-1])
    current_sma_20 = float(sma_20.iloc[-1])
    current_sma_50 = float(sma_50.iloc[-1])

    # 1. Trend Signal (Moving Average Cross)
    trend_signal = "Neutral"
    if current_sma_20 > current_sma_50 and current_price > current_sma_20:
        trend_signal = "Bullish"
        trend_reason = f"Golden Cross alignment: SMA 20 (${current_sma_20:,.2f}) > SMA 50 (${current_sma_50:,.2f}) with price trading above SMA 20."
    elif current_sma_20 < current_sma_50 and current_price < current_sma_20:
        trend_signal = "Bearish"
        trend_reason = f"Death Cross alignment: SMA 20 (${current_sma_20:,.2f}) < SMA 50 (${current_sma_50:,.2f}) with price trading below SMA 20."
    else:
        trend_reason = f"Mixed trend: Price is consolidating near moving averages (SMA 20: ${current_sma_20:,.2f}, SMA 50: ${current_sma_50:,.2f})."

    # 2. RSI (14-day)
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    current_rsi = float(rsi.dropna().iloc[-1]) if not rsi.dropna().empty else 50.0

    if current_rsi > 70:
        rsi_signal = "Overbought"
        rsi_reason = f"RSI 14 is {current_rsi:.1f} (>70), indicating short-term momentum exhaustion and increased pullback probability."
    elif current_rsi < 30:
        rsi_signal = "Oversold"
        rsi_reason = f"RSI 14 is {current_rsi:.1f} (<30), indicating potential oversold mean-reversion rebound opportunity."
    else:
        rsi_signal = "Neutral"
        rsi_reason = f"RSI 14 is {current_rsi:.1f} (in normal healthy 30-70 range)."

    # 3. Bollinger Bands (20-day, 2 std)
    rolling_std = prices.rolling(20).std()
    upper_band = sma_20 + (2 * rolling_std)
    lower_band = sma_20 - (2 * rolling_std)
    current_upper = float(upper_band.iloc[-1])
    current_lower = float(lower_band.iloc[-1])

    if current_price >= current_upper:
        bb_signal = "Upper Band Breach"
        bb_reason = f"Price (${current_price:,.2f}) is testing or above the upper Bollinger Band (${current_upper:,.2f})."
    elif current_price <= current_lower:
        bb_signal = "Lower Band Breach"
        bb_reason = f"Price (${current_price:,.2f}) is testing or below the lower Bollinger Band (${current_lower:,.2f})."
    else:
        bb_signal = "Within Bands"
        bb_reason = f"Price is trading normally within Bollinger Bands (${current_lower:,.2f} – ${current_upper:,.2f})."

    # 4. Volatility Risk Classification Tier
    dr = prices.pct_change().dropna()
    ann_vol = float(dr.std() * sqrt(252) * 100) if not dr.empty else 0.0
    if ann_vol < 18.0:
        risk_tier = "Low Risk (Conservative)"
        risk_color = "#22c55e"
    elif ann_vol <= 32.0:
        risk_tier = "Moderate Risk (Balanced)"
        risk_color = "#eab308"
    else:
        risk_tier = "High Risk (Aggressive)"
        risk_color = "#ef4444"

    # 5. Composite Quantitative Rule: Educational Buy / Hold / Avoid
    score = 0
    if trend_signal == "Bullish":
        score += 1
    elif trend_signal == "Bearish":
        score -= 1

    if rsi_signal == "Oversold":
        score += 0.5
    elif rsi_signal == "Overbought":
        score -= 0.5

    reasons = [trend_reason, rsi_reason, bb_reason, f"Annualized volatility is {ann_vol:.1f}% ({risk_tier})."]

    if score >= 1.0:
        action = "BUY (Accumulate)"
        action_color = "#22c55e"
    elif score <= -1.0:
        action = "AVOID (Defensive / Reduce)"
        action_color = "#ef4444"
    else:
        action = "HOLD (Neutral / Range-Bound)"
        action_color = "#eab308"

    return {
        "ticker": ticker,
        "current_price": current_price,
        "action": action,
        "action_color": action_color,
        "trend_signal": trend_signal,
        "rsi_value": round(current_rsi, 1),
        "rsi_signal": rsi_signal,
        "sma_20": round(current_sma_20, 2),
        "sma_50": round(current_sma_50, 2),
        "bb_upper": round(current_upper, 2),
        "bb_lower": round(current_lower, 2),
        "bb_signal": bb_signal,
        "annualized_volatility_pct": round(ann_vol, 2),
        "risk_tier": risk_tier,
        "risk_color": risk_color,
        "supporting_reasons": reasons,
    }
