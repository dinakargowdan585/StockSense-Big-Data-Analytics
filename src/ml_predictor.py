"""Machine Learning prediction module for StockSense.

Implements strict chronological train/test split without lookahead bias,
historical feature generation, model training, evaluation, next-session price forecasting,
and empirical benchmarking against a Naive Persistence Baseline (Current Close Price).

Models compared:
0. Naive Baseline (Previous-Day / Origin Persistence Benchmark)
1. Linear Regression (Statistical Baseline with Scaler fit on Train only)
2. Random Forest Regressor (Ensemble Bagging)
3. XGBoost / Gradient Boosting Regressor (Boosting)
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Check for XGBoost
_XGBOOST_AVAILABLE = False
try:
    from xgboost import XGBRegressor
    _XGBOOST_AVAILABLE = True
except Exception:
    _XGBOOST_AVAILABLE = False


def create_ml_features(
    ohlcv_df: pd.DataFrame,
    horizon: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, list[str]]:
    """Generate predictive technical and lagged features from raw/processed OHLCV data.

    Strict chronological generation without lookahead data leakage.
    Target Alignment:
    - Horizon h represents the h-th subsequent trading session (e.g., h=1 is next session,
      h=5 is the 5th trading day ahead), excluding non-trading days.
    - Target: Close price shifted backwards by `horizon` rows: df[close_col].shift(-horizon).
    - Rows with missing targets (the trailing h rows) are excluded from training/testing,
      but the very last valid feature row (at forecast origin t_latest) is captured in
      clean_df.attrs["latest_features"] to project future predictions without leakage.
    """
    df = ohlcv_df.copy()

    # Ensure Datetime Index if Date column present
    for date_candidate in ["Date", "date", "timestamp", "Timestamp"]:
        if date_candidate in df.columns:
            df[date_candidate] = pd.to_datetime(df[date_candidate])
            df = df.set_index(date_candidate)
            break

    # Determine Close price column
    close_col = "Close"
    for candidate in ["Adj_Close", "Adj Close", "Close", "close"]:
        if candidate in df.columns:
            close_col = candidate
            break

    # 1. Price Returns & Lags (No backward-filling)
    if "Return_1d" not in df.columns and "Daily_Return" not in df.columns:
        df["Return_1d"] = df[close_col].pct_change()
    elif "Daily_Return" in df.columns and "Return_1d" not in df.columns:
        df["Return_1d"] = df["Daily_Return"]

    for lag in [1, 2, 3, 5]:
        col_name = f"Lag_{lag}"
        if col_name not in df.columns:
            df[col_name] = df[close_col].shift(lag)

    # 2. Moving Averages
    for window in [5, 10, 20, 50]:
        col_name = f"SMA_{window}"
        if col_name not in df.columns:
            df[col_name] = df[close_col].rolling(window, min_periods=1).mean()

    # 3. Rolling Volatility
    if "Vol_10" not in df.columns:
        df["Vol_10"] = df["Return_1d"].rolling(10, min_periods=1).std().fillna(0.0)
    if "Vol_20" not in df.columns:
        if "Volatility_20d" in df.columns:
            df["Vol_20"] = df["Volatility_20d"].fillna(0.0)
        else:
            df["Vol_20"] = df["Return_1d"].rolling(20, min_periods=1).std().fillna(0.0)

    # 4. RSI (14-day)
    if "RSI_14" not in df.columns:
        delta = df[close_col].diff()
        gain = delta.where(delta > 0, 0.0).rolling(14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14, min_periods=1).mean()
        rs = gain / loss.replace(0, np.nan)
        df["RSI_14"] = (100 - (100 / (1 + rs))).fillna(50.0)

    # 5. Volume Ratio
    if "Volume_Ratio" not in df.columns:
        vol_col = "Volume" if "Volume" in df.columns else ("volume" if "volume" in df.columns else None)
        if vol_col and vol_col in df.columns:
            vol_sma20 = df[vol_col].rolling(20, min_periods=1).mean().replace(0, np.nan)
            df["Volume_Ratio"] = (df[vol_col] / vol_sma20).fillna(1.0)
        else:
            df["Volume_Ratio"] = 1.0

    # 6. High-Low Spread
    if "HL_Spread" not in df.columns:
        high_col = "High" if "High" in df.columns else ("high" if "high" in df.columns else None)
        low_col = "Low" if "Low" in df.columns else ("low" if "low" in df.columns else None)
        if high_col and low_col and high_col in df.columns and low_col in df.columns:
            df["HL_Spread"] = ((df[high_col] - df[low_col]) / df[close_col]).fillna(0.0)
        else:
            df["HL_Spread"] = 0.0

    # Target variable: Close price shifted forward by horizon trading sessions
    df["Target_Close"] = df[close_col].shift(-horizon)

    feature_cols = [
        "Lag_1", "Lag_2", "Lag_3", "Lag_5",
        "SMA_5", "SMA_10", "SMA_20", "SMA_50",
        "Vol_10", "Vol_20", "RSI_14", "Volume_Ratio", "HL_Spread",
    ]

    # Capture the latest available observation with valid features (Forecast Origin)
    df_valid_features = df.dropna(subset=feature_cols).copy()
    if df_valid_features.empty:
        raise ValueError("No historical records remain after technical feature extraction.")

    latest_row = df_valid_features.iloc[[-1]]
    origin_date = latest_row.index[0]
    if not isinstance(origin_date, pd.Timestamp):
        origin_date = pd.to_datetime(origin_date)

    # Calculate target trading date: horizon business trading days forward
    target_dates = pd.bdate_range(start=origin_date, periods=horizon + 1)
    target_trading_date = target_dates[-1]

    origin_price = float(latest_row[close_col].iloc[0])
    latest_features = latest_row[feature_cols]

    # Eligible historical dataset for model training and out-of-sample testing
    clean_df = df.dropna(subset=feature_cols + ["Target_Close"]).copy()
    X = clean_df[feature_cols]
    y = clean_df["Target_Close"]

    # Store forecast context metadata in DataFrame attrs
    clean_df.attrs["forecast_origin_date"] = origin_date
    clean_df.attrs["target_trading_date"] = target_trading_date
    clean_df.attrs["origin_price"] = origin_price
    clean_df.attrs["latest_features"] = latest_features
    clean_df.attrs["close_col"] = close_col
    clean_df.attrs["horizon"] = horizon

    return clean_df, X, y, feature_cols


def train_and_evaluate_model(
    ohlcv_df: pd.DataFrame,
    model_name: str = "Random Forest",
    horizon: int = 1,
    test_size: float = 0.20,
) -> dict[str, Any]:
    """Train stock prediction model using chronological train/test split.

    Evaluates MAE, RMSE, R², and Directional Accuracy.
    Includes empirical comparison against Naive Persistence Baseline.
    """
    clean_df, X, y, feature_cols = create_ml_features(ohlcv_df, horizon=horizon)
    close_col = clean_df.attrs.get("close_col", "Close")

    if len(clean_df) < 50:
        raise ValueError(
            f"Insufficient historical data ({len(clean_df)} records). "
            "Please select a date range with at least 6 months of data."
        )

    # Chronological Split (Strictly NO SHUFFLE to prevent lookahead leakage)
    split_idx = int(len(clean_df) * (1.0 - test_size))

    train_df = clean_df.iloc[:split_idx]
    test_df = clean_df.iloc[split_idx:]

    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]

    # Select Model & Configure Pipeline (fit transformers on X_train only)
    model = None
    engine_name = model_name

    if model_name == "Linear Regression":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ])
        engine_name = "Linear Regression (Standardized)"
    elif model_name == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        engine_name = "Random Forest Regressor"
    elif model_name in ("XGBoost", "Gradient Boosting"):
        if _XGBOOST_AVAILABLE:
            model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
            engine_name = "XGBoost Regressor"
        else:
            model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
            engine_name = "Gradient Boosting Regressor"
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        engine_name = "Random Forest Regressor"

    # Fit Model strictly on training partition
    model.fit(X_train, y_train)

    # Predict on out-of-sample Test Partition
    y_pred = model.predict(X_test)

    # Compute Evaluation Metrics
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))

    # Directional Accuracy: Did predicted move match actual move from current price?
    actual_current_close = test_df[close_col].values
    actual_direction = np.sign(y_test.values - actual_current_close)
    predicted_direction = np.sign(y_pred - actual_current_close)
    directional_accuracy = float(np.mean(actual_direction == predicted_direction) * 100.0)

    # Empirical Naive Persistence Baseline (Price at origin t persists to t+h)
    naive_pred = actual_current_close
    baseline_mae = float(mean_absolute_error(y_test, naive_pred))
    baseline_rmse = float(np.sqrt(mean_squared_error(y_test, naive_pred)))
    baseline_r2 = float(r2_score(y_test, naive_pred))

    # Baseline Directional Accuracy (Dynamically calculated: Previous-session momentum persistence)
    prior_close = test_df["Lag_1"].values
    baseline_direction = np.sign(actual_current_close - prior_close)
    baseline_directional_acc = float(np.mean(actual_direction == baseline_direction) * 100.0)

    # Extract Feature Importance / Coefficients safely
    feature_importance = {}
    if hasattr(model, "feature_importances_"):
        for f, imp in zip(feature_cols, model.feature_importances_):
            feature_importance[f] = round(float(imp), 4)
    elif hasattr(model, "named_steps") and "regressor" in model.named_steps:
        reg = model.named_steps["regressor"]
        if hasattr(reg, "coef_"):
            for f, coef in zip(feature_cols, reg.coef_):
                feature_importance[f] = round(float(coef), 4)
    elif hasattr(model, "coef_"):
        for f, coef in zip(feature_cols, model.coef_):
            feature_importance[f] = round(float(coef), 4)

    sorted_features = dict(sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True))

    # Future Forecast using latest available historical state (Forecast Origin t_latest)
    origin_date = clean_df.attrs["forecast_origin_date"]
    target_trading_date = clean_df.attrs["target_trading_date"]
    origin_price = clean_df.attrs["origin_price"]
    latest_features = clean_df.attrs["latest_features"]

    future_pred = float(model.predict(latest_features)[0])
    predicted_change = future_pred - origin_price
    predicted_change_pct = (predicted_change / origin_price) * 100.0

    # Build Test Visualization DataFrame
    viz_df = pd.DataFrame({
        "Date": test_df.index,
        "Actual_Close": y_test.values,
        "Predicted_Close": y_pred,
        "Naive_Baseline": naive_pred,
        "Error": y_test.values - y_pred,
        "Current_Close": actual_current_close,
    }).set_index("Date")

    test_start_date = test_df.index[0].strftime("%b %d, %Y")
    test_end_date = test_df.index[-1].strftime("%b %d, %Y")

    return {
        "model_name": model_name,
        "engine_name": engine_name,
        "horizon_days": horizon,
        "forecast_origin_date": origin_date.strftime("%Y-%m-%d"),
        "target_trading_date": target_trading_date.strftime("%Y-%m-%d"),
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "test_period": f"{test_start_date} to {test_end_date}",
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "r2_score": round(r2, 4),
        "directional_accuracy_pct": round(directional_accuracy, 2),
        "baseline_mae": round(baseline_mae, 2),
        "baseline_rmse": round(baseline_rmse, 2),
        "baseline_r2": round(baseline_r2, 4),
        "baseline_directional_acc": round(baseline_directional_acc, 2),
        "current_price": round(origin_price, 2),
        "next_day_forecast": round(future_pred, 2),
        "forecast_price": round(future_pred, 2),
        "predicted_change": round(predicted_change, 2),
        "predicted_change_pct": round(predicted_change_pct, 2),
        "feature_importance": sorted_features,
        "viz_df": viz_df,
    }


def compare_all_models(
    ohlcv_df: pd.DataFrame,
    horizon: int = 1,
    test_size: float = 0.20,
) -> pd.DataFrame:
    """Train all models and benchmark against Naive Persistence Baseline."""
    suite = evaluate_models_suite(ohlcv_df, horizon=horizon, test_size=test_size)
    return suite["comparison_df"]


def evaluate_models_suite(
    ohlcv_df: pd.DataFrame,
    horizon: int = 1,
    test_size: float = 0.20,
) -> dict[str, Any]:
    """Train all models, compile side-by-side comparison, test-set predictions, and identify best model.

    Evaluates models on the out-of-sample chronological test set.
    Directly compares models against the Naive Baseline.
    """
    models = ["Linear Regression", "Random Forest", "Gradient Boosting"]
    results = []
    models_res = {}

    for m in models:
        try:
            res = train_and_evaluate_model(ohlcv_df, model_name=m, horizon=horizon, test_size=test_size)
            models_res[m] = res
        except Exception as exc:
            logger.warning(f"Error evaluating model {m}: {exc}")

    if not models_res:
        raise RuntimeError("No prediction models could be trained successfully.")

    sample_res = next(iter(models_res.values()))
    baseline_mae = sample_res["baseline_mae"]
    baseline_rmse = sample_res["baseline_rmse"]
    baseline_r2 = sample_res["baseline_r2"]
    baseline_dir_acc = sample_res["baseline_directional_acc"]
    current_price = sample_res["current_price"]

    # Baseline Entry in Scorecard (Dynamically Evaluated)
    results.append({
        "Model": "Naive Baseline (Previous-Day Persistence)",
        "Type": "Benchmark",
        "MAE ($)": baseline_mae,
        "RMSE ($)": baseline_rmse,
        "R² Score": baseline_r2,
        "Directional Accuracy (%)": f"{baseline_dir_acc:.1f}%",
        "Predicted Price ($)": f"${current_price:,.2f}",
        "Expected Move": "0.00%",
    })

    best_model_name = None
    best_mae = float("inf")

    for m, res in models_res.items():
        results.append({
            "Model": res["engine_name"],
            "Type": "Machine Learning",
            "MAE ($)": res["mae"],
            "RMSE ($)": res["rmse"],
            "R² Score": res["r2_score"],
            "Directional Accuracy (%)": f"{res['directional_accuracy_pct']:.1f}%",
            "Predicted Price ($)": f"${res['next_day_forecast']:,.2f}",
            "Expected Move": f"{res['predicted_change_pct']:+.2f}%",
        })
        if res["mae"] < best_mae:
            best_mae = res["mae"]
            best_model_name = m

    best_res = models_res[best_model_name]
    beats_baseline = bool(best_mae < baseline_mae)
    improvement_pct = round(((baseline_mae - best_mae) / baseline_mae) * 100.0, 2) if baseline_mae > 0 else 0.0

    combined_viz = pd.DataFrame(index=sample_res["viz_df"].index)
    combined_viz["Actual_Close"] = sample_res["viz_df"]["Actual_Close"]
    combined_viz["Naive_Baseline"] = sample_res["viz_df"]["Naive_Baseline"]

    for m, res in models_res.items():
        combined_viz[f"Pred_{m}"] = res["viz_df"]["Predicted_Close"]

    return {
        "comparison_df": pd.DataFrame(results),
        "models_res": models_res,
        "best_model_name": best_model_name,
        "best_model_res": best_res,
        "beats_baseline": beats_baseline,
        "improvement_pct": improvement_pct,
        "combined_viz_df": combined_viz,
        "forecast_origin_date": sample_res["forecast_origin_date"],
        "target_trading_date": sample_res["target_trading_date"],
        "current_price": current_price,
    }
