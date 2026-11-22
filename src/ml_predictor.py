"""Machine Learning prediction module for StockSense.

Implements strict chronological train/test split without lookahead bias,
historical feature generation, model training, evaluation, next-day price forecasting,
and academic comparison against a Naive Baseline (Previous-Day Price).
Models compared:
0. Naive Baseline (Previous-Day Persistence Benchmark)
1. Linear Regression (Statistical Baseline)
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

logger = logging.getLogger(__name__)

# Check for XGBoost
_XGBOOST_AVAILABLE = False
try:
    from xgboost import XGBRegressor
    _XGBOOST_AVAILABLE = True
except Exception:
    _XGBOOST_AVAILABLE = False


def create_ml_features(ohlcv_df: pd.DataFrame, horizon: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, list[str]]:
    """Generate predictive technical and lagged features from raw OHLCV data.

    Strict chronological generation without lookahead data leakage.
    Target: Future Close Price shifted backwards by `horizon` days.
    """
    df = ohlcv_df.copy()
    close_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    
    # 1. Price Returns & Lags
    df["Return_1d"] = df[close_col].pct_change()
    df["Lag_1"] = df[close_col].shift(1)
    df["Lag_2"] = df[close_col].shift(2)
    df["Lag_3"] = df[close_col].shift(3)
    df["Lag_5"] = df[close_col].shift(5)

    # 2. Moving Averages
    df["SMA_5"] = df[close_col].rolling(5).mean()
    df["SMA_10"] = df[close_col].rolling(10).mean()
    df["SMA_20"] = df[close_col].rolling(20).mean()
    df["SMA_50"] = df[close_col].rolling(50).mean()

    # 3. Rolling Volatility
    df["Vol_10"] = df["Return_1d"].rolling(10).std()
    df["Vol_20"] = df["Return_1d"].rolling(20).std()

    # 4. RSI (14-day)
    delta = df[close_col].diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # 5. Volume Ratio
    if "Volume" in df.columns:
        vol_sma20 = df["Volume"].rolling(20).mean().replace(0, np.nan)
        df["Volume_Ratio"] = df["Volume"] / vol_sma20
    else:
        df["Volume_Ratio"] = 1.0

    # 6. High-Low Spread
    if "High" in df.columns and "Low" in df.columns:
        df["HL_Spread"] = (df["High"] - df["Low"]) / df[close_col]
    else:
        df["HL_Spread"] = 0.0

    # Target variable: Close price shifted by horizon
    df["Target_Close"] = df[close_col].shift(-horizon)

    feature_cols = [
        "Lag_1", "Lag_2", "Lag_3", "Lag_5",
        "SMA_5", "SMA_10", "SMA_20", "SMA_50",
        "Vol_10", "Vol_20", "RSI_14", "Volume_Ratio", "HL_Spread",
    ]

    clean_df = df.dropna(subset=feature_cols + ["Target_Close"]).copy()
    X = clean_df[feature_cols]
    y = clean_df["Target_Close"]

    return clean_df, X, y, feature_cols


def train_and_evaluate_model(
    ohlcv_df: pd.DataFrame,
    model_name: str = "Random Forest",
    horizon: int = 1,
    test_size: float = 0.20,
) -> dict[str, Any]:
    """Train stock prediction model using chronological train/test split.

    Evaluates MAE, RMSE, R², and Directional Accuracy.
    Includes comparison against Naive Persistence Baseline.
    """
    close_col = "Adj Close" if "Adj Close" in ohlcv_df.columns else "Close"
    clean_df, X, y, feature_cols = create_ml_features(ohlcv_df, horizon=horizon)

    if len(clean_df) < 50:
        raise ValueError(
            f"Insufficient historical data ({len(clean_df)} records). "
            "Please select a date range with at least 6 months of data."
        )

    # Chronological Split (Strictly NO SHUFFLE)
    split_idx = int(len(clean_df) * (1.0 - test_size))
    
    train_df = clean_df.iloc[:split_idx]
    test_df = clean_df.iloc[split_idx:]

    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]

    # Select Model
    model = None
    engine_name = model_name

    if model_name == "Linear Regression":
        model = LinearRegression()
    elif model_name == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
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

    # Fit Model
    model.fit(X_train, y_train)

    # Predict on Test Set
    y_pred = model.predict(X_test)

    # Compute Evaluation Metrics
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))

    # Directional Accuracy
    actual_current_close = test_df[close_col].values
    actual_direction = np.sign(y_test.values - actual_current_close)
    predicted_direction = np.sign(y_pred - actual_current_close)
    directional_accuracy = float(np.mean(actual_direction == predicted_direction) * 100.0)

    # Compute Naive Baseline (Previous-Day Price)
    naive_pred = actual_current_close
    baseline_mae = float(mean_absolute_error(y_test, naive_pred))
    baseline_rmse = float(np.sqrt(mean_squared_error(y_test, naive_pred)))
    baseline_r2 = float(r2_score(y_test, naive_pred))
    baseline_directional_acc = 50.0

    # Feature Importance (if tree-based) or Coefficients
    feature_importance = {}
    if hasattr(model, "feature_importances_"):
        for f, imp in zip(feature_cols, model.feature_importances_):
            feature_importance[f] = round(float(imp), 4)
    elif hasattr(model, "coef_"):
        for f, coef in zip(feature_cols, model.coef_):
            feature_importance[f] = round(float(coef), 4)

    sorted_features = dict(sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True))

    # Next-Day Forecast using latest available record
    latest_features = X.iloc[[-1]]
    next_day_pred = float(model.predict(latest_features)[0])
    current_latest_close = float(clean_df[close_col].iloc[-1])
    predicted_change = next_day_pred - current_latest_close
    predicted_change_pct = (predicted_change / current_latest_close) * 100.0

    # Build Visualization DataFrame
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
        "current_price": round(current_latest_close, 2),
        "next_day_forecast": round(next_day_pred, 2),
        "predicted_change": round(predicted_change, 2),
        "predicted_change_pct": round(predicted_change_pct, 2),
        "feature_importance": sorted_features,
        "viz_df": viz_df,
    }


def compare_all_models(ohlcv_df: pd.DataFrame, horizon: int = 1, test_size: float = 0.20) -> pd.DataFrame:
    """Train all models and benchmark against Naive Persistence Baseline."""
    models = ["Linear Regression", "Random Forest", "Gradient Boosting"]
    results = []

    # First run to get baseline metrics
    base_res = train_and_evaluate_model(ohlcv_df, model_name="Random Forest", horizon=horizon, test_size=test_size)
    results.append({
        "Model": "Naive Baseline (Previous-Day Price)",
        "Type": "Benchmark",
        "MAE ($)": base_res["baseline_mae"],
        "RMSE ($)": base_res["baseline_rmse"],
        "R² Score": base_res["baseline_r2"],
        "Directional Accuracy (%)": "50.0%",
        "Predicted Price ($)": f"${base_res['current_price']:,.2f}",
        "Expected Move": "0.00%",
    })

    for m in models:
        try:
            res = train_and_evaluate_model(ohlcv_df, model_name=m, horizon=horizon, test_size=test_size)
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
        except Exception as exc:
            logger.warning(f"Error evaluating model {m}: {exc}")
            continue

    return pd.DataFrame(results)
