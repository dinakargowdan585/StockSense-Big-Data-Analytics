"""End-to-end verification and integration test suite for StockSense.

Tests the complete stock prediction workflow:
1. Stock Selection (NVDA, RELIANCE_NS)
2. Historical Data Ingestion (Offline & Live support)
3. Big Data PySpark ETL Pipeline Execution
4. Feature Engineering (Lags, Moving Averages, Volatility, Outlier treatment)
5. Snappy Parquet Storage
6. Machine Learning Model Training (Linear Regression, Random Forest, Gradient Boosting)
7. Chronological Evaluation against Naive Baseline
8. Future Price Forecasting & Directional Delta Calculation
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import pandas as pd
from src.data_fetcher import (
    get_processed_data_dir,
    get_sample_data_dir,
    list_sample_datasets,
    load_local_dataset,
)
from src.ml_predictor import (
    compare_all_models,
    create_ml_features,
    evaluate_models_suite,
    train_and_evaluate_model,
)
from src.spark_engine import (
    get_spark_status,
    process_stock_data_spark,
)


def run_tests() -> None:
    print("=" * 65)
    print("STOCKSENSE: BIG DATA PREDICTION SYSTEM TEST SUITE")
    print("=" * 65)

    # 1. Check Spark Status
    print("\n[Stage 1] Checking Spark Environment...")
    status = get_spark_status()
    print("Spark status:", status)
    assert status["pyspark_installed"], "PySpark 3.5 should be installed."
    print("✓ Apache PySpark environment verified.")

    # 2. Check Offline Datasets
    print("\n[Stage 2] Discovering Offline Datasets...")
    samples = list_sample_datasets()
    print(f"Discovered {len(samples)} offline sample datasets.")
    assert len(samples) >= 4, "Should have at least 4 sample datasets."
    sample_tickers = [s["ticker"] for s in samples]
    print(f"Available tickers: {sample_tickers}")
    assert any("NVDA" in t for t in sample_tickers), "NVDA dataset must be present."
    assert any("RELIANCE" in t for t in sample_tickers), "RELIANCE dataset must be present."
    print("✓ Offline datasets verified.")

    # 3. Test Full Prediction Workflow on NVIDIA (NVDA)
    print("\n[Stage 3] Testing End-to-End Prediction Workflow on NVIDIA (NVDA)...")
    nvda_path = get_sample_data_dir() / "NVDA.parquet"
    if not nvda_path.exists():
        nvda_path = get_sample_data_dir() / "NVDA.csv"
    assert nvda_path.exists(), f"NVDA dataset not found at {nvda_path}"

    df_nvda = load_local_dataset(nvda_path)
    print(f"1. Ingestion: Loaded NVDA ({df_nvda.shape[0]} rows, {df_nvda.shape[1]} columns)")
    assert not df_nvda.empty

    proc_dir = str(get_processed_data_dir())
    spark_nvda = process_stock_data_spark(df_nvda, ticker="NVDA", output_dir=proc_dir)
    print(f"2. PySpark Processing Engine: {spark_nvda['engine']}")
    print(f"3. Records Processed: {spark_nvda['processed_records']} (Outliers Flagged: {spark_nvda['outliers_detected']})")
    print(f"4. Generated Features ({len(spark_nvda['generated_features'])}): {spark_nvda['generated_features']}")
    print(f"5. Parquet Serialized Path: {spark_nvda['output_parquet']}")
    
    assert Path(spark_nvda["output_parquet"]).exists(), "Parquet output file must exist on disk."
    assert "Lag_1" in spark_nvda["generated_features"]
    assert "SMA_20" in spark_nvda["generated_features"]
    assert "Outlier_Flag" in spark_nvda["generated_features"]

    # 6. Train Models on Processed Dataset
    processed_nvda = spark_nvda["processed_df"]
    rf_nvda = train_and_evaluate_model(processed_nvda, model_name="Random Forest", horizon=1)
    print(f"6. Random Forest Model: Test MAE=${rf_nvda['mae']}, Dir. Accuracy={rf_nvda['directional_accuracy_pct']}%")
    print(f"7. Forecast: Current=${rf_nvda['current_price']}, Forecast=${rf_nvda['next_day_forecast']} (Move: {rf_nvda['predicted_change_pct']:+.2f}%)")
    assert rf_nvda["next_day_forecast"] > 0

    suite_nvda = evaluate_models_suite(processed_nvda, horizon=1)
    print("8. Multi-Model Scorecard:")
    print(suite_nvda["comparison_df"][["Model", "Type", "MAE ($)", "RMSE ($)", "R² Score", "Directional Accuracy (%)"]])
    print(f"9. Best Model on Test Set: {suite_nvda['best_model_name']} (Beats Baseline: {suite_nvda['beats_baseline']})")
    print("✓ NVIDIA prediction workflow passed successfully.")

    # 4. Repeat Full Workflow on Indian NSE Equity: Reliance Industries (RELIANCE.NS)
    print("\n[Stage 4] Testing Workflow on Indian NSE Equity: Reliance Industries (RELIANCE.NS)...")
    rel_path = get_sample_data_dir() / "RELIANCE_NS.parquet"
    if not rel_path.exists():
        rel_path = get_sample_data_dir() / "RELIANCE_NS.csv"
    assert rel_path.exists(), f"Reliance dataset not found at {rel_path}"

    df_rel = load_local_dataset(rel_path)
    print(f"1. Ingestion: Loaded Reliance ({df_rel.shape[0]} rows)")

    spark_rel = process_stock_data_spark(df_rel, ticker="RELIANCE.NS", output_dir=proc_dir)
    print(f"2. PySpark Processing Engine: {spark_rel['engine']}")
    print(f"3. Records Processed: {spark_rel['processed_records']}")
    assert Path(spark_rel["output_parquet"]).exists(), "Reliance Parquet file must exist."

    suite_rel = evaluate_models_suite(spark_rel["processed_df"], horizon=5)
    print(f"4. 5-Day Horizon Forecast (Best Model: {suite_rel['best_model_name']}):")
    best_rel = suite_rel["best_model_res"]
    print(f"   Current: ₹{best_rel['current_price']:.2f} -> 5-Day Target: ₹{best_rel['next_day_forecast']:.2f} ({best_rel['predicted_change_pct']:+.2f}%)")
    print(f"   Test MAE: ₹{best_rel['mae']:.2f} | Baseline MAE: ₹{best_rel['baseline_mae']:.2f}")
    assert best_rel["next_day_forecast"] > 0
    print("✓ Reliance prediction workflow passed successfully.")

    # 5. Verify Multi-Horizon Behavior
    print("\n[Stage 5] Verifying Multi-Horizon Support (1, 5, 10, 15 days)...")
    for h in [1, 5, 10, 15]:
        res_h = train_and_evaluate_model(df_nvda, model_name="Linear Regression", horizon=h)
        assert res_h["horizon_days"] == h
        assert res_h["next_day_forecast"] > 0
    print("✓ Multi-horizon support verified.")

    print("\n" + "=" * 65)
    print("ALL 5 WORKFLOW TEST STAGES PASSED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    run_tests()
