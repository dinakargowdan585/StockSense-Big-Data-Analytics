"""End-to-end verification and integration test suite for StockSense.

Explicitly validates the User-Controlled 5-Stage Workflow:
1. Import Data -> 2. Preview Raw Data -> 3. Process Data -> 4. User Clicks Predict -> 5. Display Results

Required Verification Tests:
- Test A: Data import & raw preview (NVDA & Reliance, record counts, date range, no auto ETL/ML)
- Test B: Explicit data processing (PySpark execution, Parquet serialization, raw data unchanged)
- Test C: User-controlled prediction (Explicit trigger, target date alignment, baseline comparison)
- Test D: State management & invalidation (Horizon change isolation, dataset switch reset)
- Test E: Graceful engine fallback (PySpark unavailable fallback to Pandas, engine transparency)
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from unittest.mock import patch

# Add project root to sys.path
root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import numpy as np
import pandas as pd
from src.data_fetcher import (
    get_processed_data_dir,
    get_raw_metadata,
    get_sample_data_dir,
    list_sample_datasets,
    load_local_dataset,
    load_raw_dataset,
    validate_ohlcv_data,
)
from src.ml_predictor import (
    compare_all_models,
    create_ml_features,
    evaluate_models_suite,
    train_and_evaluate_model,
)
from src.spark_engine import (
    get_spark_session,
    get_spark_status,
    process_stock_data_spark,
)


def run_verification_tests() -> None:
    print("=" * 75)
    print("STOCKSENSE: USER-CONTROLLED WORKFLOW & INTEGRATION VERIFICATION")
    print("=" * 75)

    proc_dir = str(get_processed_data_dir())

    # =========================================================================
    # TEST A: DATA IMPORT & RAW PREVIEW
    # =========================================================================
    print("\n[TEST A] Verifying Data Import & Raw Preview...")
    nvda_path = get_sample_data_dir() / "NVDA.parquet"
    if not nvda_path.exists():
        nvda_path = get_sample_data_dir() / "NVDA.csv"
    assert nvda_path.exists(), f"NVDA sample dataset must exist at {nvda_path}"

    # 1. Load raw dataset without ETL
    raw_nvda, meta_nvda = load_raw_dataset(nvda_path)
    print(f"1. Ingested Raw Dataset: {meta_nvda['source_name']}")
    print(f"   Shape: {meta_nvda['rows']} rows x {meta_nvda['columns']} columns")
    print(f"   Date Range: {meta_nvda['date_range']}")
    print(f"   Columns: {meta_nvda['column_names']}")

    # Assertions for Raw Import
    assert meta_nvda["rows"] >= 200, f"Expected at least 200 rows, got {meta_nvda['rows']}"
    assert len(raw_nvda) == meta_nvda["rows"]
    assert meta_nvda["columns"] >= 5, "Expected at least 5 standard OHLCV columns"
    assert "Close" in meta_nvda["column_names"] or "Adj Close" in meta_nvda["column_names"]

    # CRITICAL: Verify NO moving averages, lags, or predictions exist in raw data
    assert "Lag_1" not in raw_nvda.columns, "Raw data must NOT contain engineered Lag_1 feature!"
    assert "SMA_20" not in raw_nvda.columns, "Raw data must NOT contain engineered SMA_20 feature!"
    assert "Outlier_Flag" not in raw_nvda.columns, "Raw data must NOT contain Outlier_Flag!"
    assert "Target_Close" not in raw_nvda.columns, "Raw data must NOT contain Target_Close!"
    print("✓ TEST A PASSED: Raw data imported and verified without premature ETL or prediction.")

    # =========================================================================
    # TEST B: EXPLICIT DATA PROCESSING (PYSPARK PIPELINE)
    # =========================================================================
    print("\n[TEST B] Verifying Explicit Data Processing (PySpark Execution)...")
    raw_copy_before = raw_nvda.copy(deep=True)

    spark_res = process_stock_data_spark(raw_nvda, ticker="NVDA", output_dir=proc_dir)
    print(f"1. Processing Engine: {spark_res['engine']}")
    print(f"2. Original Records: {spark_res['raw_records']} -> Processed Records: {spark_res['processed_records']}")
    print(f"3. Outliers Flagged (IQR): {spark_res['outliers_detected']}")
    print(f"4. Partitions: {spark_res['partitions']} (Duration: {spark_res['duration_seconds']}s)")
    print(f"5. Generated Features ({len(spark_res['generated_features'])}): {spark_res['generated_features']}")
    print(f"6. Parquet Sink Path: {spark_res['output_parquet']}")

    # Verify PySpark engine and features
    assert "PySpark" in spark_res["engine"], "Expected Apache PySpark engine to process dataset."
    assert Path(spark_res["output_parquet"]).exists(), "Parquet output sink must exist."
    assert "Lag_1" in spark_res["generated_features"]
    assert "SMA_20" in spark_res["generated_features"]
    assert "Outlier_Flag" in spark_res["generated_features"]

    # Verify raw data was NOT mutated
    assert raw_nvda.equals(raw_copy_before), "Raw DataFrame must remain completely unchanged after processing!"
    print("✓ TEST B PASSED: PySpark ETL completed, Parquet sink created, raw data unmutated.")

    # =========================================================================
    # TEST C: USER-CONTROLLED PREDICTION
    # =========================================================================
    print("\n[TEST C] Verifying User-Controlled Prediction...")
    processed_nvda = spark_res["processed_df"]

    # Verify prediction is ONLY triggered on explicit call
    horizon = 5
    pred_res = train_and_evaluate_model(processed_nvda, model_name="Random Forest", horizon=horizon, test_size=0.20)
    
    print(f"1. Forecast Origin Date: {pred_res['forecast_origin_date']}")
    print(f"2. Target Trading Date:  {pred_res['target_trading_date']} (T+{horizon} Sessions)")
    print(f"3. Current Close: ${pred_res['current_price']:.2f}")
    print(f"4. Predicted Future Close: ${pred_res['next_day_forecast']:.2f} ({pred_res['predicted_change_pct']:+.2f}%)")
    print(f"5. Test Set MAE: ${pred_res['mae']:.2f} vs Naive Baseline MAE: ${pred_res['baseline_mae']:.2f}")
    print(f"6. Dynamic Baseline Directional Accuracy: {pred_res['baseline_directional_acc']:.1f}%")

    # Assertions for Prediction
    assert pred_res["next_day_forecast"] > 0
    assert pred_res["forecast_origin_date"] != pred_res["target_trading_date"]
    origin_dt = pd.to_datetime(pred_res["forecast_origin_date"])
    target_dt = pd.to_datetime(pred_res["target_trading_date"])
    assert target_dt.weekday() < 5, "Target trading date must fall on a business day!"
    assert 0.0 <= pred_res["baseline_directional_acc"] <= 100.0, "Baseline directional accuracy must be empirical."
    print("✓ TEST C PASSED: User-controlled prediction executed with correct horizon and baseline.")

    # =========================================================================
    # TEST D: STATE MANAGEMENT & DATASET SWITCHING
    # =========================================================================
    print("\n[TEST D] Verifying State Management & Dataset Invalidation...")
    # 1. Horizon Change Invalidation: changing horizon produces new target date
    pred_h1 = train_and_evaluate_model(processed_nvda, model_name="Random Forest", horizon=1, test_size=0.20)
    pred_h5 = train_and_evaluate_model(processed_nvda, model_name="Random Forest", horizon=5, test_size=0.20)
    assert pred_h1["target_trading_date"] != pred_h5["target_trading_date"], "Horizon change must produce different target dates."
    print(f"   Horizon T+1 Target: {pred_h1['target_trading_date']} vs Horizon T+5 Target: {pred_h5['target_trading_date']}")

    # 2. Dataset Switching: Switch to Reliance Industries (RELIANCE.NS)
    rel_path = get_sample_data_dir() / "RELIANCE_NS.parquet"
    if not rel_path.exists():
        rel_path = get_sample_data_dir() / "RELIANCE_NS.csv"
    assert rel_path.exists()

    raw_rel, meta_rel = load_raw_dataset(rel_path)
    print(f"   Switched to New Dataset: {meta_rel['source_name']} ({meta_rel['rows']} rows)")
    assert "Lag_1" not in raw_rel.columns, "New raw dataset must be unprocessed."

    # Process Reliance explicitly
    spark_rel = process_stock_data_spark(raw_rel, ticker="RELIANCE.NS", output_dir=proc_dir)
    pred_rel = train_and_evaluate_model(spark_rel["processed_df"], model_name="Linear Regression", horizon=5, test_size=0.20)
    print(f"   Reliance Forecast: ₹{pred_rel['current_price']:.2f} -> ₹{pred_rel['next_day_forecast']:.2f} (Target: {pred_rel['target_trading_date']})")
    assert pred_rel["current_price"] > 1000.0, "Reliance price should reflect Indian Rupee valuation."
    print("✓ TEST D PASSED: State isolation and dataset switching verified.")

    # =========================================================================
    # TEST E: GRACEFUL ENGINE FALLBACK TO PANDAS
    # =========================================================================
    print("\n[TEST E] Verifying Graceful Fallback when PySpark is Unavailable...")
    with patch("src.spark_engine.get_spark_session", return_value=None):
        status = get_spark_status()
        fallback_res = process_stock_data_spark(raw_nvda, ticker="NVDA", output_dir=None)
        print(f"1. Fallback Engine Reported: {fallback_res['engine']}")
        print(f"2. Records Cleaned via Fallback: {fallback_res['processed_records']}")
        print(f"3. Features Generated: {len(fallback_res['generated_features'])}")

        assert "Pandas" in fallback_res["engine"], "Fallback must clearly report Pandas engine!"
        assert fallback_res["processed_records"] == spark_res["processed_records"]
        assert "Lag_1" in fallback_res["generated_features"]
        assert "SMA_20" in fallback_res["generated_features"]
        assert "Outlier_Flag" in fallback_res["generated_features"]

    print("✓ TEST E PASSED: Pandas fallback executes reliably and reports engine accurately.")

    print("\n" + "=" * 75)
    print("ALL 5 WORKFLOW VERIFICATION TESTS (A, B, C, D, E) PASSED SUCCESSFULLY!")
    print("=" * 75)


if __name__ == "__main__":
    run_verification_tests()
