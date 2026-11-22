"""End-to-end integration and smoke tests for StockSense."""

import sys
from pathlib import Path

# Add project root
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

import pandas as pd
from src.data_fetcher import (
    load_local_dataset,
    list_sample_datasets,
    get_stock_info,
    US_PRESETS,
    NSE_PRESETS,
)
from src.spark_engine import (
    get_spark_status,
    get_spark_session,
    process_stock_data_spark,
)
from src.analytics import (
    compute_descriptive_stats,
    compute_diagnostic_analytics,
    compute_market_performance_rankings,
    generate_prescriptive_signals,
)
from src.ml_predictor import (
    train_and_evaluate_model,
    compare_all_models,
)
from src.benchmarking import run_full_benchmark

def run_tests():
    print("=== 1. Checking Spark Status ===")
    status = get_spark_status()
    print("Spark status:", status)
    assert status["pyspark_installed"], "PySpark should be installed"

    print("\n=== 2. Testing Sample Data Loading ===")
    samples = list_sample_datasets()
    print(f"Discovered {len(samples)} offline sample datasets.")
    assert len(samples) >= 4, "Should have at least 4 sample datasets"

    sample = samples[0]
    df = load_local_dataset(sample["path"])
    print(f"Loaded {sample['filename']}: shape={df.shape}")
    assert not df.empty, "Dataset must not be empty"

    print("\n=== 3. Testing PySpark Data Ingestion & Preprocessing ===")
    spark_res = process_stock_data_spark(df, ticker="AAPL")
    print(f"Engine: {spark_res['engine']}")
    print(f"Records: {spark_res['processed_records']}, Missing Handled: {spark_res['missing_values_handled']}, Outliers: {spark_res['outliers_detected']}")
    assert spark_res["processed_records"] > 0, "Should have processed records"

    print("\n=== 4. Testing Analytics Pillars ===")
    # Descriptive
    desc = compute_descriptive_stats(df, "AAPL")
    print(f"Descriptive -> Mean Price: {desc['mean_price']}, Ann. Return: {desc['annualized_return_pct']}%")
    assert desc["mean_price"] > 0

    # Diagnostic
    diag = compute_diagnostic_analytics(df, "AAPL")
    print(f"Diagnostic -> Max Drawdown: {diag['max_drawdown_pct']}%, VaR 95%: {diag['var_95_pct']}%")
    assert diag["max_drawdown_pct"] <= 0

    # Prescriptive
    pres = generate_prescriptive_signals(df, "AAPL")
    print(f"Prescriptive -> Action: {pres['action']}, Risk: {pres['risk_tier']}, RSI: {pres['rsi_value']}")
    assert "action" in pres and "supporting_reasons" in pres

    print("\n=== 5. Testing Machine Learning Prediction ===")
    rf_res = train_and_evaluate_model(df, model_name="Random Forest", test_size=0.20)
    print(f"Random Forest -> MAE: ${rf_res['mae']}, Directional Accuracy: {rf_res['directional_accuracy_pct']}%, Forecast: ${rf_res['next_day_forecast']}")
    assert rf_res["mae"] > 0

    lr_res = train_and_evaluate_model(df, model_name="Linear Regression", test_size=0.20)
    print(f"Linear Regression -> MAE: ${lr_res['mae']}, R2: {lr_res['r2_score']}")
    assert lr_res["mae"] > 0

    comp_df = compare_all_models(df, test_size=0.20)
    print("Multi-Model Comparison Table:")
    print(comp_df)
    assert len(comp_df) >= 2

    print("\n=== 6. Testing Performance Benchmark Suite ===")
    bench_out = run_full_benchmark(df, record_counts=[1000])
    b_df = bench_out["benchmark_df"]
    print(b_df[["Records", "Operation", "Pandas (s)", "PySpark (s)", "Faster Engine"]])
    assert len(b_df) == 3

    print("\n=== ALL TEST CHECKS PASSED SUCCESSFULLY! ===")

if __name__ == "__main__":
    run_tests()
