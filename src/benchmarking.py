"""Benchmarking module for StockSense.

Performs real, measurable performance benchmarks comparing Pandas (single-threaded in-memory)
against Apache PySpark (distributed DataFrame engine) across financial data operations:
1. Multi-key Aggregation (GroupBy Month & Ticker)
2. Window Analytics (Rolling 20-day Moving Average & Volatility)
3. Complex Filtering & Transformation

Measures wall-clock time, records processed, partitions, and memory footprint.
Includes academic analysis on distributed computing overhead vs horizontal scaling.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Tuple

import numpy as np
import pandas as pd

from src.spark_engine import get_spark_session

logger = logging.getLogger(__name__)


def generate_benchmark_dataset(base_df: pd.DataFrame, target_records: int = 50000) -> pd.DataFrame:
    """Generate a realistic scaled dataset based on real stock data for benchmarking."""
    flat_df = base_df.copy()
    if isinstance(flat_df.columns, pd.MultiIndex):
        flat_df.columns = [f"{c[0]}_{c[1]}" if c[1] else c[0] for c in flat_df.columns]
    flat_df = flat_df.reset_index()

    # Identify primary columns
    close_col = next((c for c in flat_df.columns if "Close" in c or "close" in c), flat_df.columns[1])
    vol_col = next((c for c in flat_df.columns if "Volume" in c or "volume" in c), flat_df.columns[2])

    records_per_copy = len(flat_df)
    if records_per_copy == 0:
        raise ValueError("Base DataFrame is empty.")

    copies_needed = max(int(np.ceil(target_records / records_per_copy)), 1)
    
    ticker_pool = ["AAPL", "NVDA", "MSFT", "GOOGL", "AMZN", "TSLA", "VOO", "QQQM", "RELIANCE.NS", "TCS.NS"]
    
    dfs = []
    for i in range(copies_needed):
        copy_df = flat_df.copy()
        ticker_assigned = ticker_pool[i % len(ticker_pool)]
        copy_df["Ticker"] = f"{ticker_assigned}_{i}"
        # Inject mild synthetic variance across replicated chunks
        multiplier = 1.0 + (np.sin(i) * 0.05)
        copy_df["Close"] = copy_df[close_col] * multiplier
        copy_df["Volume"] = (copy_df[vol_col] * multiplier).astype(int)
        copy_df["Date"] = pd.date_range(start="2015-01-01", periods=len(copy_df), freq="D")
        dfs.append(copy_df[["Date", "Ticker", "Close", "Volume"]])

    scaled_df = pd.concat(dfs, ignore_index=True).iloc[:target_records]
    scaled_df["Month"] = scaled_df["Date"].dt.to_period("M").astype(str)
    return scaled_df


def benchmark_aggregation(df: pd.DataFrame, spark_session: Any) -> Tuple[float, float, int]:
    """Benchmark GroupBy Aggregation: Mean, Max, Min, Std on Close and Volume by Month."""
    # 1. Pandas Benchmark
    t0 = time.perf_counter()
    pandas_agg = df.groupby(["Month", "Ticker"]).agg({
        "Close": ["mean", "max", "min", "std"],
        "Volume": ["mean", "sum"],
    })
    pandas_duration = time.perf_counter() - t0

    # 2. PySpark Benchmark
    if spark_session is None:
        return pandas_duration, -1.0, 1

    from pyspark.sql import functions as F
    
    t_start_spark = time.perf_counter()
    spark_df = spark_session.createDataFrame(df)
    spark_agg = (
        spark_df.groupBy("Month", "Ticker")
        .agg(
            F.avg("Close").alias("avg_close"),
            F.max("Close").alias("max_close"),
            F.min("Close").alias("min_close"),
            F.stddev("Close").alias("std_close"),
            F.avg("Volume").alias("avg_vol"),
            F.sum("Volume").alias("sum_vol"),
        )
    )
    # Trigger Spark action to force complete distributed execution
    _ = spark_agg.count()
    spark_duration = time.perf_counter() - t_start_spark
    partitions = spark_df.rdd.getNumPartitions()

    return pandas_duration, spark_duration, partitions


def benchmark_window_operations(df: pd.DataFrame, spark_session: Any) -> Tuple[float, float, int]:
    """Benchmark Window Functions: Rolling 20-row Moving Average & Volatility."""
    # 1. Pandas Benchmark
    t0 = time.perf_counter()
    df_sorted = df.sort_values(by=["Ticker", "Date"]).copy()
    df_sorted["SMA_20"] = df_sorted.groupby("Ticker")["Close"].transform(lambda x: x.rolling(20, min_periods=1).mean())
    df_sorted["STD_20"] = df_sorted.groupby("Ticker")["Close"].transform(lambda x: x.rolling(20, min_periods=1).std())
    pandas_duration = time.perf_counter() - t0

    # 2. PySpark Benchmark
    if spark_session is None:
        return pandas_duration, -1.0, 1

    from pyspark.sql import functions as F
    from pyspark.sql.window import Window

    t_start_spark = time.perf_counter()
    spark_df = spark_session.createDataFrame(df)
    window_spec = Window.partitionBy("Ticker").orderBy("Date").rowsBetween(-19, 0)
    
    spark_windowed = (
        spark_df
        .withColumn("SMA_20", F.avg("Close").over(window_spec))
        .withColumn("STD_20", F.stddev("Close").over(window_spec))
    )
    _ = spark_windowed.count()
    spark_duration = time.perf_counter() - t_start_spark
    partitions = spark_df.rdd.getNumPartitions()

    return pandas_duration, spark_duration, partitions


def benchmark_complex_filtering(df: pd.DataFrame, spark_session: Any) -> Tuple[float, float, int]:
    """Benchmark Complex Filtering: Multi-predicate filter, derived metrics, and sort."""
    avg_vol = float(df["Volume"].mean())

    # 1. Pandas Benchmark
    t0 = time.perf_counter()
    filtered_pandas = df[
        (df["Volume"] > avg_vol) & (df["Close"] > 100.0)
    ].copy()
    filtered_pandas["Normalized_Value"] = filtered_pandas["Close"] * filtered_pandas["Volume"]
    sorted_pandas = filtered_pandas.sort_values(by="Normalized_Value", ascending=False)
    _ = len(sorted_pandas)
    pandas_duration = time.perf_counter() - t0

    # 2. PySpark Benchmark
    if spark_session is None:
        return pandas_duration, -1.0, 1

    from pyspark.sql import functions as F

    t_start_spark = time.perf_counter()
    spark_df = spark_session.createDataFrame(df)
    filtered_spark = (
        spark_df
        .filter((F.col("Volume") > avg_vol) & (F.col("Close") > 100.0))
        .withColumn("Normalized_Value", F.col("Close") * F.col("Volume"))
        .sort(F.desc("Normalized_Value"))
    )
    _ = filtered_spark.count()
    spark_duration = time.perf_counter() - t_start_spark
    partitions = spark_df.rdd.getNumPartitions()

    return pandas_duration, spark_duration, partitions


def run_full_benchmark(
    base_df: pd.DataFrame,
    record_counts: list[int] = [10000, 50000, 100000],
) -> dict[str, Any]:
    """Execute complete benchmarking suite across multiple dataset sizes."""
    spark = get_spark_session()
    results = []

    for count in record_counts:
        scaled_df = generate_benchmark_dataset(base_df, target_records=count)
        memory_mb = round(scaled_df.memory_usage(deep=True).sum() / (1024 * 1024), 2)

        # 1. Aggregation
        p_time, s_time, parts = benchmark_aggregation(scaled_df, spark)
        results.append({
            "Records": count,
            "Memory (MB)": memory_mb,
            "Operation": "GroupBy & Aggregation",
            "Pandas (s)": round(p_time, 4),
            "PySpark (s)": round(s_time, 4) if s_time > 0 else "N/A",
            "Partitions": parts,
            "Faster Engine": "Pandas" if (s_time < 0 or p_time < s_time) else "PySpark",
            "Speedup": round(p_time / s_time, 2) if s_time > 0 and s_time < p_time else (round(s_time / p_time, 2) if s_time > 0 else 1.0),
        })

        # 2. Window Operations
        p_time_w, s_time_w, parts_w = benchmark_window_operations(scaled_df, spark)
        results.append({
            "Records": count,
            "Memory (MB)": memory_mb,
            "Operation": "Window Rolling Functions",
            "Pandas (s)": round(p_time_w, 4),
            "PySpark (s)": round(s_time_w, 4) if s_time_w > 0 else "N/A",
            "Partitions": parts_w,
            "Faster Engine": "Pandas" if (s_time_w < 0 or p_time_w < s_time_w) else "PySpark",
            "Speedup": round(p_time_w / s_time_w, 2) if s_time_w > 0 and s_time_w < p_time_w else (round(s_time_w / p_time_w, 2) if s_time_w > 0 else 1.0),
        })

        # 3. Filtering
        p_time_f, s_time_f, parts_f = benchmark_complex_filtering(scaled_df, spark)
        results.append({
            "Records": count,
            "Memory (MB)": memory_mb,
            "Operation": "Filtering & Transformation",
            "Pandas (s)": round(p_time_f, 4),
            "PySpark (s)": round(s_time_f, 4) if s_time_f > 0 else "N/A",
            "Partitions": parts_f,
            "Faster Engine": "Pandas" if (s_time_f < 0 or p_time_f < s_time_f) else "PySpark",
            "Speedup": round(p_time_f / s_time_f, 2) if s_time_f > 0 and s_time_f < p_time_f else (round(s_time_f / p_time_f, 2) if s_time_f > 0 else 1.0),
        })

    benchmark_df = pd.DataFrame(results)

    academic_analysis = (
        "### Academic Big Data Analysis: Distributed Processing vs Single-Node\n\n"
        "1. **JVM & Driver Overhead:** For modest in-memory datasets (< 50,000–100,000 records), "
        "Pandas executed natively in C/NumPy typically exhibits lower latency than PySpark. "
        "This occurs because PySpark incurs fixed overhead costs: Python-to-JVM Py4J serialization, "
        "DAG query planning, and thread scheduling.\n\n"
        "2. **Horizontal Scalability Threshold:** As datasets scale into millions of records and exceed single-node RAM, "
        "Pandas triggers out-of-memory (OOM) fatal crashes. In contrast, PySpark leverages distributed partition memory, "
        "lazy evaluation, Catalyst query optimization, and spill-to-disk capabilities to process datasets of arbitrary scale.\n\n"
        "3. **Partitioning Impact:** PySpark splits DataFrames across partitions (configured conservative local mode: 4 partitions). "
        "On multi-core or multi-node clusters, parallel execution across partitions delivers linear speedups on large data."
    )

    return {
        "benchmark_df": benchmark_df,
        "academic_analysis": academic_analysis,
        "spark_available": spark is not None,
    }
