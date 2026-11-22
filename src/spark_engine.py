"""Apache Spark integration and distributed data processing engine for StockSense.

Provides a robust SparkSession manager with conservative local resource allocation,
Parquet and CSV ingestion, distributed transformations, window aggregations,
and an automatic graceful fallback to Pandas if PySpark or Java is unavailable.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Spark import check
_PYSPARK_AVAILABLE = False
_SPARK_ERROR_REASON = ""

try:
    import pyspark
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window
    from pyspark.sql.types import (
        StructType,
        StructField,
        StringType,
        DoubleType,
        LongType,
        DateType,
        TimestampType,
    )
    _PYSPARK_AVAILABLE = True
except Exception as e:
    _PYSPARK_AVAILABLE = False
    _SPARK_ERROR_REASON = str(e)


_SPARK_SESSION: Optional[Any] = None


def get_spark_status() -> dict[str, Any]:
    """Return diagnostic status of PySpark and Java environment."""
    java_home = os.environ.get("JAVA_HOME", "Default system path")
    return {
        "pyspark_installed": _PYSPARK_AVAILABLE,
        "error_reason": _SPARK_ERROR_REASON if not _PYSPARK_AVAILABLE else None,
        "java_home": java_home,
        "spark_active": _SPARK_SESSION is not None,
    }


def get_spark_session(app_name: str = "StockSense-BigData") -> Optional[Any]:
    """Initialize or retrieve a local SparkSession configured for modest hardware.

    Conservative local configuration:
    - Master: local[2] (uses 2 worker threads)
    - Driver memory: 1g
    - Shuffle partitions: 4
    - UI disabled to avoid port contention
    """
    global _SPARK_SESSION

    if not _PYSPARK_AVAILABLE:
        return None

    if _SPARK_SESSION is not None:
        try:
            # Check if alive
            _SPARK_SESSION.sparkContext.getConf()
            return _SPARK_SESSION
        except Exception:
            _SPARK_SESSION = None

    try:
        spark = (
            SparkSession.builder.appName(app_name)
            .master("local[2]")
            .config("spark.driver.memory", "1g")
            .config("spark.sql.shuffle.partitions", "4")
            .config("spark.ui.enabled", "false")
            .config("spark.driver.bindAddress", "127.0.0.1")
            .config("spark.sql.execution.arrow.pyspark.enabled", "false")
            .getOrCreate()
        )
        # Suppress excessive info logs from Py4J/Spark
        spark.sparkContext.setLogLevel("WARN")
        _SPARK_SESSION = spark
        return _SPARK_SESSION
    except Exception as exc:
        logger.warning(f"Could not initialize PySpark session: {exc}")
        return None


def stop_spark_session() -> None:
    """Safely shut down the active SparkSession."""
    global _SPARK_SESSION
    if _SPARK_SESSION is not None:
        try:
            _SPARK_SESSION.stop()
        except Exception as exc:
            logger.warning(f"Error stopping Spark session: {exc}")
        finally:
            _SPARK_SESSION = None


def process_stock_data_spark(
    df: pd.DataFrame,
    ticker: str,
    output_dir: Optional[str] = None,
) -> dict[str, Any]:
    """Process a stock dataset using Apache Spark DataFrame APIs.

    If Spark is unavailable or fails, gracefully falls back to Pandas processing.
    The returned dictionary explicitly states the engine used.
    """
    start_time = time.perf_counter()
    spark = get_spark_session()

    # Normalize incoming DataFrame
    flat_df = df.copy()
    if isinstance(flat_df.columns, pd.MultiIndex):
        if ticker in flat_df.columns.get_level_values(1):
            flat_df = flat_df.xs(ticker, axis=1, level=1)
        elif ticker in flat_df.columns.get_level_values(0):
            flat_df = flat_df[ticker]

    flat_df = flat_df.reset_index()
    date_col = "Date" if "Date" in flat_df.columns else flat_df.columns[0]
    flat_df[date_col] = pd.to_datetime(flat_df[date_col])

    # Standardize column names
    col_mapping = {}
    for col in flat_df.columns:
        clean_name = str(col).replace(" ", "_")
        col_mapping[col] = clean_name
    flat_df = flat_df.rename(columns=col_mapping)
    date_col_clean = col_mapping[date_col]

    raw_count = len(flat_df)
    missing_before = int(flat_df.isna().sum().sum())
    duplicates_before = int(flat_df.duplicated(subset=[date_col_clean]).sum())

    # --- PANDAS FALLBACK IMPLEMENTATION ---
    if spark is None:
        cleaned_df = flat_df.drop_duplicates(subset=[date_col_clean]).sort_values(by=date_col_clean)
        cleaned_df = cleaned_df.ffill().bfill().dropna()
        
        close_col = "Adj_Close" if "Adj_Close" in cleaned_df.columns else "Close"
        cleaned_df["Daily_Return"] = cleaned_df[close_col].pct_change().fillna(0.0)
        cleaned_df["SMA_20"] = cleaned_df[close_col].rolling(20, min_periods=1).mean()
        cleaned_df["SMA_50"] = cleaned_df[close_col].rolling(50, min_periods=1).mean()
        cleaned_df["Volatility_20d"] = cleaned_df["Daily_Return"].rolling(20, min_periods=1).std().fillna(0.0)
        
        # Outlier detection: IQR on Daily Return
        q1 = cleaned_df["Daily_Return"].quantile(0.25)
        q3 = cleaned_df["Daily_Return"].quantile(0.75)
        iqr = q3 - q1
        outliers = int(((cleaned_df["Daily_Return"] < (q1 - 1.5 * iqr)) | (cleaned_df["Daily_Return"] > (q3 + 1.5 * iqr))).sum())
        
        saved_path = None
        if output_dir:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            saved_path = str(out_path / f"{ticker}_processed.parquet")
            cleaned_df.to_parquet(saved_path, index=False)

        duration = time.perf_counter() - start_time
        return {
            "engine": "Pandas (Spark Unavailable Fallback)",
            "ticker": ticker,
            "raw_records": raw_count,
            "processed_records": len(cleaned_df),
            "missing_values_handled": missing_before,
            "duplicates_removed": duplicates_before,
            "outliers_detected": outliers,
            "partitions": 1,
            "duration_seconds": round(duration, 4),
            "output_parquet": saved_path,
            "schema": [(c, str(dtype)) for c, dtype in cleaned_df.dtypes.items()],
            "preview_df": cleaned_df.tail(10),
        }

    # --- PYSPARK DISTRIBUTED IMPLEMENTATION ---
    try:
        spark_df = spark.createDataFrame(flat_df)
        
        # Clean duplicates & handle nulls
        spark_df = spark_df.dropDuplicates([date_col_clean]).dropna(how="all")
        spark_df = spark_df.na.fill(value=0.0)
        
        close_col = "Adj_Close" if "Adj_Close" in spark_df.columns else "Close"

        # PySpark Window Functions for Feature Engineering
        window_spec_order = Window.partitionBy(F.lit("GLOBAL")).orderBy(date_col_clean)
        window_20 = Window.partitionBy(F.lit("GLOBAL")).orderBy(date_col_clean).rowsBetween(-19, 0)
        window_50 = Window.partitionBy(F.lit("GLOBAL")).orderBy(date_col_clean).rowsBetween(-49, 0)

        # Lag price to compute return
        spark_df = spark_df.withColumn("Prev_Close", F.lag(F.col(close_col), 1).over(window_spec_order))
        spark_df = spark_df.withColumn(
            "Daily_Return",
            F.when(F.col("Prev_Close").isNull() | (F.col("Prev_Close") == 0), 0.0)
            .otherwise((F.col(close_col) - F.col("Prev_Close")) / F.col("Prev_Close")),
        )
        spark_df = spark_df.withColumn("SMA_20", F.avg(F.col(close_col)).over(window_20))
        spark_df = spark_df.withColumn("SMA_50", F.avg(F.col(close_col)).over(window_50))
        spark_df = spark_df.withColumn("Volatility_20d", F.stddev(F.col("Daily_Return")).over(window_20))
        spark_df = spark_df.na.fill(value=0.0)

        # Count & Repartition into Parquet
        num_partitions = spark_df.rdd.getNumPartitions()
        record_count = spark_df.count()

        # Outlier calculation
        q_quantiles = spark_df.approxQuantile("Daily_Return", [0.25, 0.75], 0.01)
        q1, q3 = q_quantiles[0], q_quantiles[1]
        iqr = q3 - q1
        outliers = spark_df.filter(
            (F.col("Daily_Return") < (q1 - 1.5 * iqr)) | (F.col("Daily_Return") > (q3 + 1.5 * iqr))
        ).count()

        saved_path = None
        if output_dir:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            saved_path = str(out_path / f"{ticker}_processed.parquet")
            # Write via PySpark or Pandas bridge
            spark_df.toPandas().to_parquet(saved_path, index=False)

        duration = time.perf_counter() - start_time
        preview = spark_df.orderBy(F.desc(date_col_clean)).limit(10).toPandas().iloc[::-1]

        return {
            "engine": "Apache PySpark (Distributed Local)",
            "ticker": ticker,
            "raw_records": raw_count,
            "processed_records": record_count,
            "missing_values_handled": missing_before,
            "duplicates_removed": duplicates_before,
            "outliers_detected": int(outliers),
            "partitions": num_partitions,
            "duration_seconds": round(duration, 4),
            "output_parquet": saved_path,
            "schema": [(field.name, str(field.dataType)) for field in spark_df.schema],
            "preview_df": preview,
        }

    except Exception as exc:
        logger.error(f"PySpark processing error: {exc}. Reverting to Pandas fallback.")
        # Graceful fallback on any Spark runtime issue
        cleaned_df = flat_df.drop_duplicates(subset=[date_col_clean]).sort_values(by=date_col_clean)
        cleaned_df = cleaned_df.ffill().bfill().dropna()
        close_col = "Adj_Close" if "Adj_Close" in cleaned_df.columns else "Close"
        cleaned_df["Daily_Return"] = cleaned_df[close_col].pct_change().fillna(0.0)
        cleaned_df["SMA_20"] = cleaned_df[close_col].rolling(20, min_periods=1).mean()
        cleaned_df["SMA_50"] = cleaned_df[close_col].rolling(50, min_periods=1).mean()
        cleaned_df["Volatility_20d"] = cleaned_df["Daily_Return"].rolling(20, min_periods=1).std().fillna(0.0)
        
        duration = time.perf_counter() - start_time
        return {
            "engine": "Pandas (PySpark Runtime Exception Fallback)",
            "ticker": ticker,
            "raw_records": raw_count,
            "processed_records": len(cleaned_df),
            "missing_values_handled": missing_before,
            "duplicates_removed": duplicates_before,
            "outliers_detected": 0,
            "partitions": 1,
            "duration_seconds": round(duration, 4),
            "output_parquet": None,
            "schema": [(c, str(dtype)) for c, dtype in cleaned_df.dtypes.items()],
            "preview_df": cleaned_df.tail(10),
            "warning": f"PySpark raised an exception: {exc}",
        }
