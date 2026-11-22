# ⚡ StockSense – Big Data Analytics in Stock Market Prediction

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Apache Spark](https://img.shields.io/badge/Apache%20Spark-3.5-E25A1C?logo=apachespark&logoColor=white)](https://spark.apache.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#license)

**StockSense** is an end-to-end academic Big Data Analytics and Machine Learning platform for equity market intelligence. It demonstrates large-scale financial data ingestion, distributed preprocessing and window transformations using **Apache Spark (PySpark)**, multi-asset statistical analytics, chronological predictive modeling, and empirical benchmarking between single-node Pandas and distributed Spark execution.

---

## 🎯 Academic Project Objectives

1. **Large-Scale Financial Ingestion:** Ingest historical OHLCV data across US Equities (NYSE/NASDAQ) and Indian National Stock Exchange (NSE) assets, with support for CSV, Parquet, and uploaded datasets.
2. **Distributed Data Processing:** Leverage Apache Spark DataFrames, partition management, and window functions (`F.lag`, `F.avg`, `F.stddev`) for feature engineering without single-machine memory bottlenecks.
3. **Four Pillars of Analytics:**
   - **Descriptive Analytics:** Statistical moments (Mean, Variance, Skewness, Kurtosis), price distributions, and monthly returns.
   - **Diagnostic Analytics:** Value-at-Risk (95% & 99% VaR), Conditional VaR (CVaR), maximum drawdown curves, and volume/price shock anomaly detection.
   - **Predictive Analytics:** Next-day stock price forecasting using chronological cross-validation with zero future-data leakage.
   - **Prescriptive Analytics:** Rule-based quantitative market signals (Golden/Death Cross, RSI overbought/oversold, Bollinger Band bounds, and risk classification tiers).
4. **Empirical Big Data Benchmarking:** Measure real wall-clock latency, partition scalability, and throughput comparing in-memory Pandas vs. distributed PySpark.

---

## 🌐 The Five Vs of Big Data Demonstrated

| Big Data 'V' | Implementation in StockSense |
| :--- | :--- |
| **Volume** | Scalable multi-year historical datasets spanning hundreds of thousands of financial records; partitioned PySpark execution handles datasets exceeding single-node RAM. |
| **Velocity** | Automated ingestion cadence via Yahoo Finance API with dynamic caching (`@st.cache_data`) and on-demand live market refresh. |
| **Variety** | Multi-modal data ingestion: binary Apache Parquet files, structured tabular CSV, nested MultiIndex market arrays, and company metadata. |
| **Veracity** | Robust data hygiene pipeline: duplicate timestamp elimination, backward/forward holiday imputation, schema type enforcement, and IQR return outlier detection. |
| **Value** | Actionable quantitative output: probabilistic next-day price forecasts, risk tier classifications, and empirical distributed performance benchmarks. |

---

## 🏗️ System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                              DATA SOURCES                              │
│   Yahoo Finance API  ·  Indian NSE (.NS)  ·  Local CSV & Parquet Files │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   INGESTION & DATA HYGIENE PIPELINE                    │
│   • Strict Schema Casting (Timestamp, DoubleType)                      │
│   • Holiday Imputation & Duplicate Timestamp Deduplication             │
│   • Interquartile Range (IQR) Outlier Detection (Q1 - 1.5*IQR)         │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                  DISTRIBUTED APACHE SPARK ENGINE                       │
│   • PySpark DataFrame Operations (Master: local[2], Partitions: 4)     │
│   • Distributed Window Aggregations (Lag-1 returns, SMA 20, SMA 50)    │
│   • Optimized Columnar Apache Parquet Serialization (data/processed/)  │
└───────────────────┬────────────────────────────────┬───────────────────┘
                    │                                │
                    ▼                                ▼
┌───────────────────────────────────────┐ ┌──────────────────────────────┐
│           ANALYTICS ENGINE            │ │   MACHINE LEARNING ENGINE    │
│ • Descriptive: Moments & Returns      │ │ • Chronological 80/20 Split  │
│ • Diagnostic: VaR, Drawdown, Anomalies│ │ • Random Forest Regressor    │
│ • Prescriptive: Rule Signals & Tiers  │ │ • Linear Regression Baseline │
│                                       │ │ • Gradient Boosting Regressor│
└───────────────────┬───────────────────┘ └──────────────┬───────────────┘
                    │                                    │
                    └─────────────────┬──────────────────┘
                                      │
                                      ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   STREAMLIT FINANCIAL DASHBOARD UI                     │
│  Market Overview · Stock Deep Dive · Spark Pipeline · ML Prediction   │
│         Performance Benchmarking Suite · Analytical Reports            │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Installation & Local Execution

### 1. Prerequisites
- **Python:** 3.10 to 3.12
- **Java:** OpenJDK 11 or 17 (Required for Apache Spark runtime)

Verify Java installation:
```bash
java -version
```

### 2. Setup Virtual Environment
```bash
cd stock-analysis-dashboard

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate       # macOS / Linux
# .venv\Scripts\activate        # Windows (PowerShell)

# Install required dependencies
pip install -r requirements.txt
```

### 3. Generate Offline Sample Datasets (Optional but Recommended)
Populate the `data/sample/` directory with local CSV and Parquet datasets for offline operation:
```bash
python scripts/generate_sample_data.py
```

### 4. Launch the Dashboard
```bash
streamlit run dashboard/app.py
```
Open your browser to: **`http://localhost:8501`**

---

## 🖥️ Dashboard Page Navigation

1. **🌐 Market Overview:** Multi-asset snapshot cards, normalized Base = 100 relative performance chart, and comparative performance rankings.
2. **📈 Stock Analysis:** Single-equity deep dive with interactive OHLCV Candlesticks, 20/50-day Simple Moving Averages, volume profiles, and return histograms.
3. **⚡ Big Data Processing:** Spark ingestion metrics, distributed window transformations, schema inspection, IQR outlier detection, and Parquet storage verification.
4. **🔬 Big Data Analytics:**
   - *Descriptive:* Distribution moments, price statistics, and monthly return aggregations.
   - *Diagnostic:* Value-at-Risk (95% & 99%), Conditional VaR (CVaR), maximum drawdown curves, and Z-score anomaly detection.
   - *Prescriptive:* Rule-based quantitative signals (Golden/Death Cross, RSI 14 momentum, Bollinger Bands, and risk tiers).
5. **🤖 Stock Prediction:** Chronological train/test validation, actual vs. predicted price visualization, residual error distributions, and comparative evaluation across Random Forest, Linear Regression, and Gradient Boosting.
6. **⏱️ Performance Benchmark:** Real empirical wall-clock timing tests comparing Pandas against Apache PySpark across GroupBy aggregations, window functions, and multi-predicate filtering.
7. **📑 Reports & Export:** Automated executive briefing generation, with one-click export for cleaned CSV, Parquet, and Markdown summary reports.
8. **🎓 About Project:** Comprehensive academic presentation detailing project architecture, problem statement, methodology, and the 5 Vs of Big Data.

---

## 📂 Project Repository Structure

```
stock-analysis-dashboard/
├── dashboard/
│   └── app.py                     # Main Streamlit Big Data dashboard (8 pages)
├── src/
│   ├── __init__.py
│   ├── spark_engine.py            # Apache Spark session manager & distributed pipeline
│   ├── analytics.py               # Descriptive, Diagnostic, and Prescriptive analytics
│   ├── ml_predictor.py            # Machine Learning training, evaluation & forecasting
│   ├── benchmarking.py            # Empirical Pandas vs. PySpark benchmark suite
│   ├── data_fetcher.py            # Yahoo Finance ingestion & local dataset handler
│   └── analyzer.py                # Core financial calculations (preserved)
├── data/
│   ├── sample/                    # Pre-generated offline CSV & Parquet files
│   └── processed/                 # Partitioned Parquet output from Spark jobs
├── notebooks/
│   └── stock_analysis.ipynb       # Exploratory Jupyter research notebook
├── scripts/
│   └── generate_sample_data.py    # Offline dataset generation script
├── requirements.txt               # Updated dependency specifications
├── streamlit_app.py               # Root deployment entrypoint
└── README.md                      # Academic documentation & execution guide
```

---

## 🔬 Academic Limitations & Engineering Notes

1. **Local Spark Deployment:** For reproducible academic demonstration on standard laptops, Spark runs in `local[2]` master mode rather than a distributed multi-node Hadoop cluster.
2. **Pandas vs. PySpark Trade-off:** On modest in-memory datasets (< 50,000 records), Pandas executed natively in C exhibits lower latency than PySpark due to fixed JVM initialization, Py4J serialization, and task scheduling overhead. The benchmark suite demonstrates where the horizontal scaling crossover point emerges.
3. **Stationarity in Financial Forecasting:** Daily equity prices are non-stationary time series; models predict next-day probabilistic price tendencies based on historical momentum and volatility rather than guaranteed outcomes.

---

## 📜 License

This project is licensed under the **MIT License**.
