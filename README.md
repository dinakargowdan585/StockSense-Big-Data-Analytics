# ⚡ StockSense – Big Data Analytics in Stock Market Prediction

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Apache Spark](https://img.shields.io/badge/Apache%20Spark-3.5-E25A1C?logo=apachespark&logoColor=white)](https://spark.apache.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#license)

**StockSense** is an academic Big Data analytics and machine learning application dedicated to **Stock Market Prediction**. It demonstrates end-to-end distributed data engineering using **Apache PySpark 3.5**, feature engineering over distributed window partitions, columnar **Apache Parquet** storage, chronological train/test partitioning without data leakage, and multi-model forecasting benchmarked against a persistence baseline.

---

## 🎯 Academic Project Identity & Objectives

- **Primary Objective:** Big Data Analytics in Stock Market Prediction.
- **Stock Universe:** Global tech leaders (`NVDA`, `AAPL`, `MSFT`) and Indian Blue-Chips on the National Stock Exchange (`RELIANCE.NS`, `TCS.NS`).
- **Core Workflow:**
  $$\text{Stock Selection} \to \text{Ingestion} \to \text{PySpark Validation} \to \text{Feature Engineering} \to \text{Parquet Sink} \to \text{Chronological ML} \to \text{Prediction Dashboard}$$

---

## 📱 Focused 4-Page Application Architecture

The application is structured into four focused pages:

### 1. Prediction Dashboard (Primary Landing Page)
- **Stock Selector:** Instantly switches between `NVDA`, `AAPL`, `MSFT`, `RELIANCE.NS`, and `TCS.NS`.
- **Data Source Indicator:** Transparently displays whether data is `Live Market Data (Yahoo Finance API)`, `Cached Ingestion`, or `Offline Sample Dataset`.
- **Latest Closing Price:** Prominent real-time closing price with day delta and market identification.
- **Historical Price Chart:** Interactive Plotly chart with 20-day and 50-day moving averages.
- **Prediction Controls:** Horizon selector (1, 5, 10, 15, 30 days) and Model selector.
- **Dominant Prediction Card:** Displays predicted future price, expected price movement ($), and expected percentage change (%).
- **Out-of-Sample Performance Chart:** Visualizes ground truth actual prices versus model forecasts over the chronological test period.
- **Prediction Summary & Feature Weights:** Feature contribution bar chart explaining the inputs behind the model.

### 2. Data Processing (Big Data PySpark Pipeline)
- **5-Stage Visual Workflow:** Ingestion $\to$ Schema Validation $\to$ Missing-Value Imputation $\to$ IQR Outlier Detection $\to$ Window Feature Engineering $\to$ Snappy Parquet Storage.
- **Cluster Telemetry:** Input record count, output record count, duplicates removed, missing values handled, partitions (4), and execution duration.
- **Documented Outlier Treatment:** IQR filtering flags anomalies in the `Outlier_Flag` column without silently deleting genuine extreme market days, preserving original price history.
- **Engine Transparency:** Explicitly labels `Apache PySpark (Distributed Local)` or `Pandas Fallback`.
- **Engineered Features:** Lags ($t-1, t-2, t-3, t-5$), moving averages ($SMA_5, SMA_{10}, SMA_{20}, SMA_{50}$), daily returns, and 20-day rolling volatility.
- **Before & After Previews:** Compares raw ingested schema against the processed feature-rich Parquet dataset.

### 3. Model Evaluation (Out-of-Sample Benchmarking)
- **Chronological 80/20 Partition:** Strictly enforces time-ordered splitting to prevent lookahead data leakage.
- **Models Evaluated:** Linear Regression (Statistical Baseline), Random Forest Regressor (Ensemble Bagging), and Gradient Boosting Regressor (Ensemble Boosting).
- **Naive Persistence Baseline:** Benchmarks every model against $Close_t = Close_{t-1}$.
- **Metrics Evaluated:** Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), $R^2$ Score, and Directional Accuracy (%).
- **Best Model Identification:** Dynamically selects the top performer based on out-of-sample test results (never training performance).
- **Actual vs. Multi-Model Overlay Chart:** Overlays all models and the baseline on the test set.

### 4. Project Information (Academic ABL Dossier)
- Problem statement and proposed solution.
- System architecture diagram.
- 5Vs of Big Data contextualized for financial markets (Volume, Velocity, Variety, Veracity, Value).
- Role of Apache Spark (Catalyst optimizer, distributed window aggregations, out-of-core scalability).
- Role of Machine Learning (capturing nonlinear interactions between momentum, moving averages, and volatility).
- Academic limitations and future scope.

---

## 🌐 The 5Vs of Big Data in Stock Market Prediction

| Big Data 'V' | Implementation in StockSense |
| :--- | :--- |
| **Volume** | Multi-year historical trade records; PySpark partitions computation across distributed workers, preventing single-node RAM saturation. |
| **Velocity** | Live API feeds via Yahoo Finance with on-demand refresh and caching. |
| **Variety** | Multi-modal ingestion: Snappy columnar Parquet, structured CSV, and nested market arrays. |
| **Veracity** | Duplicate timestamp elimination, forward/backward null imputation, and IQR anomaly tagging. |
| **Value** | Probabilistic future price estimates, directional movement conviction, and empirical test benchmarking. |

---

## 🛠️ Tech Stack & Hardware Compatibility

- **Python:** 3.12
- **Distributed Engine:** Apache PySpark 3.5.9 (configured with `local[2]`, 1GB driver memory, 4 shuffle partitions for 4GB RAM laptop compatibility)
- **Storage:** Apache Parquet (Snappy compressed columnar format in `data/processed/`)
- **Machine Learning:** Scikit-Learn (Linear Regression, Random Forest, Gradient Boosting)
- **Frontend / UI:** Streamlit with custom institutional dark terminal CSS
- **Visualizations:** Plotly Graph Objects

---

## 🚀 Quickstart & Application Launch

### 1. Launch the Streamlit Dashboard
```bash
streamlit run dashboard/app.py --server.port 8501
```
Open **`http://localhost:8501`** in your browser.

### 2. Run the Automated Verification Suite
```bash
python tests/test_full_app.py
```

---

## 📂 Repository Structure

```
stock-analysis-dashboard/
├── dashboard/
│   └── app.py                  # Focused 4-Page Streamlit Financial Terminal
├── src/
│   ├── spark_engine.py         # PySpark 3.5 Distributed ETL & Feature Engineering
│   ├── ml_predictor.py         # Chronological ML Estimators & Naive Baseline
│   └── data_fetcher.py         # Resilient Live API & Offline Parquet Ingestion
├── data/
│   ├── sample/                 # 10 Pre-cached Offline Equities (.parquet & .csv)
│   └── processed/              # Parquet Output from PySpark Pipeline
├── tests/
│   └── test_full_app.py        # Automated 5-stage end-to-end verification suite
├── PRESENTATION_GUIDE.md       # 5-Mark ABL presentation pitch & viva cheat sheet
├── BLUEPRINTS.md               # Complete technical blueprints & element catalog
└── requirements.txt            # Python dependencies
```
