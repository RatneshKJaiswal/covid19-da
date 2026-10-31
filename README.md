# COVID-19 Big Data Analytics Platform (Apache Spark & HDFS/Parquet)

A distributed big data analysis platform built with **Apache Spark (PySpark)** and an interactive **Streamlit Web Application** to examine pandemic dynamics, regional infection waves, and multi-variable case correlations across global populations.

---

## 1. System Architecture

<p align="center">
  <img src="COVID-19%20Spark%20Analytics%20Pipeline-2026-10-31-134306.svg" alt="COVID-19 Spark Analytics Pipeline Architecture" width="100%">
</p>

---

## 2. Key Findings & Insights

1. **Vaccination Efficacy vs. Mortality**:
   * Calculated Pearson correlation: **$r = -0.4247$** between `people_fully_vaccinated_per_hundred` and `case_fatality_ratio_pct`.
   * Demonstrates a significant inverse correlation: countries with higher full-vaccination coverage experienced markedly lower mortality per confirmed case.

2. **Epidemiological Latency (Cross-Correlation)**:
   * Time-lagged cross-correlation between daily reported cases and daily reported deaths peaks at **Day 7 ($r = 0.3382$)** and **Day 14 ($r = 0.3083$)**, proving the statistical latency between case surges and mortality outcomes.

3. **Demographic & Healthcare Correlations**:
   * `median_age` strongly correlates with `human_development_index` ($r = 0.668$) and `gdp_per_capita` ($r = 0.452$).
   * Reported cases per million correlate moderately with GDP per capita ($r = 0.362$) and hospital beds ($r = 0.228$), reflecting higher diagnostic and testing infrastructure in wealthier nations.

---

## 3. Interactive Web Application Features

The Streamlit web application (`app.py`) provides 5 interactive analytical modules:

1. **🗺️ Spatial Distribution & Overview**:
   * Interactive **Plotly 3D/2D Choropleth World Map** with dynamic indicator selection (*Cases, Deaths, Cases/Million, Deaths/Million, Full Vaccination %, Case Fatality Rate*).
   * Sortable **Top 15 Countries Leaderboard** and summary data tables.

2. **📈 Regional Growth Dynamics & Waves**:
   * Multi-country comparative time-series trajectory (Linear vs. Logarithmic scales).
   * **Growth Factor ($G_t$) & Epidemic Phase Classifier**: Categorizes days into *Exponential Growth ($G_t > 1.05$)*, *Plateau ($0.95 \le G_t \le 1.05$)*, and *Decline ($G_t < 0.95$)*.
   * Single-day peak infection wave records across countries.

3. **🔬 Multi-Variable Case Correlations**:
   * Dynamic correlation matrix with customizable feature selection (Pearson or Spearman).
   * Bivariate scatter plots with **OLS Regression Trendlines** and automated correlation $r$ calculation.

4. **⏱️ Epidemiological Latency & Time-Lag Explorer**:
   * Cross-correlation time-shift slider ($\tau \in [0, 45]$ days).
   * Analyzes how daily cases anticipate future daily deaths for selected countries or global aggregates.

5. **⚡ Big Data & Spark Engine Diagnostics**:
   * HDFS/Parquet partition breakdown (`continent=Asia`, `continent=Europe`, etc.).
   * On-demand **"Re-Run PySpark Big Data Pipeline"** button with live execution logs.
   * CSV download buttons for all analyzed extracts.

---

## 4. How to Launch the Web Application

### Step 1: Activate Virtual Environment
```bash
source .venv/bin/activate
```

### Step 2: Start Streamlit
```bash
streamlit run app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser.

---

## 5. Running the Batch PySpark Pipeline (CLI)

You can also run the backend pipeline in terminal mode:
```bash
./scripts/run_pipeline.sh
```
Or modularly:
```bash
python3 -m src.etl.ingest               # 1. Download dataset
python3 -m src.etl.clean                # 2. Spark Parquet conversion
python3 -m src.analysis.growth          # 3. Compute growth patterns
python3 -m src.analysis.correlations    # 4. Compute correlation matrix & time-lags
python3 -m src.visualization.visualize  # 5. Generate PNG charts
```
