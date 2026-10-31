"""
Case Correlation Analysis using Apache Spark.
Performs:
1. Multi-variable correlation matrix across demographic, economic, and epidemiological metrics.
2. Time-lagged cross-correlation between daily cases and daily deaths (estimating mortality latency).
3. Vaccination rate vs. mortality & case fatality rate (CFR) correlation.
"""

import logging
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from src.config import get_spark_session, PARQUET_PATH, OUTPUT_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def analyze_correlations(spark=None):
    if spark is None:
        spark = get_spark_session("COVID19-CorrelationAnalysis")

    logger.info(f"Loading cleaned data from {PARQUET_PATH}...")
    df = spark.read.parquet(str(PARQUET_PATH))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # 1. Multi-Variable Country-Level Summary (Max/Latest available metrics)
    # =========================================================================
    logger.info("Computing Country-Level Demographic & Epidemiological Aggregates...")
    country_summary = (
        df.groupBy("location")
        .agg(
            F.max("total_cases_per_million").alias("total_cases_per_million"),
            F.max("total_deaths_per_million").alias("total_deaths_per_million"),
            F.max("people_fully_vaccinated_per_hundred").alias("people_fully_vaccinated_per_hundred"),
            F.first("median_age", ignorenulls=True).alias("median_age"),
            F.first("gdp_per_capita", ignorenulls=True).alias("gdp_per_capita"),
            F.first("population_density", ignorenulls=True).alias("population_density"),
            F.first("hospital_beds_per_thousand", ignorenulls=True).alias("hospital_beds_per_thousand"),
            F.first("diabetes_prevalence", ignorenulls=True).alias("diabetes_prevalence"),
            F.first("cardiovasc_death_rate", ignorenulls=True).alias("cardiovasc_death_rate"),
            F.first("human_development_index", ignorenulls=True).alias("human_development_index")
        )
    )

    feature_cols = [
        "total_cases_per_million",
        "total_deaths_per_million",
        "people_fully_vaccinated_per_hundred",
        "median_age",
        "gdp_per_capita",
        "population_density",
        "hospital_beds_per_thousand",
        "diabetes_prevalence",
        "cardiovasc_death_rate",
        "human_development_index"
    ]

    # Calculate Pearson Correlation pairwise using Spark's stat.corr
    corr_matrix = {}
    for col1 in feature_cols:
        corr_matrix[col1] = {}
        for col2 in feature_cols:
            if col1 == col2:
                corr_matrix[col1][col2] = 1.0
            else:
                c = country_summary.stat.corr(col1, col2, method="pearson")
                corr_matrix[col1][col2] = round(c, 4) if c is not None else None

    corr_df = pd.DataFrame(corr_matrix)
    logger.info("Country-level Correlation Matrix calculated:")
    logger.info("\n" + str(corr_df.round(3)))
    corr_df.to_csv(OUTPUT_DIR / "correlation_matrix.csv")

    # =========================================================================
    # 2. Time-Lagged Cross-Correlation: Daily Cases vs. Daily Deaths
    # =========================================================================
    logger.info("Computing Time-Lagged Cross-Correlation (Cases vs. Deaths latency)...")
    window_loc_date = Window.partitionBy("location").orderBy("date")

    # Test lag shifts from 0 to 35 days (i.e., correlate Cases(t) with Deaths(t + lag))
    lags = [0, 7, 14, 21, 28, 35]
    lagged_df = df.filter(F.col("new_cases").isNotNull() & F.col("new_deaths").isNotNull())

    for lag_days in lags:
        lagged_df = lagged_df.withColumn(f"future_deaths_{lag_days}d", F.lead("new_deaths", lag_days).over(window_loc_date))

    lag_correlations = []
    for lag_days in lags:
        col_target = f"future_deaths_{lag_days}d"
        r = lagged_df.stat.corr("new_cases", col_target, method="pearson")
        lag_correlations.append({"lag_days": lag_days, "pearson_r": round(r, 4) if r is not None else None})

    lag_results_df = pd.DataFrame(lag_correlations)
    logger.info(f"Time-Lag Correlation Results:\n{lag_results_df}")
    lag_results_df.to_csv(OUTPUT_DIR / "lagged_correlation.csv", index=False)

    # =========================================================================
    # 3. Vaccination Rate vs. Case Fatality Rate (CFR)
    # =========================================================================
    logger.info("Computing Vaccination Rate vs Case Fatality Rate (CFR)...")
    cfr_vax_df = (
        country_summary
        .filter(
            (F.col("total_cases_per_million") > 1000) &
            (F.col("people_fully_vaccinated_per_hundred").isNotNull()) &
            (F.col("total_deaths_per_million").isNotNull())
        )
        .withColumn(
            "case_fatality_ratio_pct",
            F.round((F.col("total_deaths_per_million") / F.col("total_cases_per_million")) * 100, 3)
        )
    )

    vax_cfr_r = cfr_vax_df.stat.corr("people_fully_vaccinated_per_hundred", "case_fatality_ratio_pct", method="pearson")
    logger.info(f"Correlation between Full Vaccination % and Case Fatality Rate: r = {vax_cfr_r:.4f}")

    vax_cfr_pd = cfr_vax_df.toPandas()
    vax_cfr_pd.to_csv(OUTPUT_DIR / "vaccination_vs_cfr.csv", index=False)

    logger.info("Correlation analysis completed successfully.")
    return corr_df, lag_results_df, vax_cfr_pd

if __name__ == "__main__":
    analyze_correlations()
