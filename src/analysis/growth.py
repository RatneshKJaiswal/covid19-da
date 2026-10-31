"""
Regional Growth Pattern Analysis using Apache Spark.
Computes rolling averages, growth factors, wave peaks, and regional aggregations.
"""

import logging
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from src.config import get_spark_session, PARQUET_PATH, OUTPUT_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def analyze_growth_patterns(spark=None):
    if spark is None:
        spark = get_spark_session("COVID19-GrowthAnalysis")

    logger.info(f"Loading cleaned data from {PARQUET_PATH}...")
    df = spark.read.parquet(str(PARQUET_PATH))

    # Window partitioned by location ordered by date
    window_loc_date = Window.partitionBy("location").orderBy("date")
    window_7d = Window.partitionBy("location").orderBy("date").rowsBetween(-6, 0)

    logger.info("Computing 7-day rolling averages and daily growth factors...")
    growth_df = (
        df
        .withColumn("rolling_cases_7d", F.avg("new_cases").over(window_7d))
        .withColumn("rolling_deaths_7d", F.avg("new_deaths").over(window_7d))
        .withColumn("prev_cases_7d", F.lag("rolling_cases_7d", 1).over(window_loc_date))
        # Growth factor = Today's 7d rolling / Yesterday's 7d rolling
        .withColumn(
            "growth_factor",
            F.when((F.col("prev_cases_7d") > 0) & (F.col("rolling_cases_7d").isNotNull()),
                   F.round(F.col("rolling_cases_7d") / F.col("prev_cases_7d"), 4))
            .otherwise(None)
        )
        .withColumn(
            "growth_phase",
            F.when(F.col("growth_factor") > 1.05, "Exponential Growth")
             .when((F.col("growth_factor") >= 0.95) & (F.col("growth_factor") <= 1.05), "Plateau / Stable")
             .when(F.col("growth_factor") < 0.95, "Decline / Controlled")
             .otherwise("Unknown")
        )
    )

    # Continent-level daily time series aggregation
    logger.info("Aggregating daily time series by continent...")
    continent_ts = (
        growth_df.groupBy("continent", "date")
        .agg(
            F.sum("new_cases").alias("daily_cases"),
            F.sum("new_deaths").alias("daily_deaths"),
            F.avg("rolling_cases_7d").alias("continent_rolling_cases_7d"),
            F.avg("rolling_deaths_7d").alias("continent_rolling_deaths_7d")
        )
        .orderBy("continent", "date")
    )

    # Peak infection analysis per country
    logger.info("Identifying peak infection dates and max cases per country...")
    peak_window = Window.partitionBy("location").orderBy(F.col("new_cases").desc_nulls_last())
    country_peaks = (
        growth_df
        .withColumn("rank", F.row_number().over(peak_window))
        .filter(F.col("rank") == 1)
        .select(
            "continent",
            "location",
            F.col("date").alias("peak_date"),
            F.col("new_cases").alias("peak_daily_cases"),
            F.col("rolling_cases_7d").alias("peak_rolling_cases"),
            "total_cases",
            "total_deaths",
            "population"
        )
        .withColumn("cases_per_100k_pop", F.round((F.col("total_cases") / F.col("population")) * 100000, 2))
        .orderBy(F.col("peak_daily_cases").desc_nulls_last())
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Export summarized data to CSV for visualization / reporting
    logger.info(f"Writing growth patterns and continent time-series to {OUTPUT_DIR}...")
    country_peaks.toPandas().to_csv(OUTPUT_DIR / "country_infection_peaks.csv", index=False)
    continent_ts.toPandas().to_csv(OUTPUT_DIR / "continent_growth_timeseries.csv", index=False)

    logger.info("Growth pattern analysis completed successfully.")
    return growth_df, country_peaks, continent_ts

if __name__ == "__main__":
    analyze_growth_patterns()

