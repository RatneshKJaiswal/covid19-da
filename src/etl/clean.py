"""
Data Cleaning and Transformation Module using Apache Spark.
Converts raw CSV to optimized, partitioned Parquet format and handles missing/invalid records.
"""

import logging
from pyspark.sql import functions as F
from pyspark.sql.types import DateType, DoubleType
from src.config import get_spark_session, RAW_CSV_PATH, PARQUET_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def clean_and_transform(spark=None):
    if spark is None:
        spark = get_spark_session("COVID19-DataCleaning")

    logger.info(f"Reading raw COVID data from {RAW_CSV_PATH}...")
    df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(str(RAW_CSV_PATH))
    )

    total_records = df.count()
    logger.info(f"Loaded {total_records:,} raw records.")

    # 1. Cast date to proper DateType
    df = df.withColumn("date", F.to_date(F.col("date"), "yyyy-MM-dd"))

    # 2. Key numeric columns to ensure proper Double casting
    numeric_cols = [
        "total_cases", "new_cases", "new_cases_smoothed",
        "total_deaths", "new_deaths", "new_deaths_smoothed",
        "total_cases_per_million", "new_cases_per_million",
        "total_deaths_per_million", "new_deaths_per_million",
        "reproduction_rate", "icu_patients", "hosp_patients",
        "total_tests", "new_tests", "positive_rate", "tests_per_case",
        "total_vaccinations", "people_vaccinated", "people_fully_vaccinated",
        "total_boosters", "new_vaccinations",
        "stringency_index", "population", "population_density",
        "median_age", "aged_65_older", "aged_70_older", "gdp_per_capita",
        "cardiovasc_death_rate", "diabetes_prevalence", "hospital_beds_per_thousand",
        "life_expectancy", "human_development_index"
    ]

    for col_name in numeric_cols:
        if col_name in df.columns:
            # Negative daily reports are retrospective corrections; floor them at 0 for growth analysis
            if col_name.startswith("new_"):
                df = df.withColumn(col_name, F.when(F.col(col_name).cast(DoubleType()) < 0, 0.0)
                                              .otherwise(F.col(col_name).cast(DoubleType())))
            else:
                df = df.withColumn(col_name, F.col(col_name).cast(DoubleType()))

    # 3. Filter for valid country-level records (where continent is not null)
    country_df = df.filter(F.col("continent").isNotNull() & (F.trim(F.col("continent")) != ""))

    logger.info(f"Filtered country-level records: {country_df.count():,}")

    # 4. Save to optimized columnar Parquet format partitioned by continent
    logger.info(f"Saving partitioned Parquet data to {PARQUET_PATH}...")
    (
        country_df.write
        .mode("overwrite")
        .partitionBy("continent")
        .parquet(str(PARQUET_PATH))
    )
    logger.info("Data successfully transformed and stored as Parquet.")
    return country_df

if __name__ == "__main__":
    clean_and_transform()

