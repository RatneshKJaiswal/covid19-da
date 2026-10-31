import os
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.conf import SparkConf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "output"

OWID_COVID_CSV_URL = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"
RAW_CSV_PATH = RAW_DATA_DIR / "owid-covid-data.csv"
PARQUET_PATH = PROCESSED_DATA_DIR / "covid_data_parquet"

def get_spark_session(app_name: str = "COVID19-Analysis"):
    """Initializes and returns an Apache Spark session configured for the 3-node cluster."""
    master_url = os.environ.get("SPARK_MASTER_URL") or SparkConf().get("spark.master", "spark://spark-master:7077")

    builder = (
        SparkSession.builder
        .appName(app_name)
        .master(master_url)
        .config("spark.executor.memory", "1g")
        .config("spark.executor.cores", "2")
        .config("spark.cores.max", "4")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.ui.showConsoleProgress", "false")
    )
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark
