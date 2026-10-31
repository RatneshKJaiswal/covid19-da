import os
import sys
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "output"

# Dataset URL (Our World in Data COVID-19 dataset)
OWID_COVID_CSV_URL = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"
RAW_CSV_PATH = RAW_DATA_DIR / "owid-covid-data.csv"
PARQUET_PATH = PROCESSED_DATA_DIR / "covid_data_parquet"

# Environment configuration for macOS + Spark + JDK
DEFAULT_JAVA_HOME = "/Library/Java/JavaVirtualMachines/jdk-27.jdk/Contents/Home"
if "JAVA_HOME" not in os.environ and os.path.exists(DEFAULT_JAVA_HOME):
    os.environ["JAVA_HOME"] = DEFAULT_JAVA_HOME

if "JAVA_HOME" in os.environ:
    os.environ["PATH"] = f"{os.environ['JAVA_HOME']}/bin:{os.environ.get('PATH', '')}"

os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

def get_spark_session(app_name: str = "COVID19-Analysis"):
    """
    Creates or retrieves an Apache Spark session configured for local execution.
    """
    from pyspark.sql import SparkSession

    builder = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
    )
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark
