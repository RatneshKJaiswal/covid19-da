"""
Data Ingestion Module.
Downloads the Our World in Data (OWID) COVID-19 complete dataset.
"""

import os
import urllib.request
import logging
from src.config import OWID_COVID_CSV_URL, RAW_CSV_PATH, RAW_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def download_dataset(force: bool = False) -> str:
    """
    Downloads the COVID-19 dataset if not already present.
    """
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    if RAW_CSV_PATH.exists() and not force:
        logger.info(f"Dataset already exists at: {RAW_CSV_PATH} ({RAW_CSV_PATH.stat().st_size / (1024*1024):.2f} MB)")
        return str(RAW_CSV_PATH)

    logger.info(f"Downloading COVID-19 dataset from {OWID_COVID_CSV_URL}...")
    urllib.request.urlretrieve(OWID_COVID_CSV_URL, RAW_CSV_PATH)
    logger.info(f"Downloaded successfully to {RAW_CSV_PATH} ({RAW_CSV_PATH.stat().st_size / (1024*1024):.2f} MB)")
    return str(RAW_CSV_PATH)

if __name__ == "__main__":
    download_dataset()

