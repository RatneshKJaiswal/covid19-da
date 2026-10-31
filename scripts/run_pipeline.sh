#!/usr/bin/env bash
set -e

# Setup directories and paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python3"

echo "============================================================"
echo " Starting COVID-19 Big Data Processing Pipeline (PySpark)  "
echo "============================================================"

# Ensure JAVA_HOME points to installed JDK
export JAVA_HOME="/Library/Java/JavaVirtualMachines/jdk-27.jdk/Contents/Home"
export PATH="$JAVA_HOME/bin:$PATH"
export SPARK_LOCAL_IP="127.0.0.1"

cd "$PROJECT_ROOT"

echo "[1/4] Ingesting dataset..."
"$VENV_PYTHON" -m src.etl.ingest

echo "[2/4] Cleaning dataset and saving to partitioned Parquet..."
"$VENV_PYTHON" -m src.etl.clean

echo "[3/4] Running Spark regional growth pattern analysis..."
"$VENV_PYTHON" -m src.analysis.growth

echo "[4/4] Running Spark case correlation analysis..."
"$VENV_PYTHON" -m src.analysis.correlations

echo "[Bonus] Generating analytical charts..."
"$VENV_PYTHON" -m src.visualization.visualize

echo "============================================================"
echo " Pipeline execution completed successfully!                 "
echo " Results and plots saved to data/output/                    "
echo "============================================================"

