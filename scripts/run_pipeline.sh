#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

export PATH="/Applications/Docker.app/Contents/Resources/bin:$PATH"

echo "================================================================="
echo " Executing COVID-19 Analytics on 3-Node Spark Cluster           "
echo " (1 Master Node + 2 Worker Nodes)                                "
echo "================================================================="

echo "[1/3] Running Data Cleaning & Parquet Ingestion..."
docker exec -i \
  -e PYTHONPATH=/opt/workspace \
  -e SPARK_MASTER_URL=spark://spark-master:7077 \
  spark-master /opt/spark/bin/spark-submit \
  --master spark://spark-master:7077 \
  --conf spark.driver.host=spark-master \
  --conf spark.driver.bindAddress=0.0.0.0 \
  --total-executor-cores 4 \
  --executor-memory 1G \
  /opt/workspace/src/etl/clean.py

echo "[2/3] Running Regional Growth Pattern Analysis..."
docker exec -i \
  -e PYTHONPATH=/opt/workspace \
  -e SPARK_MASTER_URL=spark://spark-master:7077 \
  spark-master /opt/spark/bin/spark-submit \
  --master spark://spark-master:7077 \
  --conf spark.driver.host=spark-master \
  --conf spark.driver.bindAddress=0.0.0.0 \
  --total-executor-cores 4 \
  --executor-memory 1G \
  /opt/workspace/src/analysis/growth.py

echo "[3/3] Running Case Correlation Engine..."
docker exec -i \
  -e PYTHONPATH=/opt/workspace \
  -e SPARK_MASTER_URL=spark://spark-master:7077 \
  spark-master /opt/spark/bin/spark-submit \
  --master spark://spark-master:7077 \
  --conf spark.driver.host=spark-master \
  --conf spark.driver.bindAddress=0.0.0.0 \
  --total-executor-cores 4 \
  --executor-memory 1G \
  /opt/workspace/src/analysis/correlations.py

echo "================================================================="
echo " Cluster pipeline execution completed successfully.             "
echo " View details on the Spark Master Web UI: http://localhost:8080 "
echo "================================================================="
