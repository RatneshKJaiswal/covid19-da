#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

export PATH="/Applications/Docker.app/Contents/Resources/bin:$PATH"

echo "=========================================================="
echo " Starting Apache Spark 3-Node Cluster (1 Master, 2 Workers)"
echo "=========================================================="

cd "$PROJECT_ROOT"
docker compose up -d

echo ""
echo "=========================================================="
echo " Cluster is running!"
echo " • Master Web UI:  http://localhost:8080"
echo " • Worker 1 UI:    http://localhost:8081"
echo " • Worker 2 UI:    http://localhost:8082"
echo " • Master RPC URL: spark://localhost:7077"
echo "=========================================================="

