#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

export PATH="/Applications/Docker.app/Contents/Resources/bin:$PATH"

echo "=========================================================="
echo " Stopping Apache Spark 3-Node Cluster                     "
echo "=========================================================="

cd "$PROJECT_ROOT"
docker compose down

echo "Cluster stopped successfully."

