#!/usr/bin/env bash
set -euo pipefail

cd /app

BENCH_CONFIG="${BENCH_CONFIG:-configs/llama_cpp_container.yaml}"

echo "Checking benchmark config: $BENCH_CONFIG"
if [[ ! -f "$BENCH_CONFIG" ]]; then
  echo "Benchmark config not found: $BENCH_CONFIG"
  exit 1
fi

echo "Checking llama-server binary..."
if [[ ! -x "/app/llama-server" ]]; then
  echo "llama-server not executable: /app/llama-server"
  command -v llama-server || true
  exit 1
fi

echo "Checking for stale llama-server..."
if pgrep -af "llama-server" >/dev/null 2>&1; then
  echo "Existing llama-server process found:"
  pgrep -af "llama-server"
  exit 1
fi

echo "Running benchmark..."
python3 runner/benchmark.py --config "$BENCH_CONFIG"