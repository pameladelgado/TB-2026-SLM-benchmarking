#!/usr/bin/env bash

set -euo pipefail

CONFIG="configs/llama_cpp.yaml"
SERVER_BIN="engines/llama.cpp/build/bin/llama-server"
PORT=8000

MODEL_PATH="$(
python -c '
import sys, yaml
with open(sys.argv[1], "r", encoding="utf-8") as f:
    print(yaml.safe_load(f)["model"]["path"])
' "$CONFIG"
)"

export LD_LIBRARY_PATH="engines/llama.cpp/build/bin:engines/llama.cpp/build/lib:${LD_LIBRARY_PATH:-}"

echo "Starting llama.cpp server..."

[ -x "$SERVER_BIN" ] || { echo "Server binary not found or not executable: $SERVER_BIN"; exit 1; }
[ -f "$MODEL_PATH" ] || { echo "Model file not found: $MODEL_PATH"; exit 1; }

"$SERVER_BIN" \
  -m "$MODEL_PATH" \
  -c 8192 \
  --port "$PORT" \
  > server.log 2>&1 &

SERVER_PID=$!

echo "Server PID: $SERVER_PID"

cleanup() {
  echo "Stopping server..."
  kill "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for server to be ready..."

for i in $(seq 1 60); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Server process exited early"
    echo "---- server.log ----"
    cat server.log
    exit 1
  fi

  if curl -s "http://127.0.0.1:$PORT/v1/models" > /dev/null; then
    echo "Server is ready"
    break
  fi

  sleep 1

  if [ "$i" -eq 60 ]; then
    echo "Server failed to start"
    echo "---- server.log ----"
    cat server.log
    exit 1
  fi
done

echo "Running benchmark..."
python runner/benchmark.py --config "$CONFIG"
echo "Benchmark finished"