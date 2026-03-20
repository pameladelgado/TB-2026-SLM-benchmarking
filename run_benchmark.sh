#!/usr/bin/env bash

set -u

CONFIG="configs/llama_cpp.yaml"
SERVER_BIN="engines/llama.cpp/build/bin/llama-server"
MODEL_PATH="models/Llama-3.2-1B-Instruct-Q8_0.gguf"
PORT=8000

# Add local build libraries for llama.cpp runtime linking
export LD_LIBRARY_PATH="engines/llama.cpp/build/bin:engines/llama.cpp/build/lib:${LD_LIBRARY_PATH:-}"

echo "Starting llama.cpp server..."

if [ ! -x "$SERVER_BIN" ]; then
  echo "Server binary not found or not executable: $SERVER_BIN"
  exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
  echo "Model file not found: $MODEL_PATH"
  exit 1
fi

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

READY=0
for i in $(seq 1 60); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Server process exited early"
    echo "---- server.log ----"
    cat server.log
    exit 1
  fi

  if curl -s "http://127.0.0.1:$PORT/v1/models" > /dev/null; then
    READY=1
    echo "Server is ready"
    break
  fi

  sleep 1
done

if [ "$READY" -ne 1 ]; then
  echo "Server failed to start"
  echo "---- server.log ----"
  cat server.log
  exit 1
fi

echo "Running benchmark..."
python runner/benchmark.py --config "$CONFIG"
echo "Benchmark finished"