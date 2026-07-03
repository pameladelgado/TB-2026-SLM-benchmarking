#!/usr/bin/env bash

set -eu

CONFIG="configs/vllm.yaml"

echo "Loading vLLM settings from $CONFIG ..."

MODEL_NAME=$(python - <<'PY'
import yaml
with open("configs/vllm.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
print(cfg["vllm"]["model"])
PY
)

HOST=$(python - <<'PY'
import yaml
with open("configs/vllm.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
print(cfg["vllm"].get("host", "127.0.0.1"))
PY
)

PORT=$(python - <<'PY'
import yaml
with open("configs/vllm.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
print(cfg["vllm"].get("port", 8000))
PY
)

API_KEY=$(python - <<'PY'
import yaml
with open("configs/vllm.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
print(cfg["vllm"].get("api_key", "dummy"))
PY
)

DTYPE=$(python - <<'PY'
import yaml
with open("configs/vllm.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
print(cfg["vllm"].get("dtype", "auto"))
PY
)

GPU_MEMORY_UTILIZATION=$(python - <<'PY'
import yaml
with open("configs/vllm.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
print(cfg["vllm"].get("gpu_memory_utilization", 0.9))
PY
)

MAX_MODEL_LEN=$(python - <<'PY'
import yaml
with open("configs/vllm.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
print(cfg["vllm"].get("max_model_len", 8192))
PY
)

TRUST_REMOTE_CODE=$(python - <<'PY'
import yaml
with open("configs/vllm.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
print(str(cfg["vllm"].get("trust_remote_code", False)).lower())
PY
)

echo "Starting vLLM server..."
echo "Model: $MODEL_NAME"
echo "Host:  $HOST"
echo "Port:  $PORT"

CMD=(
  vllm serve "$MODEL_NAME"
  --host "$HOST"
  --port "$PORT"
  --api-key "$API_KEY"
  --dtype "$DTYPE"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-model-len "$MAX_MODEL_LEN"
)

if [ "$TRUST_REMOTE_CODE" = "true" ]; then
  CMD+=(--trust-remote-code)
fi

"${CMD[@]}" > server.log 2>&1 &
SERVER_PID=$!

echo "Server PID: $SERVER_PID"

cleanup() {
  echo "Stopping server..."
  kill "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for server to be ready..."

READY=0
for i in $(seq 1 120); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Server process exited early"
    echo "---- server.log ----"
    cat server.log
    exit 1
  fi

  if curl -s "http://$HOST:$PORT/v1/models" > /dev/null; then
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