#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
source .venv/bin/activate

export LD_LIBRARY_PATH="$PWD/engines/llama.cpp/build/bin:$PWD/engines/llama.cpp/build/src:${LD_LIBRARY_PATH:-}"

CONFIG="${1:-configs/llama_cpp.yaml}"
python runner/benchmark.py --config "$CONFIG"