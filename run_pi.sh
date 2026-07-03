#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
source .venv/bin/activate

export LD_LIBRARY_PATH="$PWD/engines/llama.cpp/build/bin:$PWD/engines/llama.cpp/build/src:${LD_LIBRARY_PATH:-}"

python runner/benchmark.py --config configs/llama_cpp.yaml