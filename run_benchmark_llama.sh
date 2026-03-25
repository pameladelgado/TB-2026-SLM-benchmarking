#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/llama_cpp.yaml}"
python runner/benchmark.py --config "$CONFIG"