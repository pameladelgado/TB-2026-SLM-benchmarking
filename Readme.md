
# Benchmarking Resource Efficiency in Small vs Large Language Models

This repository implements a **reproducible benchmarking engine (`runner/`)** for evaluating efficiency–performance trade-offs in large language models (LLMs).

The framework measures both:
- **Task-level performance** (accuracy on benchmarks)
- **System-level efficiency** (latency, throughput, resource usage)

It is designed to evaluate models under **realistic deployment conditions** across heterogeneous hardware platforms.

---

## Overview

Language models are typically compared using accuracy, but real-world deployment also depends on:

- latency (response time)
- throughput (tokens per second)
- memory usage
- hardware constraints

This project provides a framework to evaluate these aspects jointly across:

- GPU servers (vLLM)
- CPU systems (laptop)
- Edge devices (Raspberry Pi)

---

## Core Component: `runner/`

The `runner/` directory contains the core benchmarking engine of this project.

It is responsible for:
- orchestrating benchmark execution
- sending inference requests to models
- collecting performance and system-level metrics
- handling datasets and evaluation logic

### Main modules

- `benchmark.py`  
  Benchmarking pipeline for CPU / llama.cpp environments  

- `benchmark_vllm.py`  
  Benchmarking pipeline for GPU execution with vLLM  

- `dataset_loader.py` / `dataset_loader_vllm.py`  
  Dataset loading and formatting (MMLU, GSM8K, HellaSwag, Dolly)  

- `system_metrics.py` / `system_metrics_vllm.py`  
  Inline collection of:
  - latency (TTFT, end-to-end)
  - throughput (tokens/sec)
  - CPU / GPU utilization
  - memory usage  

### Key design principles

- Metrics collected **inline during inference**
- Support for **streaming generation**
- **Run-level isolation** (fresh server per run)
- **Multi-run aggregation** for stability
- Designed for **reproducibility**

---

## Repository Structure

```

configs/        # YAML experiment configurations
docker/         # Docker setup
docs/           # Figures and exported notebooks
figures/        # Final visualization-ready figures
notebooks/      # Analysis and visualization
results/        # Raw and processed results
runner/         # Core benchmarking engine
tables_report/  # Final tables for the report

````

---

## How It Works

Each benchmark run follows a controlled pipeline:

1. Start inference server (vLLM or llama.cpp)
2. Run warm-up phase
3. Execute benchmark samples
4. Collect metrics during inference:
   - Time to First Token (TTFT)
   - End-to-end latency
   - Token throughput
   - CPU / GPU / memory usage
5. Store results (CSV + logs)

Each configuration is repeated multiple times to ensure stability.

---

## Requirements

This project depends on external inference backends. You must install **at least one** depending on your setup.

---

### Option 1 — GPU (vLLM)

Used for high-performance inference.

Install:

```bash
pip install vllm
````

Requirements:

* NVIDIA GPU
* CUDA installed
* Working drivers (`nvidia-smi`)

Optional:

```bash
pip install openai
```

---

### Option 2 — CPU / Edge (llama.cpp)

Used for CPU and Raspberry Pi experiments.

Install:

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
make
```

Ensure the `llama-server` binary is available.

---

### Python dependencies

```bash
pip install -r requirements.txt
```

---

## Configuration

Experiments are defined in YAML files:

* `configs/vllm.yaml`
* `configs/llama_cpp_container.yaml`
* `configs/pi.yaml`

They specify:

* model
* dataset
* generation parameters
* number of runs

---

## Running the Benchmark

### CPU / GPU

```bash
./run_benchmark.sh
```

### Raspberry Pi

```bash
./run_pi.sh
```

---

## Results

Results are stored in:

```
results/
├── cpu/
├── pi5/
├── vllm/
├── processed/
└── reports/
```

Final report tables:

```
tables_report/
```

---

## Analysis & Visualization

Jupyter notebooks in `notebooks/` are used to:

* analyze results
* generate figures
* compare platforms

Final figures are available in:

```
figures/visualization_ready/
```


---

## Notes

* vLLM is used for GPU experiments
* llama.cpp is used for CPU and edge
* Metrics are collected **during inference**, not externally

---

## Author

Stéphane Nascimento Santos
Bachelor Thesis – HEIG-VD

