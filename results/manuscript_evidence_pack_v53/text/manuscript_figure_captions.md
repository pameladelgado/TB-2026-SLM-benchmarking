# Manuscript figure captions

**Figure 1.** Overall architecture of the benchmarking framework. Experiments are defined by configuration files and executed through a Python orchestration layer. Platform-specific vLLM and llama.cpp backends expose an OpenAI-compatible streaming API. Task-level and inline system-level metrics are recorded and exported as traceable structured artifacts.

**Figure 2.** Color-blind-safe grouped bars report peak system RAM on the Raspberry Pi 5 for each feasible model configuration and workload. Models are ordered by increasing Dolly peak RAM, and workload bars are alphabetical. The 8 GiB reference line makes the edge-context ceiling explicit; Dolly uses the maximum peak across its split acquisitions.

**Figure 2a1.** Grouped bars report mean GPU energy per request for GPU model configurations on Dolly and GSM8K only. The split avoids mixing long-generation and short-answer regimes on the same joule scale. Request energy is obtained by trapezoidal integration of recorded GPU power over request time, with audited fallbacks. Models are ordered by increasing Dolly mean energy.

**Figure 2a2.** Grouped bars report mean GPU energy per request for GPU model configurations on HellaSwag and MMLU only. These short-answer workloads are separated from Dolly and GSM8K so the joule scale remains interpretable. Figure 2a2 keeps the same model order as Figure 2a1.

**Figure 3.** Color-blind-safe grouped bars report peak system RAM on the local CPU platform for each model configuration and workload. Models are ordered by increasing Dolly peak RAM, workload bars are alphabetical, and Dolly uses the maximum peak across its split acquisitions.

**Figure 3a.** Absolute GPU VRAM deployment footprint by model configuration and workload. Markers show the median recorded VRAM, thick vertical intervals show the 5th–95th percentiles, and thin vertical intervals show the observed minimum–maximum range. This representation remains readable when the interquartile range collapses. Models are ordered by increasing Dolly median VRAM; trace measurements are descriptive repeated observations within requests.

**Figure 3b1.** Grouped boxplots report the distribution of GPU energy per request for Dolly and GSM8K only. Separating these workloads avoids compressing the higher-energy regime into the short-answer regime. Figure 3b1 keeps the same increasing Dolly mean-energy model order as Figure 2a1.

**Figure 3b2.** Grouped boxplots report the distribution of GPU energy per request for HellaSwag and MMLU only. These short-answer workloads use a separate joule scale and the same model order as Figure 2a2.

**Figure 4.** Accuracy [0, 1] by model and benchmark, separated into GSM8K, HellaSwag, and MMLU panels. Accuracy is shown on the y-axis, and the complete model labels are repeated beneath every panel. Edge, Local CPU, and GPU-server configurations occupy fixed context blocks and follow the shared GSM8K-based order used in Figure 5.

**Figure 5.** Reported token throughput by model and workload, separated into Dolly, GSM8K, HellaSwag, and MMLU panels. The complete model labels are repeated beneath every panel. The shared logarithmic y-axis exposes the observed order-of-magnitude range, and context blocks, model order, and x-coordinate positions are identical to Figure 4. Edge and Local CPU report active-generation tokens/s, whereas the GPU server reports run-level sustained tokens/s; boxed values quantify the fastest-to-slowest spread within each benchmark and context.

**Figure 6.** GSM8K accuracy [0, 1]–latency Pareto frontiers within each deployment context. Every Pareto-relevant configuration is labeled; the GPU panel uses a dedicated right-side label column so annotations do not obscure the data.

**Figure 7.** GSM8K accuracy [0, 1]–memory Pareto frontiers within each deployment context. Every Pareto-relevant configuration is labeled. The GPU panel uses a dedicated right-side label column. Memory denotes peak system RAM for edge/local CPU and peak VRAM for the GPU server.
