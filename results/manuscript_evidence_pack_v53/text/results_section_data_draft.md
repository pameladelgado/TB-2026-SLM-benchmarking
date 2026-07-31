### Edge context

The evaluated set contains **4 context-specific model configurations**.
Observed peak system RAM spans **1.92–7.50 GiB** across workloads and configurations.
On MMLU, **Phi-4 Mini 3.8B (Q4)** has the highest observed accuracy (0.564), while **Gemma 3 1B (Q8)** has the lowest mean end-to-end latency (1.57 s).
On GSM8K, **Phi-4 Mini 3.8B (Q4)** has the highest observed accuracy (0.639), while **Llama 3.2 1B (Q4)** has the lowest mean end-to-end latency (16.01 s).
On HellaSwag, **Phi-4 Mini 3.8B (Q4)** has the highest observed accuracy (0.489), while **Gemma 3 1B (Q8)** has the lowest mean end-to-end latency (2.06 s).

### Local CPU context

The evaluated set contains **4 context-specific model configurations**.
Observed peak system RAM spans **5.13–14.30 GiB** across workloads and configurations.
On MMLU, **Ministral 3 3B (Q4)** has the highest observed accuracy (0.630), while **Gemma 3 1B (Q8)** has the lowest mean end-to-end latency (1.13 s).
On GSM8K, **Phi-4 Mini 3.8B (Q4)** has the highest observed accuracy (0.640), while **Llama 3.2 1B (Q8)** has the lowest mean end-to-end latency (3.22 s).
On HellaSwag, **Phi-4 Mini 3.8B (Q4)** has the highest observed accuracy (0.510), while **Gemma 3 1B (Q8)** has the lowest mean end-to-end latency (1.55 s).

### GPU server context

The evaluated set contains **15 context-specific model configurations**.
Observed peak VRAM spans **85.75–90.43 GiB** across workloads and configurations.
On MMLU, **Phi-4 14B (bf16)** has the highest observed accuracy (0.820), while **Llama 3.2 1B (bf16)** has the lowest mean end-to-end latency (0.24 s).
On GSM8K, **Llama 3.3 70B (FP8)** has the highest observed accuracy (0.745), while **Llama 3.2 1B (bf16)** has the lowest mean end-to-end latency (1.10 s).
On HellaSwag, **Phi-4 14B (bf16)** has the highest observed accuracy (0.850), while **Llama 3.2 1B (bf16)** has the lowest mean end-to-end latency (0.24 s).

### GPU timing interpretation

Across the GPU configurations and workloads represented in the normalized table, mean TTFT spans **128.7–224.1 ms**. End-to-end latency and sustained generation throughput should therefore carry more weight than TTFT when outputs differ substantially in length.

On the GPU server, Dolly run-level sustained throughput ranges from **11.5** tokens/s for **Ministral 3 14B** to **98.2** tokens/s for **Llama 3.2 1B (bf16)**. Edge and Local CPU throughput is reported separately as active-generation tokens per second and should not be interpreted as the same serving-level denominator.
