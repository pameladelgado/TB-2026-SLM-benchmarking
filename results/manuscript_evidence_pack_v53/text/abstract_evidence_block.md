## Abstract evidence block

**Method facts.** The normalized analysis contains **21 distinct model and precision/quantization configurations** instantiated as **23 context-specific deployments** across three deployment contexts and four workloads. Accuracy is available for MMLU, GSM8K, and HellaSwag; Dolly is treated as an open-ended generation workload.

**Result facts.** The accuracy-leading configuration changes by context and dataset. Examples from the current data are: Edge: Phi-4 Mini 3.8B (Q4) (GSM8K accuracy 0.639); Local CPU: Phi-4 Mini 3.8B (Q4) (GSM8K accuracy 0.640); GPU server: Phi-4 14B (bf16) (HellaSwag accuracy 0.850). Context-local Pareto frontiers identify multiple relevant configurations rather than one universal winner.

**Interpretation.** The evidence supports a deployment-context-aware claim: feasibility and memory dominate on edge hardware, latency dominates on local CPU, and the GPU context exposes an accuracy–throughput trade-off. Avoid claiming that one model is globally best.
