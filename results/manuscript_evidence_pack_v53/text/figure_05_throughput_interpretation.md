# Figure 5 throughput interpretation

The observed reported-throughput range spans 45.1× from the smallest to the largest value, so Figure 5 uses a logarithmic y-axis.

Cross-context differences must not be attributed to model size alone. Edge and Local CPU use llama.cpp active-generation tokens/s, while the GPU server uses vLLM run-level sustained tokens/s. Hardware, backend, precision/quantization, model architecture, and the throughput denominator all differ.

Within a fixed benchmark and deployment context, the boxed fastest/slowest ratios provide the cleaner comparison. Those spreads reflect the observed combined effect of model scale, architecture, precision/quantization, runtime behavior, and generated-output behavior under the same workload.

## Within-benchmark, within-context spreads

| benchmark_display | context_display | fastest_model          | fastest_throughput_tokens_per_sec | slowest_model        | slowest_throughput_tokens_per_sec | within_context_spread_x | throughput_measurement_basis |
|-------------------|-----------------|------------------------|-----------------------------------|----------------------|-----------------------------------|-------------------------|------------------------------|
| Dolly             | Edge            | Llama 3.2 1B (Q4)      | 9.247                             | Phi-4 Mini 3.8B (Q4) | 3.259                             | 2.837                   | Active generation            |
| Dolly             | Local CPU       | Gemma 3 1B (Q8)        | 15.617                            | Ministral 3 3B (Q4)  | 6.407                             | 2.438                   | Active generation            |
| Dolly             | GPU server      | Llama 3.2 1B (bf16)    | 98.172                            | Ministral 3 14B      | 11.543                            | 8.505                   | Run-level sustained          |
| GSM8K             | Edge            | Llama 3.2 1B (Q4)      | 9.493                             | Phi-4 Mini 3.8B (Q4) | 3.106                             | 3.056                   | Active generation            |
| GSM8K             | Local CPU       | Gemma 3 1B (Q8)        | 14.413                            | Ministral 3 3B (Q4)  | 5.757                             | 2.504                   | Active generation            |
| GSM8K             | GPU server      | Llama 3.2 1B (bf16)    | 93.236                            | Ministral 3 14B      | 10.493                            | 8.886                   | Run-level sustained          |
| HellaSwag         | Edge            | Llama 3.2 1B (Q4)      | 12.359                            | Phi-4 Mini 3.8B (Q4) | 3.627                             | 3.407                   | Active generation            |
| HellaSwag         | Local CPU       | Llama 3.2 1B (Q8)      | 18.017                            | Ministral 3 3B (Q4)  | 8.210                             | 2.195                   | Active generation            |
| HellaSwag         | GPU server      | Phi-4 Mini 3.8B (bf16) | 9.930                             | Apertus 70B (bf16)   | 2.179                             | 4.557                   | Run-level sustained          |
| MMLU              | Edge            | Llama 3.2 1B (Q4)      | 12.697                            | Phi-4 Mini 3.8B (Q4) | 3.550                             | 3.577                   | Active generation            |
| MMLU              | Local CPU       | Llama 3.2 1B (Q8)      | 18.139                            | Ministral 3 3B (Q4)  | 8.348                             | 2.173                   | Active generation            |
| MMLU              | GPU server      | Phi-4 Mini 3.8B (bf16) | 7.478                             | Apertus 70B (bf16)   | 2.269                             | 3.295                   | Run-level sustained          |
