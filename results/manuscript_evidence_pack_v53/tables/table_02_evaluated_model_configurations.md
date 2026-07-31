### Evaluated model and precision/quantization configurations. A configuration is counted once even when evaluated in multiple deployment contexts.

| Model configuration        | Family      | Total parameters (B) | Active parameters (B) | Architecture | Precision / quantization | Deployment context(s) |
|----------------------------|-------------|----------------------|-----------------------|--------------|--------------------------|-----------------------|
| Gemma 3 1B (Q8)            | Gemma 3     | 1.000                | 1.000                 | dense        | Q8_0                     | Edge; Local CPU       |
| Llama 3.2 1B (Q4)          | Llama 3.2   | 1.000                | 1.000                 | dense        | Q4_K_M                   | Edge                  |
| Llama 3.2 1B (Q8)          | Llama 3.2   | 1.000                | 1.000                 | dense        | Q8_0                     | Local CPU             |
| Llama 3.2 1B (bf16)        | Llama 3.2   | 1.000                | 1.000                 | dense        | bf16                     | GPU server            |
| Qwen 2.5 1.5B (bf16)       | Qwen 2.5    | 1.540                | 1.540                 | dense        | bf16                     | GPU server            |
| Ministral 3 3B (AWQ)       | Ministral 3 | 3.000                | 3.000                 | dense        | AWQ                      | GPU server            |
| Ministral 3 3B (Q4)        | Ministral 3 | 3.000                | 3.000                 | dense        | Q4_K_M                   | Local CPU             |
| SmolLM3 3B (Q4)            | SmolLM3     | 3.000                | 3.000                 | dense        | Q4_K_M                   | Edge                  |
| Phi-4 Mini 3.8B (GPTQ-4)   | Phi-4 Mini  | 3.800                | 3.800                 | dense        | GPTQ-4                   | GPU server            |
| Phi-4 Mini 3.8B (Q4)       | Phi-4 Mini  | 3.800                | 3.800                 | dense        | Q4_K_M                   | Edge; Local CPU       |
| Phi-4 Mini 3.8B (bf16)     | Phi-4 Mini  | 3.800                | 3.800                 | dense        | bf16                     | GPU server            |
| Qwen 3 4B (AWQ)            | Qwen 3      | 4.000                | 4.000                 | dense        | AWQ                      | GPU server            |
| Granite 4 H-Tiny (bf16)    | Granite 4   | 7.000                | 1.000                 | moe          | bf16                     | GPU server            |
| Qwen 2.5 7B (AWQ)          | Qwen 2.5    | 7.620                | 7.620                 | dense        | AWQ                      | GPU server            |
| Ministral 3 8B (AWQ)       | Ministral 3 | 8.000                | 8.000                 | dense        | AWQ                      | GPU server            |
| Ministral 3 14B            | Ministral 3 | 14.000               | 14.000                | dense        | bf16                     | GPU server            |
| Phi-4 14B (bf16)           | Phi-4       | 14.000               | 14.000                | dense        | bf16                     | GPU server            |
| Qwen 3 30B-A3B (bf16)      | Qwen 3      | 30.500               | 3.300                 | moe          | bf16                     | GPU server            |
| Kimi Linear 48B-A3B (bf16) | Kimi Linear | 48.000               | 3.000                 | moe          | bf16                     | GPU server            |
| Apertus 70B (bf16)         | Apertus     | 70.000               | 70.000                | dense        | bf16                     | GPU server            |
| Llama 3.3 70B (FP8)        | Llama 3.3   | 70.000               | 70.000                | dense        | FP8                      | GPU server            |
