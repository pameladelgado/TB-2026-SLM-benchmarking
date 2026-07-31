### Peak resource utilization for representative configurations.

| Context    | Model                      | Dataset | CPU peak (%) | RAM peak (MB) | VRAM peak (MB) | Temperature peak (°C) |
|------------|----------------------------|---------|--------------|---------------|----------------|-----------------------|
| GPU server | Apertus 70B (bf16)         | Dolly   | 5.700        | 15629.590     | 88571.000      | 51.000                |
| GPU server | Kimi Linear 48B-A3B (bf16) | Dolly   | 6.600        | 16061.170     | 92595.000      | 54.000                |
| GPU server | Llama 3.2 1B (bf16)        | Dolly   | 5.400        | 15489.590     | 88447.000      | 54.000                |
| GPU server | Llama 3.3 70B (FP8)        | Dolly   | 5.900        | 16164.480     | 88565.000      | 53.000                |
| GPU server | Ministral 3 14B            | Dolly   | 5.100        | 16501.000     | 88211.000      | 54.000                |
| GPU server | Phi-4 14B (bf16)           | Dolly   | 6.200        | 15103.590     | 88487.000      | 52.000                |
| GPU server | Phi-4 Mini 3.8B (bf16)     | Dolly   | 5.500        | 15451.220     | 88467.000      | 54.000                |
| GPU server | Qwen 3 30B-A3B (bf16)      | Dolly   | 5.200        | 16013.150     | 88447.000      | 54.000                |
| Local CPU  | Llama 3.2 1B (Q8)          | GSM8K   | 30.400       | 5254.590      | --             | 100.000               |
| Local CPU  | Ministral 3 3B (Q4)        | MMLU    | 94.400       | 10818.850     | --             | 100.000               |
| Local CPU  | Phi-4 Mini 3.8B (Q4)       | MMLU    | 96.000       | 12329.490     | --             | 100.000               |
| Local CPU  | Gemma 3 1B (Q8)            | MMLU    | 85.100       | 11657.050     | --             | 100.000               |
| Edge       | Phi-4 Mini 3.8B (Q4)       | MMLU    | 100.000      | 6517.080      | --             | 71.900                |
| Edge       | SmolLM3 3B (Q4)            | MMLU    | 100.000      | 3987.860      | --             | 71.400                |
| Edge       | Gemma 3 1B (Q8)            | MMLU    | 100.000      | 2548.550      | --             | 67.500                |
| Edge       | Llama 3.2 1B (Q4)          | MMLU    | 100.000      | 1968.190      | --             | 70.800                |
