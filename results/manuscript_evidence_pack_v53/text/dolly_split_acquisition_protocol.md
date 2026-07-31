# Dolly split-acquisition protocol note

For some configurations, Dolly was collected in more than one acquisition batch on different dates. The analysis concatenates the underlying sample records and computes sample-level means. Consequently, a batch of 100 samples and a batch of 900 samples contribute in proportion to their sample counts; their batch means are not averaged with equal weight.

This pooling is valid only because the model, hardware, backend, generation settings, prompt construction, and measurement procedure are treated as protocol-equivalent across the acquisition dates.

The acquisition structure observed by the notebook is:

| platform | model                                    | sample_count | run_count | acquisition_date_count | acquisition_dates      | run_sample_counts | protocol_structure                       | supports_ten_independent_run_claim |
|----------|------------------------------------------|--------------|-----------|------------------------|------------------------|-------------------|------------------------------------------|------------------------------------|
| cpu      | gemma-3-1b-it-Q8_0.gguf                  | 200          | 2         | 2                      | 2026-05-01; 2026-07-10 | 100; 100          | 2 independent-looking 100-sample run IDs | No                                 |
| cpu      | Llama-3.2-1B-Instruct-Q8_0.gguf          | 100          | 1         | 1                      | 2026-04-29             | 100               | 1 independent-looking 100-sample run IDs | No                                 |
| cpu      | Ministral-3-3B-Instruct-2512-Q4_K_M.gguf | 200          | 2         | 2                      | 2026-05-01; 2026-07-10 | 100; 100          | 2 independent-looking 100-sample run IDs | No                                 |
| cpu      | Phi-4-mini-instruct-Q4_K_M.gguf          | 200          | 2         | 2                      | 2026-05-01; 2026-07-10 | 100; 100          | 2 independent-looking 100-sample run IDs | No                                 |
| pi5      | gemma-3-1b-it-Q8_0.gguf                  | 200          | 2         | 2                      | 2026-05-01; 2026-06-27 | 100; 100          | 2 independent-looking 100-sample run IDs | No                                 |
| pi5      | llama-3.2-1b-instruct-q4_k_m.gguf        | 200          | 2         | 2                      | 2026-05-01; 2026-06-27 | 100; 100          | 2 independent-looking 100-sample run IDs | No                                 |
| pi5      | Phi-4-mini-instruct-Q4_K_M.gguf          | 200          | 2         | 2                      | 2026-05-01; 2026-06-27 | 100; 100          | 2 independent-looking 100-sample run IDs | No                                 |
| pi5      | SmolLM3-3B-Q4_K_M.gguf                   | 200          | 2         | 2                      | 2026-05-01; 2026-06-27 | 100; 100          | 2 independent-looking 100-sample run IDs | No                                 |

Important: a single 900-sample server session is not statistically equivalent to nine independent 100-sample runs. The manuscript's ten-independent-run statement should therefore be restricted to workloads/configurations whose audit confirms ten distinct 100-sample run identifiers.
