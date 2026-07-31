# Manuscript evidence-pack map

## Main manuscript figures
- `figures/figure_01_framework_architecture.{png,pdf,svg}`
- `figures/figure_02_pi5_peak_system_ram.{png,pdf,svg}`
- `figures/figure_02a1_gpu_mean_energy_per_request_dolly_gsm8k.{png,pdf,svg}`
- `figures/figure_02a2_gpu_mean_energy_per_request_hellaswag_mmlu.{png,pdf,svg}`
- `figures/figure_03_cpu_peak_system_ram.{png,pdf,svg}`
- `figures/figure_03a_gpu_vram_footprint_intervals.{png,pdf,svg}`
- `figures/figure_03b1_gpu_energy_per_request_boxplots_dolly_gsm8k.{png,pdf,svg}`
- `figures/figure_03b2_gpu_energy_per_request_boxplots_hellaswag_mmlu.{png,pdf,svg}`
- `figures/figure_04_accuracy_by_context_and_dataset.{png,pdf,svg}`
- `figures/figure_05_sustained_throughput_across_contexts.{png,pdf,svg}`
- `figures/figure_06_gsm8k_accuracy_latency_frontiers.{png,pdf,svg}`
- `figures/figure_07_gsm8k_accuracy_memory_frontiers.{png,pdf,svg}`

## Figure-selection data
- `tables/figure_02_pi5_peak_system_ram_values.csv`
- `tables/figure_02a1_gpu_mean_energy_per_request_dolly_gsm8k_values.csv`
- `tables/figure_02a2_gpu_mean_energy_per_request_hellaswag_mmlu_values.csv`
- `tables/gpu_request_energy_statistics.csv`
- `tables/figure_03_cpu_peak_system_ram_values.csv`
- `tables/figure_03a_gpu_vram_footprint_intervals_statistics.csv`
- `tables/figure_03b1_gpu_energy_per_request_boxplots_dolly_gsm8k_measurement_values.csv`
- `tables/figure_03b2_gpu_energy_per_request_boxplots_hellaswag_mmlu_measurement_values.csv`
- `tables/figure_s03a_gpu_run_min_adjusted_vram_boxplots_measurement_values.csv`
- `tables/figure_04_accuracy_by_benchmark_long.csv`
- `tables/figure_04_accuracy_by_benchmark_wide.csv`
- `tables/figure_04_accuracy_panels_long.csv` (compatibility alias)
- `tables/figure_04_accuracy_panels_wide.csv` (compatibility alias)
- `tables/figure_04_accuracy_by_context_gsm8k_order.csv` (compatibility alias)
- `tables/figure_05_throughput_panels_long.csv`
- `tables/figure_05_throughput_panels_wide.csv`
- `tables/figure_05_throughput_spread_by_panel.csv`
- `tables/figure_05_throughput_by_benchmark.csv` (compatibility alias)
- `tables/figure_05_throughput_spread_by_benchmark.csv` (compatibility alias)

## Main manuscript tables
Each table is exported as CSV, Markdown, and LaTeX.
- Table 1: deployment contexts
- Table 2: evaluated model configurations and corrected metadata
- Table 3: hardware platforms
- Table 4: selected task-level accuracy
- Table 5: selected timing and throughput
- Table 6: selected peak resources
- Table 7: explicitly defined cost-effectiveness indicator

## Text-support documents
- `text/manuscript_figure_captions.md`
- `text/results_section_data_draft.md`
- `text/abstract_evidence_block.md`
- `text/canonical_results_facts.csv`
- `text/manuscript_statement_replacements.md`
- `text/dolly_split_acquisition_protocol.md`
- `text/figure_05_throughput_interpretation.md`


## Figure accessibility
- Figures 2, 3, 3a, 4, and 5 use a color-blind-friendly palette with hatch patterns and dark outlines.
- Figures 6--7 use open versus filled markers, different marker shapes, connecting lines, and direct labels.
- The scientific interpretation therefore remains available in grayscale and never depends on hue alone.

## Checks
- `checks/model_metadata_audit.csv`
- `checks/shared_model_order_gsm8k.csv`
- `checks/figure_04_05_shared_model_positions.csv`
- `checks/figure_04_export_schema.csv`
- `checks/figure_05_export_schema.csv`
- `checks/figure_02_pi5_peak_system_ram_benchmark_order.csv`
- `checks/figure_03_cpu_peak_system_ram_benchmark_order.csv`
- `checks/figure_02_pi5_peak_system_ram_model_order_dolly.csv`
- `checks/figure_02a1_gpu_mean_energy_per_request_dolly_gsm8k_model_order_dolly.csv`
- `checks/figure_02a1_gpu_mean_energy_per_request_dolly_gsm8k_benchmark_order.csv`
- `checks/figure_02a2_gpu_mean_energy_per_request_hellaswag_mmlu_model_order_dolly.csv`
- `checks/figure_02a2_gpu_mean_energy_per_request_hellaswag_mmlu_benchmark_order.csv`
- `checks/figure_03b1_gpu_energy_per_request_boxplots_dolly_gsm8k_model_order_dolly.csv`
- `checks/figure_03b1_gpu_energy_per_request_boxplots_dolly_gsm8k_benchmark_order.csv`
- `checks/figure_03b2_gpu_energy_per_request_boxplots_hellaswag_mmlu_model_order_dolly.csv`
- `checks/figure_03b2_gpu_energy_per_request_boxplots_hellaswag_mmlu_benchmark_order.csv`
- `checks/figure_03_cpu_peak_system_ram_model_order_dolly.csv`
- `checks/figure_03a_gpu_vram_footprint_intervals_model_order_dolly.csv`
- `checks/figure_03a_gpu_vram_distribution_diagnostics.csv`
- `checks/gpu_request_energy_source_audit.csv`
- `checks/gpu_request_energy_source_counts.csv`
- `checks/gpu_request_energy_request_audit.csv`
- `checks/figure_s03a_gpu_run_min_baseline_audit.csv`
- `checks/figure_03a_gpu_vram_measurement_source_audit.csv`
- `checks/figure_03a_gpu_vram_measurement_source_counts.csv`
- `checks/figure_03a_gpu_vram_measurement_trace_audit.csv`
- `checks/normalized_summary_source_audit.csv`
- `checks/throughput_metric_audit.csv`
- `checks/dolly_split_acquisition_audit.csv`
- `checks/dolly_split_pooling_audit.csv`
- `checks/dolly_processed_peak_audit.csv`
- `checks/throughput_missingness_by_context.csv`
- `checks/manuscript_consistency_audit.{csv,md}`
- `checks/output_manifest.csv`

## Supplement
- `figures/figure_s03a_gpu_run_min_adjusted_vram_boxplots.{png,pdf,svg}`
- context-local compact-model delta figures;
- context-local delta CSV;
- competitive compact-model evidence CSV;
- GSM8K Pareto-membership CSV.
