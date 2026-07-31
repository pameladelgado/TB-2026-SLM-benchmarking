# Manuscript statement replacements

These passages use the counting definitions reported in Table 2.


## Accuracy notation

Accuracy is reported as a proportion on the closed interval **[0, 1]**.
For example, 0.64 corresponds to 64% correct responses. Figures and tables
use the label `Accuracy [0, 1]` to prevent ambiguity with percentage units.

## Counting definition

A **model configuration** is one distinct model artifact and
precision/quantization variant. The same artifact is counted once even when
it is evaluated in more than one deployment context. A
**context-specific deployment** is one model configuration instantiated in
one deployment context.

Current audited counts:

- Distinct model/precision configurations: **21**
- Context-specific deployments: **23**
- Unique model families: **12**
- GPU-server configurations: **15**

## Contribution bullet replacement

We report empirical Pareto frontiers per context across
**21 distinct model and precision/quantization
configurations**, instantiated as **23
context-specific deployments**, and show that the preferred model changes
by context: no single model dominates all three environments, and the
selection changes when the operational objective shifts from accuracy to
latency or memory.

## Model-pool sentence replacement

The evaluated pool contains **21 distinct
model and precision/quantization configurations**, spanning
approximately 1.0B to 70B parameters and including dense and mixture-of-experts architectures.
Because model assignment follows deployment feasibility, these
configurations correspond to **23 context-specific
deployments** rather than a fully crossed model-by-platform design.

## GPU-results sentence replacement

On the GPU server, the model pool expands to
**15 configurations**, ranging from compact models
to approximately 70B parameters and including dense and
mixture-of-experts architectures.

## Conclusion sentence replacement

We evaluated **21 distinct model and
precision/quantization configurations**, instantiated across
**23 context-specific deployments**, using a
reproducible measurement instrument that combines task-level accuracy with
inline system-level metrics collected during streaming inference.

## Table 7 caption replacement

Short-output cost-effectiveness indicators on MMLU. Expected correct
responses per second is computed as benchmark accuracy [0, 1] multiplied by
measured sample throughput under sequential inference.

## Table 7 interpretation replacement

Table 7 reports a short-output cost-effectiveness indicator defined as
expected correct responses per second:

`expected correct responses/s = accuracy × samples/s`.

The indicator estimates the rate at which a configuration produces correct
benchmark responses under the measured sequential-request protocol. It is
most interpretable for short, similarly structured outputs such as MMLU.
It should not be treated as a general measure of reasoning efficiency,
because output length, parsing requirements, and workload structure differ
across datasets.
