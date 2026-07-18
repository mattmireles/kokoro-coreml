# MoE Expert Offload And Prefetch Prior Art Guide

This guide ingests a Deep Research Max report for learned SSD/DRAM expert
prefetching. Treat the raw report as research input, not canonical truth.

Raw report:

- `/Users/mm/Documents/GitHub/llm-workflows/outputs/create-guide/moe-expert-offload-and-prefetch-prior-art-for-ssd-dram-on-device-inference/2026-06-29T05-13-42-684Z/raw-report.md`

## Executive Summary

The useful lesson is simple: expert prediction is not the hard part. The hard
part is whether expert prediction buys enough lead time to hide real storage
latency, without wasting enough SSD bandwidth and energy to lose the system.

Most public MoE offload work targets a discrete GPU machine where CPU DRAM
feeds GPU memory over PCIe. That is not the Apple Silicon regime. On Apple
Silicon, CPU and GPU share unified memory, so the interesting boundary is the
SSD-to-DRAM boundary, plus the memory-fabric contention and energy cost of
reading weights while compute is running. Pair this guide with the
[Apple Silicon NVMe and energy measurement guide](apple-silicon/apple-silicon-nvme-energy-measurement-guide.md)
before designing Stage 0.

The first predictor to implement should be boring: trace router outputs, build
simple baselines, then replay the trace. Do not train a new MLP until the
oracle and trivial baselines show that prefetching can win physically.

## Prior-Art Map

| System | Useful Idea | Boundary Mostly Measured | Transfer To Apple Silicon UMA |
| --- | --- | --- | --- |
| Mixtral-Offloading | Cache experts and speculate near-future experts for Mixtral-8x7B. | CPU RAM to GPU over PCIe. | Low for speed claims; useful as a minimal LRU/speculation reference. |
| MoE-Infinity | Expert Activation Matrix traces guide cache replacement and prefetch. | CPU RAM to GPU over PCIe. | Medium for tracing and baselines; low for hardware conclusions. |
| HOBBIT | Mixed-precision expert offload reduces miss cost by using lower-precision resident fallbacks. | CPU/SSD tiers feeding GPU memory. | Medium; fallback idea is useful if SSD fetch misses dominate. |
| Pre-gated MoE | Changes the model/router so future expert decisions are available earlier. | CPU RAM to GPU over PCIe. | Low for this experiment; model changes are out of scope unless Stage 0-2 prove the need. |
| EdgeMoE / SSD-tier systems | Treat storage as a slow tier for edge deployment. | Storage to host memory and/or accelerator memory. | High conceptually; every claim must be remeasured on target Apple hardware. |
| HeteroLLM / mobile SoC work | Models heterogeneous accelerator placement on mobile-class systems. | Mobile SoC memory and accelerator boundaries. | High for framing UMA placement; separate from SSD expert paging. |
| Token/energy benchmark work | Measures phase-level energy instead of only tokens/sec. | Varies by framework. | High for metric discipline; do not copy absolute numbers across devices. |

The raw report names additional systems, including AdapMoE, SwapMoE, Fiddler,
FlexInfer, DuoServe-MoE, PreScope/LLaPor, ST-MoE, and NPUMoE. Before a plan
depends on any of those claims, verify the primary paper or repository and
record the exact source in the plan or note. Some raw-report entries are
secondary-source only.

## Predictor Mechanisms

### Cross-Layer Prediction

The standard trick is to use routing information or hidden-state similarity from
nearby layers/tokens to predict future expert IDs. The implementation form
varies, but the experiment only needs this contract:

- At token `t`, layer `L`, observe the current router input and top-k experts.
- Predict the experts needed at layer `L + d`, where `d` is the prefetch depth.
- Issue prefetches during the compute window before layer `L + d` requests the
  weights.
- Score the prediction by recall, precision, and whether the fetch finished
  before use.

Raw recall alone is not the metric. The right metric is hideable recall:
predicted experts that both match the future route and arrive before the
consumer stalls.

### Expert Activation Matrix

MoE-Infinity's Expert Activation Matrix is the most pragmatic first predictor.
For `num_layers x num_experts`, count routed expert use during a request, compare
the partial matrix against historical matrices, and use the nearest historical
matrix to decide cache priority and prefetch candidates.

This has two virtues:

- it is trace-first and easy to simulate;
- it gives a strong non-neural baseline before training any MLP.

Its weakness is also useful: it can fail on prompt domain shifts and first-token
cold starts. Those failures are exactly what Stage 1 should measure across code,
math, and prose prompts.

### Learned MLP Predictors

Learned predictors can be added after trace replay proves that:

- the oracle ceiling is meaningfully above demand paging;
- trivial policies do not capture most of the win;
- the measured SSD latency can be hidden with the available lead time.

Until those gates pass, a trained predictor is complexity without evidence.

## Measurement Contract

Stage 1 traces should record:

- `request_id`
- `domain`
- `token_index`
- `layer_index`
- `router_input_shape`
- `actual_topk_expert_ids`
- `predicted_expert_ids`
- `layer_compute_ns`
- `prefetch_issue_time_ns`
- `expert_ready_time_ns`
- `cache_hit`
- `evicted_expert_ids`

Stage 1 baselines:

- demand LRU with no prefetch;
- last-token expert set;
- global frequency top-k;
- Markov or n-gram over expert IDs;
- EAM cosine similarity;
- oracle future trace.

Stage 2 simulator metrics:

- effective decode tokens/sec;
- stall time per token;
- recall@k;
- precision;
- hideable recall;
- prefetch hit rate;
- wasted-byte fraction;
- cache pollution;
- SSD bytes/token;
- joules/token.

## Apple Silicon Transfer Rules

Do not transfer PCIe speedups directly to Apple Silicon. A system that wins by
hiding CPU-to-GPU transfer may have measured the wrong boundary for UMA.

Do not trust an SSD-tier result unless it proves real storage reads. macOS file
cache can turn a fake SSD benchmark into a DRAM benchmark. Use the
[NVMe measurement guide](apple-silicon/apple-silicon-nvme-energy-measurement-guide.md)
for the evidence checklist.

Do not start with a real runtime integration. The right order is:

1. Stage 0 hardware envelope.
2. Stage 1 trace and predictor replay.
3. Stage 2 discrete-event simulator.
4. Stage 3 runtime implementation only if the earlier gates pass.

## Plan Implications

The likely first kill gate is bandwidth, not ML. If active expert bytes/token
divided by measured NVMe bandwidth is below the target tokens/sec even for an
oracle, the program is dead. The simple answer is compression or a smaller
model, not a smarter prefetcher.

If hideability is much greater than one layer of compute, the experiment must
try multi-layer-ahead prediction. That lowers accuracy and raises wasted reads,
so the simulator must make the cost visible before runtime work begins.

If a learned predictor is within noise of last-token or global-frequency
prefetch, learning has no product value even if prefetch itself helps.

## Related Documentation

- [Apple Silicon NVMe and energy measurement](apple-silicon/apple-silicon-nvme-energy-measurement-guide.md)
- [Apple Silicon warmed-inference benchmark hygiene](apple-silicon/Apple-Silicon-warmed-inference-benchmark-hygiene-guide.md)
- [Core ML compute-unit scheduling](apple-silicon/CoreML-Compute-Unit-Scheduling-guide.md)
- [MoE SSD/DRAM guide triage note](../Notes/moe-ssd-dram-prefetch-guide-triage-2026-06-29.md)

## Works Cited

1. [Mixtral-Offloading repository](https://github.com/dvmazur/mixtral-offloading) - practical Mixtral expert offloading reference.
2. [MoE-Infinity / Activation-Aware Expert Offloading](https://arxiv.org/html/2401.14361v3) - Expert Activation Matrix and activation-aware offload.
3. [MoE-Infinity OpenReview entry](https://openreview.net/forum?id=BL7WMLJKZM) - claims and discussion for activation-aware expert offload.
4. [HOBBIT: A Mixed Precision Expert Offloading System](https://arxiv.org/html/2411.01433v1) - mixed-precision expert offload.
5. [Pre-gated MoE, ISCA 2024](https://dl.acm.org/doi/10.1109/ISCA59077.2024.00078) - algorithm-system co-design for earlier expert decisions.
6. [Microsoft Pre-gated MoE publication page](https://www.microsoft.com/en-us/research/publication/pre-gated-moe-an-algorithm-system-co-design-for-fast-and-scalable-mixture-of-expert-inference/) - summary and authors.
7. [ACM survey on MoE inference optimization](https://dl.acm.org/doi/10.1145/3794845) - broad taxonomy of MoE inference optimization.
