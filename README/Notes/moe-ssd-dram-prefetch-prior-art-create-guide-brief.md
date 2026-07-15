# Create-Guide Brief: MoE Expert Offload And Prefetch Prior Art

## Topic

Learned expert prediction, caching, and SSD/DRAM offload for on-device or memory-constrained Mixture-of-Experts LLM inference.

## Target Repo

`kokoro-coreml`

## Target Guide Path

`README/Guides/moe-expert-offload-prefetch-prior-art-guide.md`

## Context

We are evaluating an experiment proposal: a learned SSD-to-DRAM expert prefetcher for on-device MoE inference on Apple Silicon UMA devices. The proposed experiment assumes that predicting next-layer experts is mostly solved by prior art, while the open question is whether that predictor produces a real speed and energy win across the physical SSD/DRAM boundary on commodity unified-memory devices.

The research guide should help a future implementation agent avoid rebuilding known predictors and focus on the falsifiable system question. Treat this as an advanced developer field guide, not a survey essay.

## Primary Research Goal

Determine what prior systems actually do for MoE expert prediction, prefetch, caching, and offload; which claims are supported by measurements; which regimes they measure; and what gaps remain for Apple Silicon UMA plus NVMe plus joules/token.

## Questions To Answer

- What is the standard cross-layer expert prediction trick, exactly? Include the mechanism, required router inputs, lead time, recall/precision metrics, and common failure modes.
- Which papers or systems report approximately 90 percent expert prediction accuracy, and under what model, workload, top-k, and hardware assumptions?
- How do Mixtral-Offloading, HOBBIT, AdapMoE, EdgeMoE, MoE-Infinity, SwapMoE, Fiddler, FlexInfer, Pre-gated MoE, DuoServe-MoE, PreScope/LLaPor, HeteroLLM, and ST-MoE differ?
- Which systems use SSD as a tier versus CPU DRAM versus GPU HBM? Which hide PCIe transfer rather than true NVMe latency?
- Which systems report energy, power, thermals, or joules/token? Separate commodity hardware from custom silicon.
- What baselines are considered honest: demand paging, LRU/LFU, last-token-set, global-frequency, Markov/n-gram, oracle?
- What metrics should we copy for Stage 1 and Stage 2: recall@k, precision, hideable recall, wasted-byte fraction, prefetch hit rate, cache pollution, tokens/sec, joules/token?
- Which claims should be distrusted or treated as non-transferable to Apple Silicon UMA?
- What is the minimal off-the-shelf predictor a pragmatic engineer should implement first before considering a learned MLP?

## Source Hints

Prioritize primary sources and implementation repositories when available.

- Survey on Inference Optimization for Mixture of Experts, arXiv:2412.14219.
- HOBBIT, arXiv:2411.01433.
- Pre-gated MoE, ISCA 2024, Microsoft.
- DuoServe-MoE, arXiv:2509.07379.
- PreScope / LLaPor, arXiv:2509.23638.
- HeteroLLM, arXiv:2501.14794.
- ST-MoE, arXiv:2606.15453.
- Mixtral-Offloading.
- MoE-Infinity.
- EdgeMoE.
- AdapMoE.
- SwapMoE.
- Fiddler.
- FlexInfer, EuroMLSys 2025.

If a named source cannot be found, say so explicitly and do not invent details.

## Output Format

Produce a text-only Markdown field guide. Do not include charts, generated images, diagrams, or visualizations.

Required sections:

1. Executive summary for a systems implementer.
2. Prior-art matrix with columns for system, model family, prediction method, cache/offload tier, hardware, measured metrics, energy coverage, and transferability to Apple Silicon UMA.
3. Predictor mechanisms and how to reproduce the simplest credible one.
4. Measurement traps and non-transferable claims.
5. Recommended Stage 1 trace schema and baselines.
6. Recommended Stage 2 simulator parameters and baselines.
7. Open questions for the proposed Apple Silicon experiment.
8. Source list with direct links.

Mark speculative recommendations separately from evidence-backed findings.
