# MoE SSD/DRAM Prefetch Guide Triage

**First spotted:** 2026-06-29
**Status:** Active

## Summary

Two Deep Research Max runs completed for the proposed learned SSD-to-DRAM MoE
expert prefetch experiment. The ingested guides are useful enough to support a
repo-native plan, but several prior-art claims from the raw MoE report remain
weakly sourced and must be verified from primary papers before implementation
depends on them.

## Raw Sources

- MoE prior art raw report:
  `/Users/mm/Documents/GitHub/llm-workflows/outputs/create-guide/moe-expert-offload-and-prefetch-prior-art-for-ssd-dram-on-device-inference/2026-06-29T05-13-42-684Z/raw-report.md`
- Apple Silicon measurement raw report:
  `/Users/mm/Documents/GitHub/llm-workflows/outputs/create-guide/apple-silicon-nvme-cache-bypass-and-energy-measurement-for-ssd-tier-model-benchm/2026-06-29T05-13-42-684Z/raw-report.md`

## Related Guides

- [MoE SSD/DRAM expert prefetch experiment plan](../Plans/moe-ssd-dram-prefetch-v1.md)
- [MoE expert offload and prefetch prior art](../Guides/moe-expert-offload-prefetch-prior-art-guide.md)
- [Apple Silicon NVMe and energy measurement](../Guides/apple-silicon/apple-silicon-nvme-energy-measurement-guide.md)
- [Apple Silicon warmed-inference benchmark hygiene](../Guides/apple-silicon/Apple-Silicon-warmed-inference-benchmark-hygiene-guide.md)
- [Core ML compute-unit scheduling](../Guides/apple-silicon/CoreML-Compute-Unit-Scheduling-guide.md)

## Ingest Decisions

- Kept the MoE guide focused on experiment routing and measurement contracts,
  not as a definitive survey of every named system.
- Treated Expert Activation Matrix style prediction as the simplest credible
  first predictor before any trained MLP.
- Preserved the central Ilya-style gate: Stage 0 can kill the project before ML
  work if the oracle bandwidth ceiling is too low.
- Rewrote the Apple Silicon guide around local `man` pages for `fcntl`,
  `fs_usage`, and `powermetrics`.
- Removed or softened raw-report claims that depended on weak sources, hidden
  implementation details, or unverifiable absolute power numbers.

## Claims To Verify Before Planning

- Exact mechanisms and reported accuracy for PreScope/LLaPor, DuoServe-MoE,
  ST-MoE, HeteroLLM, NPUMoE, EdgeMoE, FlexInfer, AdapMoE, SwapMoE, and Fiddler.
- Whether any cited SSD-tier system measures true NVMe reads rather than a
  higher-level offload boundary.
- Any claim about Apple Silicon SSD queue depth, memory-fabric contention, or
  expert block size. These must be measured locally in Stage 0.

## If This Proceeds

- [x] Create a checked-in plan under `README/Plans/`.
- [ ] Freeze Stage 0 thresholds before running measurements.
- [ ] Build the Stage 0 direct-read microbenchmark before any predictor.
- [ ] Record `fs_usage` and `powermetrics` artifact paths beside every claimed
      result.

## Required Executable Memory

Not testable: this is a provenance and triage note. The strongest proof is the
presence of both raw reports plus the ingested guides linked above.
