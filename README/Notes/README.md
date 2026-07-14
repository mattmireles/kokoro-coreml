# Notes

Use this folder for **time-bound debugging trails**, **investigation writeups**,
and **audit notes**—see the **`write-notes`** skill (`.claude/skills/write-notes/`).

Prefer **one domain file** with multiple issue sections over a new file per
session; see [../Guides/content/notes-consolidation-guide.md](../Guides/content/notes-consolidation-guide.md).

Long-running learnings also appear under `README/learnings.md` and other
top-level `README/*.md` files; link from notes instead of duplicating.

## Index

- [CS1 audio-quality evaluation (2026-07-14)](cs1-audio-quality-evaluation-2026-07-14.md) - Formal perceptual evaluation for the paper: PyTorch reference vs Config F (staged policy) on the frozen bakeoff inputs, blind Gemini audio-judge lineups with a known-bad control. Initial run: 3s/7s pass, 15s/30s fail. Same-day root-cause + fix: the 15s failure was Swift whitespace suppression cutting real speech (duration model exonerated at per-token parity), the 30s "tilt" was a one-sided iSTFT scaling bug in `CustomSTFT.inverse` present on every bucket, plus a seed-0 `SeededRNG` degeneracy. Post-fix all four buckets pass blind lineups; generator packages re-exported.
- [Monolithic Core ML control experiment](monolithic-coreml-control-experiment.md) - The paper's null-hypothesis control: whole-model Kokoro export fails conversion at the data-dependent `repeat_interleave` alignment (branch a, trace-baked), and the downstream fused decoder converts but collapses to corr ~0.27 on hn-nsf SineGen phase (branch b). Both failures land on the decomposition seams.
- [iPhone performance notes](iphone-performance-notes.md) - Physical-iPhone warmed timings: Config F vs MLX Swift (kokoro-ios), per-device raw warm arrays, compute-policy disclosure.
- [iPhone debug notes](iphone-debug-notes.md) - iPhone-domain failure modes: ANECCompile `.all` rejection, jetsam memory budget, dyld dynamic-framework abort, locked-device launch gate, host xcodebuild stall reaper.
- [Core ML compute-unit ablation](coreml-compute-unit-ablation.md) - F/G/G-prime/G-double-prime benchmark logic for isolating `.all`, ANE, GPU, and CPU behavior in the Swift pipeline.
- [Restarted Kokoro guide triage](kokoro-restarted-guide-triage-2026-06-06.md) - Ingested Deep Research guide triage for warmed lower-end Mac optimization, benchmark hygiene, and source/body candidate priority.
- [Apple Silicon warmed-inference benchmark hygiene](../Guides/apple-silicon/Apple-Silicon-warmed-inference-benchmark-hygiene-guide.md) - Warmed-only benchmark contract, quiet-host gating, and Core ML compile/cache separation.
- [MoE SSD/DRAM prefetch guide triage](moe-ssd-dram-prefetch-guide-triage-2026-06-29.md) - Deep Research Max provenance and ingest decisions for learned expert prefetching, SSD/NVMe measurement, and Stage 0 kill-gate planning.
- [MoE SSD/DRAM prefetch results](moe-ssd-dram-prefetch-results.md) - Staged experiment assumptions, gate decisions, and output artifact pointers for the MoE SSD-tier prefetch plan.
- [MoE expert offload and prefetch prior art](../Guides/moe-expert-offload-prefetch-prior-art-guide.md) - Research-routing guide for expert prediction, offload baselines, and why Apple Silicon UMA needs physical SSD measurements.
- [Apple Silicon NVMe and energy measurement](../Guides/apple-silicon/apple-silicon-nvme-energy-measurement-guide.md) - macOS cache-bypass, `fs_usage`, `powermetrics`, and joules/token evidence requirements for SSD-tier model-weight experiments.
- [Kokoro Irvine M1 paper frontier guide](../Guides/apple-silicon/Kokoro-Irvine-M1-3s-7s-paper-frontier-guide.md) - Lower-end Mac `3s`/`7s` frontier strategy after corrected MLX comparisons.
- [Kokoro M1 vocoder runtime boundary guide](../Guides/apple-silicon/Kokoro-M1-vocoder-runtime-boundary-guide.md) - Runtime-boundary strategy for strict lower-end Apple Silicon wins.
- [Kokoro M1 vocoder partition and boundary guide](../Guides/apple-silicon/Kokoro-M1-vocoder-partition-boundary-guide.md) - Core ML partition and boundary mechanics for strict Kokoro vocoder bodies.
- [Kokoro M1 source/body Core ML guide](../Guides/apple-silicon/Kokoro-M1-source-body-coreml-guide.md) - Source/body implementation path against laishere without adding losing hot-path splits.
- [Kokoro HAR/STFT strict repair and distillation guide](../Guides/apple-silicon/Kokoro-HAR-STFT-strict-repair-distillation-guide.md) - Strict HAR/STFT representation-repair and tiny-adapter triage.
- [Kokoro strict source/HAR representation repair prompt](Kokoro-strict-source-HAR-representation-repair-deep-research-prompt.md) - External research brief for strict source/HAR representation repair, first-layer folding, and tiny adapter calibration.
- [Core ML ANE compiler failure triage](../Guides/apple-silicon/CoreML-ANE-compiler-failure-triage-guide.md) - Execution-plan failure taxonomy and warmed-inference separation.
- [Core ML ANE transformer layout and op compatibility](../Guides/apple-silicon/CoreML-ANE-transformer-layout-op-compatibility-guide.md) - Transferable static-shape and op-surface checks for layout rewrites.
- [Core ML split graphs and multifunction packaging](../Guides/apple-silicon/CoreML-split-graphs-multifunction-packaging-guide.md) - Boundary-count rules for split graph and multifunction package candidates.
- [Core ML ANE temporal escape hatches](../Guides/apple-silicon/CoreML-ANE-temporal-escape-hatches-guide.md) - Stateful temporal caveats for future streaming or iPhone paths.
- [iPhone Core ML device lab runbook](../Guides/apple-silicon/iPhone-CoreML-device-lab-runbook.md) - Device setup, foreground policy, and evidence capture for future physical-iPhone benchmark rows.
