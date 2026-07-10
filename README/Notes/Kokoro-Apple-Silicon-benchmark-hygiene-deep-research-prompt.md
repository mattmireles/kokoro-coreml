# Kokoro Apple Silicon Benchmark Hygiene Deep Research Prompt

June 6, 2026

Use this as an external-research prompt for paper-grade warmed-inference
benchmark hygiene on owner-operated Apple Silicon Macs and iPhone. The goal is
not generic ML benchmarking advice. The goal is a concrete protocol that makes
Kokoro/CoPro TTS runtime comparisons defensible when lower-end Macs are noisy.

## Context

We are benchmarking a first-party Swift + Core ML Kokoro/CoPro-style TTS
pipeline against popular MLX and Core ML implementations on Apple devices.
Runtime buckets are fixed: `3s`, `7s`, `10s`, `15s`, and `30s`. All comparison
claims must use warmed inference only; Core ML compile/cache time is excluded.

Corrected warmed evidence says MLX is not the current Mac blocker. The remaining
strict competitor is `laishere/kokoro-coreml` on Irvine M1 short/medium buckets.
We need reliable lower-end Mac timing before promoting a candidate to the paper.

Current operational problem:

- lower-end remote hosts can be polluted by `mediaanalysisd`, `mds_stores`,
  Spotlight indexing, Photos/media analysis, thermal state, or background login
  agents;
- `launchctl bootout gui/<uid>/com.apple.mediaanalysisd` can fail under SIP;
- `mdutil -i off /` may require admin/root;
- a run that looks fast under load is not paper-grade evidence;
- we need repeatable quiet-host gates and a record of rejected/noisy attempts.

## Primary Research Goal

Design a paper-grade Apple Silicon benchmark hygiene protocol for warmed Core ML
and MLX inference on local and SSH-controlled Macs, plus iPhone where relevant.
The protocol must distinguish true inference speed from compile/cache warmup,
background OS work, thermal throttling, power-state shifts, and measurement
artifact.

## Questions To Answer

1. What Apple-supported or field-tested steps quiet Spotlight, Photos/media
   analysis, Time Machine, iCloud sync, Xcode/indexing, and other common
   background jobs before benchmarking?
2. Which steps require root/admin, SIP changes, Full Disk Access, GUI session
   foregrounding, or user interaction, and which are safe from SSH?
3. What should a "quiet enough" gate measure: load average, CPU percent by
   process, thermal pressure, powermetrics counters, GPU/ANE counters,
   battery/power adapter state, memory pressure, swap, or fan state?
4. How should warmed Core ML runs exclude compile/cache time while still
   avoiding first-run graph specialization artifacts?
5. What N, warmup count, median/trimmed mean, confidence interval, and
   outlier-rejection policy is defensible for `3s`, `7s`, `10s`, `15s`, and
   `30s` TTS buckets?
6. How should benchmark scripts record host model, chip, OS build, power state,
   Core ML model cache state, process list, thermal state, and exact package
   hashes?
7. What methodology prevents misleading wins from different output duration,
   padding, compile/cache behavior, shape specialization, or stale artifacts?
8. What iPhone-specific setup matters: screen lock, foregrounding, Low Power
   Mode, thermal state, airplane mode, developer disk image, app backgrounding,
   and Instruments/xctrace overhead?
9. What commands and tools should be used: `powermetrics`, `pmset`, `top`,
   `ps`, `mdutil`, `launchctl`, `xctrace`, Instruments, Core ML performance
   reports, `MLComputePlan`, and MLX profiling?
10. What evidence bundle would satisfy a reviewer that the comparison is warmed
    inference only and apples-to-apples?

## Output Format

Start with an executive summary of what makes Apple Silicon warmed-inference
benchmarks misleading. Then provide:

- a concrete preflight checklist;
- exact shell commands and expected outputs;
- a quiet-host gate with thresholds;
- repeat/run-statistics recommendations;
- Core ML cache/warmup protocol;
- MLX warmup protocol;
- iPhone-specific protocol;
- evidence bundle template for a paper appendix;
- do-this / avoid-this tables;
- known false-positive benchmark patterns;
- clearly marked speculative or risky operational steps.
