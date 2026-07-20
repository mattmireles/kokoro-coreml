# Experiment Spec: Surgical Decomposition of LFM2.5 for ANE-Accelerated Prefill on Apple Silicon

**Version:** 1.1 (2026-07-20) — adds M1 Mac Mini (power-rail attribution) and iPad Pro M2 (sustained-load primary) to the device matrix; per-device dispatch-evidence rule
**Owner:** Matt Mireles
**Status:** Draft — pending Stage 0 go/no-go
**Prior art this extends:** "Surgical Inference" (Kokoro-82M + Magenta RealTime 2 case studies)

---

## 0. Context for an engineer with zero prior background

Read this section even if you think you know the domain. It explains *why* every constraint below exists so you don't remove one without understanding it.

### 0.1 What "surgical inference" means

Apple Silicon SoCs contain three compute units — CPU, GPU, and the Apple Neural Engine (ANE) — sharing one unified memory (UMA) pool. Apple's Core ML framework normally treats compute-unit placement as an opaque, whole-model decision (`MLComputeUnits.all` etc.). The surgical inference thesis: **decompose an ML pipeline into fixed-shape submodels and assign each submodel to a compute unit per-stage**, instead of letting Core ML place one monolithic graph. Prior case studies (Kokoro-82M TTS, 1.6–2.3× speedup; Magenta RealTime 2) showed this wins for **compute-bound, static-shape stages** and fails for others.

### 0.2 The quantitative spine (do not argue with this; measure against it)

- At batch=1 autoregressive **decode**, throughput ≈ DRAM bandwidth ÷ bytes-per-token. All compute units share the same UMA bus, so op placement cannot beat this wall. Decode is **bandwidth-bound**. We measure it only as a control; we expect **no placement effect**.
- **Prefill** (prompt processing) is compute-bound and fixed-shape-friendly. This is where placement matters and where this experiment lives.
- Statefulness is the enemy of decomposition: a growing KV cache forces dynamic shapes and DRAM round-trips ("state is the cliff"). Small constant-size state (e.g., a short-convolution rolling buffer) is fine.
- ANE has an on-chip SRAM working set (~32 MB observed on M4-class parts). Tensors that exceed it cause DRAM round-trips; in prior work, `.cpuAndGPU` beat `.all` at large tensor sizes (~3× penalty at 30 s inputs for merely *permitting* ANE). Expect a crossover, not a uniform win.

### 0.3 Why LFM2.5

Liquid AI's LFM2 family is a hybrid architecture: the documented LFM2.5-230M layout is **14 layers = 8 double-gated LIV (short) convolution blocks + 6 grouped-query attention (GQA) blocks** [R3, R4]. This is a natural two-class pipeline:

| Block class | Shape behavior | State | Predicted placement |
|---|---|---|---|
| Gated short-conv (LIV) | Fixed per bucket | O(d·k) constant rolling buffer | **ANE candidate** |
| GQA | Fixed at prefill; KV cache grows at decode | O(d·seq) KV cache | **GPU/CPU**, especially at decode |

The framework therefore makes a **falsifiable prediction**: conv blocks admit to and benefit from the ANE; GQA blocks (and all of decode) do not. Confirming or refuting this on a third-party model is the point. Both prior case studies were our own ports; a reviewer will notice. LFM2.5 weights are open under the LFM Open License v1.0 (free for research/non-commercial and commercial < $10M revenue) [R5].

Secondary motivation: Liquid's own materials claim CPU performance "will transfer to accelerators such as GPU and NPU after kernel optimization" [R6] — an unvalidated promissory note — and their posted engineering target is **sub-100 ms time-to-first-token** on edge devices [R7]. TTFT is a prefill metric. If surgical placement moves TTFT, it moves the number they publicly care about.

### 0.4 Critical prior art you MUST diff against before writing code

**CoreML-LLM** (john-rocky) already runs `lfm2.5-350m` on the ANE via Core ML, with published power/thermal data (≈12.7 W ANE vs ≈24.7 W GPU at full decode on Mac; GPU runtimes throttle 50%+ within ~60 s under sustained load while ANE holds) [R8]. **The port is not our contribution.** Our contribution is (a) per-block-class decomposition and placement, (b) the prefill crossover curve, (c) the decode null result as confirmation of the bandwidth wall. Stage 0 task 1 is reading their conversion path so we reuse rather than re-derive it. If their monolithic ANE model already dominates every surgical configuration at every prefill length, that is itself a publishable negative result — record it, don't hide it.

### 0.5 Hardware roles (non-negotiable)

| Device | Silicon / ANE gen | Role | Headline numbers? |
|---|---|---|---|
| Mac Studio (M2 Ultra, 64 GB) | M2 Ultra | Development, export, correctness, ANE *admission* checks | **No** — 800 GB/s bandwidth, 76-core GPU, wall power bias every ratio against the ANE |
| Mac Mini (M1) | M1, 1st-gen ANE (~11 TOPS class) | **`powermetrics` per-rail power attribution** (ANE/GPU/CPU package rails) — mechanistic evidence that surgical configs actually shift watts between units. 68 GB/s + 8-core GPU distort ratios far less than the Studio | Appendix only, clearly labeled |
| iPad Pro 11" 4th gen (M2) | M2, ~15.8 TOPS ANE | **Sustained-load (H4) primary instrument** — only fanless device in the matrix; passive cooling gives clean throttling curves without the phone's aggressive thermal governor. Also adds the M2 ANE generation datapoint. ~100 GB/s bandwidth ⇒ decode wall sits ~2× further out than A17; acceptable since decode is the control | Yes, for H4 |
| iPhone 15 Pro Max (A17 Pro) | A17 Pro | **Primary**: all published latency, TTFT, and energy-per-token numbers | Yes |
| iPhone 12 Pro (A14) | A14, 1st-gen ANE | Optional admission-generality check (redundant with M1 Mini for ANE generation; use only if a mobile-OS 1st-gen datapoint is needed) | Optional |

**Per-device dispatch rule:** Core ML's scheduler is not identical across macOS and iPadOS/iOS builds even on comparable silicon; dispatch decisions can differ per OS build for the same compiled model. Per-op dispatch tables (Xcode Performance Report / Instruments) must be collected **on each device that produces any reported number**. An admission result on the Mini or iPad does not certify the phone, and vice versa. No dispatch table for that device ⇒ no claim from that device.

---

## 1. Hypotheses

- **H1 (admission):** LFM2.5's double-gated short-conv blocks compile to and are scheduled on the ANE under fixed/enumerated shapes with fp16 activations.
- **H2 (prefill win):** There exists a prompt-length regime in which surgical placement (conv→ANE, GQA→GPU) beats every homogeneous placement (`all-ANE`, `all-GPU`, `CPU-only`, `.all`) on prefill latency and/or energy-per-prompt-token on A17 Pro.
- **H3 (decode null, control):** Batch=1 decode tok/s is placement-invariant within noise (±5%) across all configurations. A significant placement effect here would falsify the bandwidth-wall model and must be investigated, not celebrated.
- **H4 (sustained-load):** Under ≥10 min continuous prefill-heavy load (battery, no case, controlled ambient), ANE-weighted configurations degrade less than GPU-weighted ones (replicating [R8] in the surgical setting). **Primary device: iPad Pro M2** (fanless, big passive sink ⇒ clean curves); **secondary: iPhone 15 Pro Max** (aggressive governor ⇒ worst-case). If the ANE/GPU gap shrinks on the iPad relative to the phone, report the gap as a function of thermal headroom — that boundary characterization is part of the finding, not noise.

## 2. Non-goals

- No decode optimization, speculative decoding, or KV-cache tricks.
- No training/fine-tuning; instruct checkpoint used as-is.
- No Android/Snapdragon comparison.
- No Core AI (`.aimodel`) port in v1 — CoreML only. Core AI requires OS 27+ and is a follow-up (§9).

---

## 3. Stage 0 — Admission gate (1–2 days, $0 marginal cost)

Purpose: kill cheaply. Modeled on the MoE-prefetch experiment discipline: stage 0 killed that project before a dollar of real work.

### Tasks

1. **Diff prior art (½ day).** Read CoreML-LLM's LFM2.5 conversion path [R8]: how it handles the conv state, KV cache (Core ML stateful models? I/O tensors?), shape strategy, and quantization. Write a one-page summary. Decision: reuse their converter or write ours with `coremltools`.
2. **Single-block exports.** From `LiquidAI/LFM2.5-350M` [R5] (fall back to `LFM2.5-230M` [R4] if 350M layer layout complicates isolation), extract via PyTorch:
   - one double-gated LIV conv block,
   - one GQA block (prefill form: no cache read, full-sequence attention).
   Export each with `coremltools` ≥ 8.x, fp16, **enumerated shapes** over buckets {128, 256, 512, 1024, 2048} tokens [R9]. Enumerated shapes are required: range-flexible shapes historically demote ops off the ANE; enumerated shapes can remain ANE-eligible [R10].
3. **Admission check.** Xcode Core ML **Performance Report** per exported block on (a) the Mac's ANE, (b) iPhone 15 Pro Max. Record per-op dispatch (ANE/GPU/CPU) for every bucket. Admission is a **toolchain property**, so the Mac result is meaningful here — this is the one place the Studio is a valid instrument.
4. **Numerics.** Compare fp16 CoreML block outputs against fp32 PyTorch reference on 32 real prompts (max-abs and cosine). Use the real trained checkpoint only — **never** a `tiny-random-*` CI fixture; randomly initialized weights make every downstream metric meaningless (established failure mode from the MoE stage-1 postmortem).

### Kill gates (any one triggers stop-and-report, not workarounds)

- **G0a:** <80% of conv-block ops dispatch to ANE at bucket ≤512 on A17. (If the LIV gating decomposes into ANE-hostile ops — e.g., unsupported elementwise patterns or layouts — the thesis has no substrate in this model. Write it up as a negative admission finding; that is still a result.)
- **G0b:** Enumerated shapes rejected / forced to range shapes by the converter for the conv block.
- **G0c:** fp16 divergence > 1e-2 max-abs on block outputs with no identifiable fixable op.

Deliverable: `stage0-report.md` with per-op dispatch tables per bucket per device, numerics table, go/no-go recommendation.

---

## 4. Stage 1 — Full decomposition + correctness (3–5 days, Mac)

### 4.1 Decomposition

Partition the 14-layer stack into contiguous **segments by block class** (conv-run vs GQA-run), preserving execution order. Each segment becomes one Core ML model with:

- Inputs: hidden states `[1, L_bucket, d]` (fp16), plus segment state inputs.
- Conv state: rolling buffer `[1, d, k-1]` passed as explicit I/O tensor (constant-size; ANE-friendly). Do **not** use Core ML stateful models for conv state in v1 — explicit I/O keeps admission behavior predictable and matches the MRT2 methodology. (Stateful models [R11] are an ablation, §8.)
- GQA prefill segments: emit K/V for the processed prompt as outputs (they seed the decode cache); no cache inputs at prefill.
- Embedding + LM head: separate small models; expected CPU/GPU; not a bottleneck at 350M scale, but measure.

Rationale for segments over per-layer models: per-layer submodels maximize placement flexibility but pay a per-`predict()` dispatch overhead (measured ~ms-scale per call in prior work); class-contiguous segments amortize it. If LFM2.5-350M interleaves classes finely (verify against config in [R5]; the 230M layout is 8 conv + 6 GQA but the *interleaving* must be read from the checkpoint config, not assumed), fall back to a hybrid: fuse adjacent same-class layers, accept more segments.

### 4.2 Orchestrator

Swift harness (macOS/iOS shared core):

- Loads segment models with per-segment `MLModelConfiguration.computeUnits`.
- Runs prompt → embedding → segments in order → logits.
- **End-to-end equivalence test:** greedy-decode 64 tokens on 32 prompts; token-exact match against fp32 PyTorch reference (allow fp16-explainable divergence only after logit-level inspection; log every mismatch).
- Timing hooks: per-segment latency (signposts), end-to-end TTFT.

### 4.3 Quantization

v1 runs **fp16 weights** end-to-end. Rationale: quantization (W4/W8 palettization via `coremltools.optimize` [R12]) changes both bytes-moved and ANE admission behavior simultaneously — two variables. One variable at a time: placement first at fp16, quantization as a follow-up axis (§8). 350M fp16 ≈ 700 MB weights; fits phone RAM comfortably.

### Kill gates

- **G1a:** Segment-boundary I/O overhead > 30% of total prefill time at bucket 512 with all segments on GPU (i.e., decomposition tax swamps any possible placement win). Measure this explicitly: monolithic-GPU vs segmented-all-GPU, same bucket.
- **G1b:** End-to-end token mismatch not attributable to fp16.

Deliverable: working harness + `stage1-report.md` (equivalence results, decomposition tax measurement).

---

## 5. Stage 2 — Measurement on device (4–6 days: iPhone 15 Pro Max primary; iPad Pro M2 for H4; M1 Mini for rail attribution)

### 5.1 Configuration matrix

One variable at a time. Fixed: model (LFM2.5-350M fp16), prompts, OS version (record build), airplane mode, screen brightness minimum, battery 60–80%, no case, 10-min cooldown between thermal runs, ambient logged.

| Config | Conv segments | GQA segments | Purpose |
|---|---|---|---|
| C1 | CPU | CPU | Floor / llama.cpp-comparable |
| C2 | GPU | GPU | Homogeneous GPU baseline |
| C3 | ANE-permitted (`.cpuAndNeuralEngine`) | same | Homogeneous ANE baseline (≈ CoreML-LLM monolith; also run their build as external baseline) |
| C4 | `.all` | `.all` | "Let Core ML decide" baseline |
| **C5** | **ANE** | **GPU** | **The thesis** |
| C6 | GPU | ANE | Inversion control — must lose to C5 if the mechanism is what we claim |

External baselines, same prompts, same device: llama.cpp Q4_0 GGUF [R13] (note: different precision — report as context, not head-to-head) and CoreML-LLM `lfm2.5-350m` [R8].

### 5.2 Metrics

- **Prefill latency** per bucket {128, 256, 512, 1024, 2048}, N=20 runs each, report median + IQR; cold vs warm separated (first-run ANE compilation excluded from steady-state numbers, reported separately).
- **TTFT** = prefill + first decode step, vs the 100 ms line [R7].
- **Decode tok/s**, 128 tokens, all configs — the H3 control.
- **Energy:** Instruments Energy Log + battery-drain deltas over fixed 500-prompt batches → J/prompt-token per config (iPhone, headline). **Rail attribution (M1 Mini, appendix):** run C2/C3/C5/C6 under `powermetrics` and report ANE/GPU/CPU package power per config — the mechanistic evidence that C5's latency delta corresponds to watts moving from the GPU rail to the ANE rail rather than a scheduler artifact. One figure, appendix, non-headline.
- **Sustained load (H4):** 10-min continuous prefill loop (bucket 512), throughput sampled per 30 s, configs C2/C3/C5. **Primary: iPad Pro M2; secondary: iPhone 15 Pro Max** (see H4). Log `ProcessInfo.thermalState` continuously on both.
- **Peak memory** per config (Instruments Allocations / `os_proc_available_memory`).

### 5.3 Analysis

- Prefill latency vs bucket, per config, on one plot → the **crossover curve**. This figure is the paper.
- Per-segment attribution: which segments moved, by how much — mechanism, not just aggregate.
- H3: one-way check of decode across configs; anything outside ±5% gets a root-cause before write-up.
- Honest accounting of decomposition tax (G1a number) inside every C5/C6 total.

### Success criteria (pre-registered — write these down before running)

- **Strong:** C5 beats best homogeneous config by ≥15% prefill latency or ≥20% energy at ≥2 buckets, and C6 does not.
- **Weak/publishable-negative:** C3 monolith wins everywhere → "monolithic ANE placement suffices for hybrid conv/GQA models at 350M scale; decomposition tax exceeds placement benefit" + the decode null. Still a real finding; still write it.

---

## 6. Deliverables

1. `stage0-report.md`, `stage1-report.md`, full results tables (CSV) + plots.
2. HF repo `mattmireles/lfm2.5-350m-surgical-coreml`: segment models, Swift harness, conversion scripts, README with reproduction steps (mirror the kokoro-coreml / magenta-realtime-2-iphone repo conventions).
3. Findings write-up structured as a third case study for the Surgical Inference / "State Is the Cliff" line: admission findings, crossover curve, decode null, sustained-load result.
4. License note in README: LFM Open License v1.0 attribution [R5]; our code MIT.

## 7. Timeline & effort

| Stage | Effort | Gate |
|---|---|---|
| 0 Admission | 1–2 days | G0a–c |
| 1 Decomposition | 3–5 days | G1a–b |
| 2 Measurement | 3–5 days | pre-registered criteria |
| Write-up | 2 days | — |

Total ≈ 2–3 engineer-weeks. Any gate failure ends in a written negative-result report, not silent abandonment.

## 8. Follow-up axes (explicitly out of v1 scope)

- W8/W4 quantization × placement interaction (bytes-moved lever) [R12].
- Core ML **stateful models** (iOS 18+) for KV/conv state vs explicit I/O — ANE admission interaction [R11].
- LFM2.5-1.2B (does the SRAM working-set cliff bite earlier?).
- A14 (iPhone 12 Pro) generality run.
- Core AI (`.aimodel`, `coreai-torch`) port once OS 27 tooling stabilizes — pre-existing prior-art framing matters here.

## 9. Known risks

- **LIV op lowering.** The double-gated conv may lower to op sequences the ANE compiler rejects or splits; per-op dispatch tables (Stage 0) are the diagnostic. Manual re-expression of the gating (à la Apple's ANE transformer guidance [R14]) is permitted only if it is numerics-preserving and documented.
- **Enumerated-shape memory blow-up:** each enumerated shape may precompile a variant; watch model load time and disk size.
- **Scheduler opacity:** `.cpuAndNeuralEngine` is a *permission*, not a command; Core ML may silently fall back. Every headline number must be accompanied by its per-op dispatch evidence — no dispatch table, no claim.
- **Thermal confounds on phone:** enforce the cooldown protocol; log `ProcessInfo.thermalState` continuously; discard runs that enter `.serious`.

---

## References

- [R1] Surgical Inference paper (internal; Kokoro-82M and MRT2 case studies). Ports: https://huggingface.co/mattmireles/kokoro-coreml , https://huggingface.co/mattmireles/magenta-realtime-2-iphone
- [R2] HeteroInfer: "Characterizing Mobile SoC for Accelerating Heterogeneous LLM Inference," arXiv:2501.14794 — GPU/NPU partitioning on mobile SoCs, 1.34–6.02× speedups; NPU tensor-shape sensitivity. Must appear in related work. https://arxiv.org/abs/2501.14794
- [R3] LFM2 Technical Report, arXiv:2511.23404 (Liquid AI, Nov 2025) — architecture, hardware-in-the-loop search, on-device CPU benchmarks (llama.cpp Q4_0, batch=1). https://arxiv.org/abs/2511.23404
- [R4] "LFM2.5-230M: Built to Run Anywhere," Liquid AI blog (Jun 2026) — 14-layer layout: 8 double-gated LIV conv + 6 GQA. https://www.liquid.ai/blog/lfm2-5-230m
- [R5] LiquidAI/LFM2.5-350M on Hugging Face — weights, config, LFM Open License v1.0. https://huggingface.co/LiquidAI
- [R6] "Introducing LFM2," Liquid AI blog (Jul 2025) — accelerator-transfer claim. https://www.liquid.ai/blog/liquid-foundation-models-v2-our-second-series-of-generative-ai-models
- [R7] Liquid AI Edge Inference Engineer posting — sub-100 ms TTFT target. https://jobs.ashbyhq.com/liquid-ai/1ed0e32c-11f4-4f93-bfab-bdfac37f0b1b
- [R8] CoreML-LLM (john-rocky) — LFM2.5-350M on ANE via CoreML; power/thermal sustained-load data. https://github.com/john-rocky/CoreML-LLM
- [R9] coremltools documentation — conversion, compute units, flexible input shapes. https://apple.github.io/coremltools/docs-guides/
- [R10] coremltools Flexible Input Shapes guide — enumerated shapes vs range shapes and ANE eligibility. https://apple.github.io/coremltools/docs-guides/source/flexible-inputs.html
- [R11] WWDC24 10161, "Deploy machine learning and AI models on-device with Core ML" — stateful models, compute-unit dispatch via MPS Graph/BNNS Graph. https://developer.apple.com/videos/play/wwdc2024/10161/
- [R12] coremltools optimization (palettization / linear quantization). https://apple.github.io/coremltools/docs-guides/source/opt-overview.html
- [R13] LFM2.5 GGUF checkpoints + llama.cpp deployment docs. https://docs.liquid.ai/deployment/on-device/llama-cpp
- [R14] Apple ML Research, "Deploying Transformers on the Apple Neural Engine" — ANE-friendly tensor layouts and op re-expression patterns. https://machinelearning.apple.com/research/neural-engine-transformers
- [R15] STAR: "Synthesis of Tailored Architectures," arXiv:2411.17800 — the search framework that produced the LFM2 hybrid; useful for framing per-stage placement as a term in a hardware-aware search objective. https://arxiv.org/abs/2411.17800
