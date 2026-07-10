# Core ML vs MLX Scheduling for 1D ConvTranspose & ISTFTNet Vocoders on Apple Silicon: A Field Guide Anchored to the Config F Benchmark Loss

June 6, 2026

> **Scope:** Fixed per-inference cost, hot-path handoffs, and ANE residency for
> ISTFTNet/Kokoro vocoders on Apple Silicon — why Config F loses short buckets to
> laishere, why MLX can win microbenchmarks without touching the ANE, and concrete
> placement fixes (dual-output anchor, fp32 tail, collapsed boundaries). Pair with
> the compute-unit scheduling guide for generic silent-fallback mechanics; pair
> with the Kokoro M1 vocoder boundary brief for live bakeoff targets.

## Related Documentation

- **[Core ML compute unit scheduling](CoreML-Compute-Unit-Scheduling-guide.md)**:
  `MLComputeUnits`, silent fallback, powermetrics, Instruments, and LLDB — the
  generic scheduling layer this guide applies to vocoders.
- **[Kokoro M1 vocoder boundary research brief](../../Notes/Kokoro-M1-vocoder-boundary-research-brief.md)**:
  Live Irvine M1 gap, falsified placement paths, and frontier artifacts for
  beating laishere without repeating failed probes.
- **[Kokoro to Core ML conversion](Kokoro-to-CoreML-conversion.md)**:
  Two-stage pipeline architecture, bucketing, and `(B, C, 1, S)` layout context.
- **[PyTorch MPS and Core ML field guide](pytorch-mps.md)**: MLX/MPS training and
  conversion context; MLX graph-batching dispatch lessons mirror Core ML handoff cost.
- **[ANE optimization plan](../../Plans/ane-optimization-v1.md)**: Generator
  Linear→Conv1d graph surgery and `MLComputePlan` placement gates.
- **Repo tooling:** [`scripts/inspect_coreml_compute_plan.m`](../../../scripts/inspect_coreml_compute_plan.m)
  — native per-op placement summary (macOS 14.4+); pair with the Python snippet
  in **Concrete profiling commands and tools** below.
- **Fixed-cost latency fit:** [`README/Notes/fixed-cost-latency-fit.md`](../../Notes/fixed-cost-latency-fit.md)
  — warmed frontier fit that separates fixed boundary cost from duration-scaled
  generator cost before promoting new optimization families.

## TL;DR
- Config F loses lower-end short/mid buckets because **per-inference fixed cost — ANE program dispatch, IOSurface round-trips, and CPU↔ANE↔GPU handoffs across its surgically-split stages — is amortized by the 30s bucket but dominates the 3s bucket**; the inverse hardware-tier gradient (≈3% to close on M2 Air, ≈32% on M1) is the signature of a fixed cost, not a kernel-elegance problem.
- The fix is to **collapse hot-path handoffs**, not to invent new kernels: keep the vocoder fully ANE-resident with an "anchor output" trick, move the noise/source and iSTFT tail inside Core ML stages (or a single tiny fp32 tail), use fewer/fatter fixed-shape fp16 models, and validate placement with MLComputePlan + powermetrics.
- MLX wins short buckets in some setups not because Metal beats the ANE on compute, but because **lazy evaluation batches the whole vocoder graph into one command buffer**, eliminating exactly the per-op dispatch tax that a multi-stage Core ML design pays — but MLX **cannot use the ANE at all**, runs only on the GPU (suspended for backgrounded iOS apps before iOS 26), and so is the wrong long-term bet for always-on, battery-efficient on-device TTS.

## Key Findings

1. **The loss is a fixed-cost amortization problem, confirmed.** Independent ANE characterization gives a measured per-dispatch fixed cost: the skyfallsin/apple-neural-engine-field-guide reports "the best microbenchmark fit is ~119µs fixed cost + bytes / 78 GB/s for const-weight matvec dispatches" (M3 Max, Qwen3-4B, ~113 ANE dispatches/token). The Orion paper (arXiv 2603.06728, "Characterizing and Programming Apple's Neural Engine") measures a "~2.3 ms IOSurface round-trip overhead per ANE dispatch" that "is amortized during prefill (longer sequences) but dominates for single-token decode" — so decisively that Orion found CPU decode (283 tok/s) beats ANE decode (170 tok/s) for GPT-2 124M "due to per-dispatch IOSurface overhead." Core ML itself adds a 2–4× overhead for small operations where most of the time is XPC + IOKit overhead (~0.095 ms per dispatch). A 3s utterance is the vocoder's "single-token decode": too little compute to hide these costs. A 30s utterance is "prefill": compute dominates and Config F's better ANE residency wins. This is exactly the observed pattern (loses short, wins 30s on every machine).

2. **The inverse hardware-tier gradient is the fingerprint of fixed cost.** Fixed dispatch/handoff cost is roughly constant in absolute terms, but it is a *larger fraction* of a small total on slower hardware. M1 has fewer ANE cores, lower clocks, and less memory bandwidth than M2, so the same handoff overhead is a bigger share of M1's tiny 3s budget (needs 32.5% improvement, 233.6 vs 176.3 ms) than of the M2 Air's (4.2%, 148.0 vs 142.0 ms). At 30s, compute dominates on both and the gradient disappears — Config F wins 30s on M1 by ~9% (1959.4 ms) and on M2 Air by ~5% (1404.8 ms).

3. **Config F's architecture maximizes handoffs in the hot path; laishere minimizes them.** Config F interleaves Swift/Accelerate DSP (the hn-nsf harmonic source in double precision, and the iSTFT tail) *between* Core ML model calls. Each Swift↔Core ML boundary is a synchronization point and a fresh dispatch. laishere/kokoro-coreml instead keeps the SineGen + STFT + noise convolutions **inside a Core ML "Noise" stage** and the whole generator inside a Core ML "Vocoder" stage, leaving only a ~2 ms fp32 "Tail" (conv_post + exp + sin + iSTFT) off-ANE. Fewer boundary crossings in the hot path means less fixed cost to amortize — which is precisely why a *7-stage* pipeline (laishere) can beat a *5-stage* one (Config F) on short buckets: stage count matters less than hot-path handoff count and ANE residency.

4. **MLX's advantage is structural, not computational.** MLX has native `mx.conv_transpose1d` (contributed by Max-Heinrich Laves) and native FFT (`mx.fft.rfft`/`irfft`), and mlx-audio's Kokoro vocoder (`mlx_audio/tts/models/kokoro/istftnet.py`) uses them directly with an FFT-based iSTFT (`MLXSTFT` class predicting magnitude via `mx.exp` and phase via `mx.sin`, then `istft`). Crucially, MLX's lazy evaluation defers execution until `mx.eval()`, and `mx.compile` traces the graph once and fuses adjacent operations into a single Metal command buffer, turning ~120 dispatches into 1 (a reported 73% cut at batch=1). MLX maintainer awni demonstrated in MLX Issue #2180 that the apparent slowness of MLX `conv_transpose1d` is a synchronization artifact: per-op synced on M3 it shows "mlx conv_transpose1d: 5.282 ms" vs "torch(mps) conv_transpose1d: 2.912 ms," but synced once after the loop on M4 Max MLX is faster — "torch(mps) conv_transpose1d: 0.503 ms / mlx conv_transpose1d: 0.123 ms" — with the note that "in a realistic workload you should never eval the graph after each op." This is the same fixed-cost lesson from the other direction: batch the graph, pay the dispatch once.

5. **MLX cannot touch the ANE.** MLX runs on Metal GPU + CPU only; Apple exposes the ANE solely through Core ML. Even the M5 "Neural Accelerators" MLX now uses are GPU-resident matmul units, not the ANE — corroborated by the AtomGradient/hybrid-ane-mlx-bench project, which found that on macOS 26.3 "CoreML compute_units=ALL routes to GPU, not ANE (ANE power ≈ 0W)," while genuine ANE batch dispatch via a private API reached "268 tok/s (0.8B), an 11.3× speedup over sequential dispatch" and cut prefill GPU power "from 62.05W to 0.22W (282× reduction)." For backgrounded iOS apps before iOS 26, GPU work is suspended, and in-flight Metal command buffers can be aborted — a correctness/robustness hazard for a TTS engine. So MLX can win a warm-inference microbenchmark on a foreground Mac while being the wrong choice for shipping always-on iPhone TTS.

6. **Silent ANE fallback is the dominant Core ML failure mode for vocoders.** Instance/layer norm, dynamic shapes, fp32 ops, broadcast-heavy source terms, and unsupported ops cause Core ML to silently spill to GPU/CPU, and "ANE → CPU → ANE → CPU" ping-pong happens because switching ANE↔CPU is cheaper than ANE↔GPU. laishere documents the canonical trap: splitting the iSTFT tail off naively made the truncated body's `[1,128,T]` output "make the scheduler bail (168/617 ops to CPU, 353 ms vs 44 ms)" — an 8× regression from one bad boundary.

## Details

### Why surgical per-stage placement backfires on small workloads
Config F's thesis — "Apple Silicon isn't one chip, it's three" — is correct for throughput on big workloads and is why it wins every M2 Studio bucket (50.7/95.5/125.7/185.8/383.9 ms) and every 30s bucket. But surgical placement means surgical *boundaries*, and every boundary is a fixed cost: a Core ML `predict()` call (XPC + IOKit ~0.095 ms, plus the 2–4× small-op Core ML tax), an ANE program activation, and an IOSurface materialization (~2.3 ms round-trip on the measured M4-class part). Config F's hot path crosses the Swift/Core ML boundary at least twice more than laishere's (hn-nsf source, then iSTFT tail), and on a 3s bucket the *entire* vocoder compute is on the order of tens of milliseconds, so two or three extra fixed-cost crossings are a double-digit percentage. On 30s, the same crossings are noise.

The rejected paths confirm this reading:
- **Exact decoder/vocoder split lost** because split-boundary + synchronization overhead exceeded body compute savings — i.e., adding a boundary added fixed cost.
- **Strict generator noise/body split lost** once exact source generation was included — the source tensors aren't free; computing them is another dispatch.
- **Flexible RangeDim was much slower** — dynamic shapes force ANE off or trigger re-specialization, and laishere independently found "two `RangeDim` streams merging in one model causes 141+ GPU ops (6× slowdown)."
- **Palettization/LUT was slower** — decompression cost, and it "did not fix device placement," because the bottleneck was never weight bandwidth on short buckets; it was dispatch.
- **Toolchain-only rebuild didn't help** — because the problem isn't the compiler, it's the graph's boundary structure.

These failures are all consistent with "the binding constraint is fixed per-inference cost and ANE residency, not kernel/graph elegance."

### laishere/kokoro-coreml design choices, decoded
laishere ships 7 mlpackages (Albert, PostAlbert, Alignment, Prosody, Noise, Vocoder, Tail) with explicit per-stage precision and units. Reported 25× real-time on M4 Mac Mini (28s audio in ~1.1s) and 16.9× on iPhone 16 Pro. The load-bearing tricks:
- **"Force `CPU_AND_NE` first."** Don't trust `ALL` — it "may silently spill ANE-eligible ops to GPU." Set CPU_AND_NE and if ops fall off ANE, fix the *graph*, not the flag.
- **Vocoder dual-output anchor.** Keep the full vocoder graph on ANE in fp16; add `x_pre` as a second output and *discard* the original audio output. The discarded output "is a graph anchor that keeps the scheduler committed to ANE." The real audio is produced by a separate fp32 Tail.
- **Noise stage isolated in fp32 + ALL.** SineGen does `cumsum(F0/SR)*2π*300 → sin()`; in fp16 the accumulated phase overflows and correlation collapses (0.94→0.82). Splitting it out also dodges the dual-RangeDim GPU blowup.
- **Cos-Snake.** Use `sin²(αx) = (1 − cos(2αx))/2` — "slightly faster on ANE; quality identical."
- **Int8 palettization only where audio is discarded.** Palette artifacts on the vocoder never reach the listener because its audio output is thrown away; the fp32 Tail that actually makes sound is unpalettized.
- **Avoid GPU on iOS** entirely (backgrounding suspends it pre-iOS 26).

The distillation "detour" is the most important negative result for ConvTranspose: laishere suspected the `ConvTranspose1d` upsamples were the ANE blocker, distilled into a student using **nearest-neighbor upsample + Conv1d**, and found "the distilled student compiled, but most ops still wouldn't schedule onto ANE." The real blocker was the dual-RangeDim noise stream, not the transposed convolution. **Lesson: ConvTranspose1d representation was a red herring for ANE residency; the scheduler blocker was dynamic shapes.**

### How each op type schedules across CPU/GPU/ANE
- **1×1 Conv / Conv1d(k=1):** The ANE's native idiom. Apple's own guidance (ml-ane-transformers, "Deploying Transformers on the Apple Neural Engine") is to replace every `nn.Linear` with a 1×1 conv; Config F already does this (48→0 linear ops in MIL). Lands on ANE.
- **Conv1d (k>1):** ANE-friendly when fixed-shape and fp16.
- **ConvTranspose1d:** Supported (coremltools fixed its lowering, e.g. in 8.2 "Fixes in lowering of batch_norm, ConvTranspose1d, randn") and can land on ANE, but is *not* the residency blocker people assume.
- **Normalization:** Layer norm has a native ANE path; **instance norm / AdaIN are the classic fallback triggers** and, in feedback/streaming settings, accumulate fp16 error (PocketTTS's Mimi decoder produced "audible periodic beeping" on ANE from 23 streaming state tensors; lesson: stateful models with feedback should avoid ANE).
- **Elementwise (Snake/cos/exp/sin):** Cheap and ANE-eligible, but `exp` in the iSTFT magnitude path amplifies fp16 accumulation error (ANE accumulates in pure fp16, unlike GPU/PyTorch fp32 accumulators), which is why the tail must be fp32.
- **iSTFT:** No native Core ML op; must be expressed as conv/matmul inside the graph or done off-graph in Accelerate/Metal.

### The right tensor layout
Apple's ANE wants **4D, channels-first `(B, C, 1, S)`** — sequence on the last (unpacked) axis so accesses are contiguous; per the MobileI2V analysis of Apple's guidance, "making a channel dimension the last axis inflates buffers and hurts L2 residency." For a 1D vocoder this means representing `[B, C, T]` as `[B, C, 1, T]` (1D-as-2D), keeping channels on the C axis, and **minimizing reshape/transpose** between ops (Apple's Principle 3: minimize memory copies). Every layout transpose between ops is both a memory copy and a potential fallback/sync boundary. MLX, by contrast, is channels-last (`ConvWeighted` stores weights `(out, k, in)` and `sanitize()` transposes PyTorch `(0,2,1)`), which suits Metal but is the opposite of ANE's preference — another reason the two runtimes can't share a layout.

### ConvTranspose1d representation alternatives, ranked for Apple hardware
1. **Native ConvTranspose1d (Core ML/ANE) or `mx.conv_transpose1d` (MLX).** Best default now that lowering is fixed. On MLX, grouped/depthwise transposed conv is *not* supported and mlx-audio works around it with `conv_transpose1d(groups=dim_in) + mx.pad` (the istftnet.py comment: "Manually implement grouped ConvTranspose1d since MLX doesn't support groups") — a caution if your decoder uses grouped upsampling.
2. **Resize/nearest-upsample + Conv1d.** Upsample appears to run on ANE (there's an `Espresso::ANERuntimeEngine::upsample_kernel`), and this is the HiFi-GAN-friendly form — but laishere's experiment shows it does **not** by itself improve ANE residency. Use it for anti-aliasing/quality reasons, not as an ANE unlock.
3. **Zero-insert + Conv1d.** Mathematically exact to ConvTranspose but materializes a large sparse tensor — more memory traffic; avoid on bandwidth-bound short buckets.
4. **PixelShuffle / depth-to-space (reshape + permute).** Implementable in Core ML via reshape+permute (pixel shuffle is "really just a combination of reshape and transpose operations"), but the permute is a layout transpose that can force a copy/fallback; risky for ANE.
5. **Eliminate upsampling entirely (Vocos-style).** Vocos uses **no transposed convolutions** — it keeps constant temporal resolution and upsamples only via a single inverse STFT. Per the Vocos paper (arXiv 2306.00814), "Vocos processes audio up to 13 times faster than HiFi-GAN and approximately 70 times faster than BigVGAN," with CPU throughput 169.63× real-time vs 14.44× for iSTFTNet and 5.84× for HiFi-GAN (A100/EPYC, batch 16, 1s clips). This is the highest-upside *architectural* change (see Speculative section).

### ISTFTNet / iSTFT tail placement (with audio-quality preservation)
The iSTFT must be fp32 for quality (fp16 `exp`/phase accumulation causes hoarseness), which means it leaves the ANE regardless. Three viable placements, warmed-inference-comparable:
- **Inside Core ML as a separate tiny fp32 Tail (laishere):** conv_post + exp + sin + iSTFT in one small mlpackage on CPU/GPU, ~2 ms. Keeps everything in one runtime, easy to benchmark, numerically clean. **Recommended default.**
- **In Swift/Accelerate `vDSP` (Config F today):** `vDSP.FFT`/`vDSP_DFT` with overlap-add. Fast and exact, but each call is a Swift↔Core ML boundary in the hot path — the very crossing that hurts short buckets. Keep it only if you can fuse the source + tail into a single Swift DSP block so there's exactly one boundary, not two. (Watch the vDSP real-FFT packing quirk: DC and Nyquist bins are packed into the real/imag of bin 0, so split-complex multiplies need manual twiddling.)
- **Custom Metal kernel:** Maximum control, but pulls in the GPU (bad on backgrounded iOS) and adds command-buffer setup; only worth it if you're already GPU-resident.

To keep the iSTFT numerically equivalent across placements: fix `n_fft`, `hop`, `win_length`, window type (Hann), centering, and pad mode; do the transform in fp32/double; and validate with a correlation/SNR metric against the PyTorch reference (laishere tracks correlation; >0.99 is a reasonable bar).

### Benchmark methodology for warmed inference only
Your methodology is already correct in excluding compile, first-run cache, model load, and one-time Core ML compilation. To avoid false conclusions:
- **Warm explicitly:** run ≥3–5 discard iterations per bucket before timing; the ANE compile cache and Core ML specialization must be populated.
- **Report medians, not means**, over ≥20 timed iterations per bucket; report p90 too (dispatch jitter is real).
- **Pin runtime buckets** (you do — 3/7/10/15/30s) and use **fixed shapes** (no RangeDim) so no per-call re-specialization contaminates timing.
- **Control thermals:** insert cool-downs between buckets, monitor `ProcessInfo.thermalState`, and discard runs taken under `.serious`/`.critical`. Fanless Macs (M2 Air) and iPhones throttle fast; a "loss" can be thermal.
- **Counterbalance order** (Config F vs laishere alternating) to cancel drift — Config F already does counterbalanced ordering, 5 iterations, warm median; push to ≥20 for stable medians.
- **Separate the fixed cost:** run a minimal/near-zero-length bucket to estimate the per-inference floor, then fit `latency ≈ fixed + k·duration`. If Config F's `fixed` is higher than laishere's, that *is* the diagnosis.
- **Use the same iSTFT placement and audio-quality bar** across competitors, or the comparison is unfair (a Swift fp32 iSTFT vs an in-graph fp16 one is not the same product).

### Concrete profiling commands and tools
- **Xcode Core ML performance report:** drop the `.mlpackage` into Xcode, open the Performance tab; it shows per-op preferred/supported compute device and whether ops are ANE-resident.
- **MLComputePlan (macOS 14+ / iOS 17+):** compile with
  `xcrun coremlc compile foo.mlpackage /tmp/`, then load the compute plan and
  print per-op device + estimated cost (requires a compiled `foo.mlmodelc` path):
  ```python
  import coremltools as ct

  plan = ct.models.compute_plan.MLComputePlan.load_from_path(
      "foo.mlmodelc", compute_units=ct.ComputeUnit.CPU_AND_NE
  )
  program = plan.model_structure.program
  if program is None:
      raise ValueError("Expected an ML Program model.")

  main_fn = program.functions["main"]
  for op in main_fn.block.operations:
      usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
      cost = plan.get_estimated_cost_for_mlprogram_operation(op)
      device = usage.preferred_compute_device if usage else None
      print(op.operator_name, device, cost)
  ```
  This prints lines like `conv, <MLNeuralEngineComputeDevice>, ...` — ground truth
  for placement. For a checked-in wrapper, see
  [`scripts/inspect_coreml_compute_plan.m`](../../../scripts/inspect_coreml_compute_plan.m).
  (freedomtan/coreml_modelc_profling shows the same via undocumented per-op timing,
  including per-backend `est_time` and "invalid" reasons such as "Unsupported tensor
  format: fp32.")
- **Symbolic breakpoint** on `-[_ANEModel program]`: if it's hit, *some* of the model is on ANE; also watch `-[_ANEClient evaluateWithModel...]` (ANE execution time) and `-[MLNeuralNetworkEngine predictionFromFeatures:]` in Time Profiler ("Focus on subtree").
- **powermetrics (engine confirmation):** `sudo powermetrics --samplers ane_power,gpu_power,cpu_power -i 200` — if `ANE Power` stays at 0 mW during inference, you are *not* on the ANE regardless of what the flag says. `asitop` wraps this for a live view.
- **Instruments:** Core ML, Neural Engine, and Metal System Trace templates together; `xctrace record --template 'Metal System Trace'` for command-buffer timing.
- **MLProgram/mlpackage inspection:** Netron to confirm ConvTranspose vs Upsample lowering and to catch broadcast/ND layers that force fallback.
- **MLX profiling:** `mx.metal` trace utilities + Xcode Metal System Trace / GPU Frame Capture; confirm whether the vocoder is one command buffer or many.

## Recommendations

**Stage 1 — Diagnose before rewriting (1 day).** Run MLComputePlan on every Config F stage with `CPU_AND_NE` and confirm ANE residency op-by-op. Run `powermetrics` during a 3s and a 30s render and record ANE/GPU/CPU power. Fit `latency ≈ fixed + k·duration` per device. **Expected result:** Config F's `fixed` term is higher than laishere's, concentrated at the Swift/Core ML boundaries (hn-nsf and iSTFT). If this holds, proceed; if instead an op silently fell to CPU/GPU, fix that first (it may close the gap alone).

**Stage 2 — Collapse hot-path handoffs (highest expected payoff for short buckets).**
- Adopt laishere's **vocoder dual-output anchor**: keep the full generator on ANE in fp16, emit `x_pre` (or a pre-tail feature) as a second output, and run a **single** small fp32 Tail for conv_post + iSTFT. This replaces Config F's *two* Swift boundaries (source + tail) with at most one model boundary.
- **Move the hn-nsf harmonic source inside Core ML** as its own fp32 stage (laishere's "Noise" pattern) so the source is computed in-runtime rather than via a Swift round-trip — unless your fit shows the source is genuinely free, in which case keep it Swift but **fuse source + iSTFT into one Swift block** so there is exactly one boundary.
- Switch Snake to **cos-Snake** (`(1−cos(2αx))/2`) — free, slightly faster on ANE.

**Stage 3 — Per-cell tactics for the losing buckets.**
- **M2 Air 3s/7s/10s/15s (need 3.5–5.5%):** Stages 1–2 alone should close these — a single removed boundary at ~2.3 ms IOSurface + ~0.1 ms dispatch is already several percent of a 148 ms render. Verify with the refit.
- **M1 3s (need 32.5%, 233.6→<176.3 ms):** the hardest cell; one boundary won't be enough. Stack: dual-output anchor **and** in-graph source **and** confirm zero GPU ops (M1's ANE↔GPU handoff is the most expensive). Consider a **3s-specific fused model** (Decoder+Generator merged on ANE, single predict) — Config F already exports per-bucket models, so a merged 3s variant is low-risk. If still short, accept that M1 3s may need the Vocos-style architectural change (Stage 4).
- **M1 7s/10s/15s (need 11–25%):** the per-bucket merged-model + handoff collapse should close most of this as the duration term grows and the fixed fraction shrinks.
- **iPhone 12 Pro (no Config F result yet):** measure first with the Stage-1 workflow; the A14 ANE behaves like M1-class, so expect M1-like gaps and apply the M1 prescription. Do **not** adopt MLX here — GPU suspension on backgrounding is disqualifying for a TTS feature.

**Compute-unit choice:** default **`.cpuAndNeuralEngine`** per stage (never rely on `.all`, which silently spills). Use `.cpuAndGPU` *only* for a stage that genuinely can't reach the ANE and only on macOS/foreground. Never `.all` for the vocoder.

**Memory layout:** ensure the generator is exported in `(B, C, 1, T)` channels-first and audit MIL for stray transposes between conv blocks; remove any reshape that isn't load-bearing.

**Benchmarks that would change the recommendation:** if the refit shows Config F's `fixed` term is *already* equal to laishere's, the problem is compute/quantization, not handoffs — then revisit fp16 residency and per-bucket model fusion instead. If powermetrics shows ANE at 0 mW on a stage, that stage's silent fallback is the whole story.

## ConvTranspose1d / iSTFT DO / AVOID

| DO | AVOID |
|---|---|
| Keep the full vocoder ANE-resident; use a discarded "anchor" output to pin the scheduler | Splitting the iSTFT off naively (truncated `[1,128,T]` output → 168/617 ops to CPU, 8× slower) |
| Force `.cpuAndNeuralEngine` and fix the graph until ops stay on ANE | Trusting `.all` / the flag without verifying via MLComputePlan + powermetrics |
| fp16 on the ANE body; fp32 only for the tiny iSTFT/noise tail | Whole-graph fp32 (off ANE) or whole-graph fp16 (hoarse audio from `exp` accumulation) |
| Fixed shapes, per-bucket exports | RangeDim / dynamic shapes (6× slowdown from dual-RangeDim; forces GPU/CPU) |
| `(B, C, 1, T)` channels-first; minimize transposes | Channels-last or reshape/permute between conv blocks (copies + fallback) |
| Conv1d(k=1) instead of Linear | Leaving `nn.Linear` in the generator (off-ANE) |
| cos-Snake substitution | Assuming ConvTranspose1d is the ANE blocker (it usually isn't) |
| One Swift↔Core ML boundary max in the hot path | Interleaving multiple Swift DSP stages between Core ML calls on short buckets |

## Core ML Scheduling & Silent Fallback Failure Modes
- **The flag lies.** `compute_units` is honored at *compile/load*, not convert time (coremltools issue #1849: a model converted with CPU_AND_GPU reverts to ALL when reloaded), and `.all` will spill ANE-eligible ops to GPU silently. Always verify residency.
- **One bad boundary cascades.** A single unsupported/awkward intermediate output (e.g., `[1,128,T]`) can knock a quarter of the graph to CPU.
- **ANE↔CPU↔ANE ping-pong** occurs because ANE↔CPU switching is cheaper than ANE↔GPU; a few stray ND/broadcast layers cause repeated round-trips. coremltools-4+ converters tend to *prefer* the broadcastable/ND layer types that don't run on ANE — model surgery to older layer types can restore residency.
- **fp16 accumulation in feedback/iterative paths** corrupts audio (beeping/hoarseness) even when single-pass fp16 is fine.
- **`FP16ComputePrecision(op_selector=...)` is broken** — it skips `cast_to_fp16` on selected ops but doesn't insert `cast_to_fp32` at boundaries, so "fp32" ops run fp16 anyway. Use separate models for separate precisions instead.
- **`make_pipeline()` locks shapes** — another reason to keep stages as standalone mlpackages.

## MLX Scheduling: Strengths & Weaknesses for Vocoders
**Strengths:** native `conv_transpose1d` and FFT; lazy eval + `mx.compile` fuse the whole vocoder into one command buffer (kills per-op dispatch tax — the structural reason MLX can win short buckets); unified memory (zero-copy CPU↔GPU); fp16 compute with fp32 accumulation by default (no hoarseness); trivially supports the exact iSTFT (real FFT).
**Weaknesses:** **no ANE** — GPU only (confirmed: CoreML `compute_units=ALL` on macOS 26.3 routes to GPU with ANE ≈ 0W; MLX has no ANE backend), so higher power and GPU contention with app rendering; **GPU suspended for backgrounded iOS apps pre-iOS 26**, and in-flight command buffers can abort (crash risk — see whisper.cpp issue #3531); per-op synchronization is expensive if you `eval()`/`.item()` mid-graph (splits the graph, recompiles); grouped transposed conv unsupported (manual workaround); unbounded memory can kernel-panic the machine. **Verdict:** great for a foreground Mac benchmark and for research; wrong default for shipping always-on, battery-efficient iPhone TTS.

## Caveats
- I could not retrieve a published warmed-inference latency in milliseconds for mlx-audio Kokoro on a *named* low-end device (M1/M2 Air/iPhone) for short utterances; the only figures are RTF ~0.17 (device unspecified, Soniqo) and a competitor's "2.8× faster than mlx-audio on short phrases" claim (MetalRT, vendor benchmark). Generate your own using mlx-audio's built-in `real_time_factor` output before betting on MLX numbers.
- The claim that Config F has more hot-path handoffs than laishere is an architectural inference from both repos' published pipelines, not a measured handoff count; Stage-1 profiling should confirm or refute it directly.
- ANE fixed-cost figures (~119µs + bandwidth term on M3 Max; ~2.3 ms IOSurface and ~0.095 ms XPC on M4 Max) come from M3/M4-class reverse-engineering (skyfallsin field guide; Orion, arXiv 2603.06728); absolute values differ on M1/A14, but the *scaling* (fixed cost a larger fraction on slower silicon) is what matters and is robust.
- `powermetrics` power values are Apple-documented as estimates ("should not be used for any comparison between devices"); use them to confirm engine *usage* (0 vs non-zero), not for cross-device comparison.
- Speculative (clearly marked): a **Vocos-style, transposed-conv-free generator** (constant temporal resolution + single iSTFT upsample, ~13× faster than HiFi-GAN on CPU in the paper's benchmark) is the highest-upside frontier move — it removes the upsampling stack entirely and could make the M1 3s cell winnable — but it is a retrain/redesign, not a graph rewrite, and carries audio-quality and training-stability risk (iSTFTNet itself found replacing *too many* upsampling layers "drastically degrades quality"). Similarly speculative: a single fully-fused ANE generator+tail using an in-graph conv/matmul iSTFT (fp16 body, fp32 final conv) to achieve a true one-dispatch vocoder; unproven for audio quality and worth a spike, not a commitment. Also speculative and worth watching: hybrid ANE batch-dispatch via private APIs (AtomGradient/hybrid-ane-mlx-bench reported 11.3× over sequential dispatch) — not App-Store-safe today but indicative of the headroom Apple leaves on the table behind Core ML's public scheduler.

## What is genuinely new here (vs the rejected paths)
The rejected paths all tried to make the *compute* cheaper (palettization, quantization, splits, distillation, toolchain) or the *shape* flexible (RangeDim) — and all failed because the binding constraint on short buckets is **fixed per-inference cost and ANE residency**, not compute. The genuinely new prescription is to **minimize hot-path boundary crossings and guarantee ANE residency**: the dual-output anchor (keep the vocoder whole on ANE), folding the source/iSTFT into at most one off-ANE boundary, per-bucket *merged* models for the smallest buckets, and a fixed-vs-duration cost fit (`latency ≈ fixed + k·duration`) as the primary diagnostic. That is a different axis of optimization than anything on the rejected list — and notably, it is the same axis (collapse dispatches into one batched graph) that gives MLX its short-bucket edge, achieved within Core ML so you keep the ANE's power and background-execution advantages.
