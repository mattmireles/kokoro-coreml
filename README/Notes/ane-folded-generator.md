# ANE Folded Generator Notes

Institutional memory for running the Kokoro generator (iSTFTNet vocoder body)
on the Apple Neural Engine by folding its time axis into rows. Goal: move the
generator, which is 83-89% of end-to-end time on every device, from the GPU to
the ANE without changing the model's math.

**Quick filter:** `grep -n "— Active" README/Notes/ane-folded-generator.md`

Code: `scripts/probe_generator_ane_fold.py` (export + parity),
`swift/Sources/KokoroPipeline/FoldedGenerator.swift` (host fold, masks,
unfold, iSTFT tail), `swift/Tests/KokoroPipelineTests/FoldedGeneratorTests.swift`,
`tools/ane-lab/` (iPhone timing app + `run.py` driver).

---

## What we know (2026-10-22, end of day)

One-screen summary of everything below; details and numbers in each issue.

**Where things stand (30 s clip, end to end; shipping in brackets).**

| Device | Morning | End of day | Path |
| --- | ---: | ---: | --- |
| A14 (iPhone 12 Pro) | 683-726 ms | **488-497 ms** (4.25 s) | e5rt arena; Core ML path ~15-30 ms slower |
| A17 (iPhone 15 Pro) | 379-404 ms | **318-323 ms** (2.51 s) | Core ML (e5rt ties at 30 s, ~25 ms slower at 10 s) |
| M2 Air | 473-493 ms (Botnet on) | **223 ms** Botnet paused, current packages (2.3-2.45 s) | Core ML |
| M1 | 405 ms quiet | **291-306 ms** Botnet paused (1.6-3.3 s) | Core ML ≈ e5rt |

Audio 46 dB vs PyTorch throughout. Share of the estimated physics limit:
A14 ≈ 78-80%, A17 ≈ 65-72%, M1 ≈ 80% of the DRAM-aware floor (~240 ms;
~62% of the compute-only ~185 ms), M2 Air ≈ 65% of its compute-only ~145 ms
floor (223 ms; no DRAM-aware estimate yet). Peak RAM with e5rt is
20-50% below Core ML (Macs ~667 vs ~850 MB at 30 s; A17 generator 0.84 vs
1.77 GB; A14 10 s 265 vs 350 MB).

**What sets the limit now.**
- A14 generator back to back: span ~330 ms, ANE busy ~261, GPU ~245: both
  engines near saturation, ANE I/O-bound (~20 GB/s). A17 is ANE-bound.
- In the pipeline the generator is ~45 ms slower than back to back (cold
  penalty: uniform, not a ramp; no public or private fix found).
- Front stages on the A14 ~105 ms: duration 54 (BERT ANE + LSTM CPU),
  F0Ntrain 25, DecoderPre 28 overlapped with hn-NSF 25.
- Exact global AdaIN statistics are required; fp16/int8 shortcuts and moving
  LSTMs to the GPU change the output (duration rounding).

**What worked (in order of gain).** Vectorized tiled `apply_phase` (2x,
~-60 ms A14); e5rt arena (~-60 ms A14, -20-50% RAM); residual add moved to
the GPU and fused into `stats_add` and the branch sum (ANE -55 ms busy);
fast host setup (`HybridInputs.swift`); r10 package; parallel iSTFT tail,
hn-NSF STFT, vectorized sine, parallel Gaussian noise; reversed apply order;
two chunks in flight for multi-chunk text (SDK).

**What did not work.** Branch fusion, program switching, `.fastPrediction`,
pre-warm loops, audio work interval, QoS values, e5rt skip-fences/prewire,
GPU ↔ ANE event chaining (works, saves nothing), DecoderPre or LSTMs on the
GPU, first-sentence clause split, scalar-load GPU kernels, one huge arena
surface.

**Traps.** zsh does not word-split `set -- $a` (use `bash -s`); libdispatch
traps freeing a semaphore below its initial value; a GPU kernel that writes
x while other threads still read x[0] (stats_add shift) only showed up in the
full generator (19 dB), not in unit tests; packages must match `plan.json`
(now checked at load); ios-bench evicts bucket models on switch unless
`KOKORO_BENCH_KEEP_MODELS=1`.

**Ceiling with the exact model.** ~75-85% of the hardware limit per the
reviewers and the measurements; >90% needs fewer bytes than the exact model
moves (a distilled generator, excluded by CLAUDE.md).

**Next.** On the M2 Air the generator is ~151 ms of a 223 ms 30 s clip
(Botnet paused, current packages); the ~70 ms of front stages (duration 34,
F0Ntrain 16.5, DecoderPre 18) are the next Mac lever.
Decide whether the A14 and
Macs ship the private e5rt path (not App Store-safe); remaining items are
~5 ms each (phase-kernel tuning, first-AdaIN stats sharing, padding).

---

## Issue: Generator ANE speed on device — Active

**First spotted:** 2026-10-21
**Status:** Active

### Summary

The generator (83-89% of end-to-end time) runs off the GPU in two exact forms.
Both are wired into `executeKokoroSynthesis`, the repo runtime
(`KokoroPipeline`), the SDK provider (folded only), `kokoro-bench` and
`ios-bench`.

- **Folded all-ANE** (`kokoro_generator_fold_{N}s`, default when present):
  one Core ML call. Native-`sin` Snake on A17-class chips, degree-3 polynomial
  elsewhere.
- **Hybrid** (`<models>/hybrid/`, opt-in `KOKORO_HYBRID_GENERATOR=1`):
  - ANE runs every conv (53 functions per row count).
  - Metal runs AdaIN statistics, the affine, Snake and the branch joins.
  - Tensors are IOSurface-backed and shared between the engines.
  - Fastest everywhere measured. Reviewers advise keeping it an optional fast
    path, with folded as the fallback (e.g. iOS background, where Metal is
    unavailable).

Quality vs fp32 on device: folded 44.0-45.2 dB, hybrid 46.0 dB (M1).

### Current best (2026-10-22, as of 05:45 PT)

End-to-end wall time, warm, runtime placement (phones: `ios-bench --policy
runtime --hybrid-generator --folded-generator --split-duration --split-f0n`;
Macs: `kokoro-bench --compute-units staged` with all packages present).
Hybrid serves ≥ 7 s, folded 3 s; duration and F0Ntrain split (CPU + ANE);
coalesced phase kernel; iOS spin-wait; DecoderPre in the ANE layout; hn-NSF
noise made during F0Ntrain. Phones at nominal/fair thermal state, 2 rounds
each (A17 rerun on the latest build 2026-10-22 12:10 PT, generator 295-310 ms at 30 s; A14
03:25-04:00 PT). SDK chunks are 10 s on iOS (≈ 350 MB peak on the A14 with an
r4 package), 15 s on macOS.

| Device | Clip | Shipping | Now | Speedup | Peak RAM (now / shipping) |
| --- | --- | ---: | ---: | ---: | --- |
| A17 Pro | 3 s | 201 ms | **58-59 ms** | 3.4x | |
| A17 Pro | 10 s | | **143-159 ms** | | |
| A17 Pro | 30 s | 2,510 ms | **379-404 ms** | 6.4x | |
| A14 (iPhone 12 Pro) | 3 s | 420 ms | **97-98 ms** | 4.3x | 43-46 MB / 422 MB |
| A14 | 10 s | 1,424 ms | **230-242 ms** | 6.0x | 350 MB / 396 MB |
| A14 | 30 s | 4,249 ms | **697-726 ms** | 6.0x | 874-888 MB / 635 MB |
| M1 Mini (ab14, 05:41 PT, load ~2) | 3 s | 355-389 ms | **94 ms** | ~4x | 931-932 MB (process, all buckets) / 1.58 GB |
| M1 Mini | 7 s | 410 ms | **107-111 ms** | ~3.8x | |
| M1 Mini | 10 s | 573-1,049 ms | **149-150 ms** | ~5x | |
| M1 Mini | 30 s | 1.59-3.30 s | **405-406 ms** | 4-8x | 523-553 MB with r6 only at 15 s |
| M2 Air (abair3, 04:05-04:13 PT, load 2.3-4.3) | 3 s | 288-327 ms | **82-87 ms** | 3.6x | 907-959 MB (process, all buckets) / 3.19 GB |
| M2 Air | 10 s | 824-841 ms | **210-220 ms** | 3.9x | |
| M2 Air | 15 s | 998-1,085 ms | **240-304 ms** | 3.8x | |
| M2 Air | 30 s | 2.30-2.45 s | **473-493 ms** | 4.9x | |

Ranges span rounds; Botnet load varies between rounds on the Macs. The
shipping baselines are the pre-change binaries on the same host and run.

- **Rejected** (measured):
  - halo below 8
  - noise branch on the GPU in parallel with the folded graph
  - sampled AdaIN stats
  - dropping the AdaIN zero mask (24 dB)
  - duration BERT/LSTM split
  - an Accelerate LSTM port (the LSTMs are only 27 of 86 ms on CPU at t512;
    BERT dominates)
  - pre-centring AdaIN, deferred (~5% for fp16 risk on unseen voices)
- **Policy:**
  - The hybrid serves buckets ≥ 7 s (`HybridGenerator.minBucketSec`, test
    `testHybridServesOnlyBucketsFromSevenSeconds`). Its ~100 Core ML/Metal
    round trips make it lose at 3 s on the Macs (211/200 ms vs 105-135/103 ms
    folded).
  - On a failed Metal command buffer (iOS background) the executor re-runs
    that clip on the folded generator. Not testable on the host: it needs a
    backgrounded iOS app.
  - SDK opt-in: the manifest lists `coreml/hybrid/plan.json`.
- **Hybrid chains vs RAM** (`KOKORO_HYBRID_CHAINS`, M2 Air, load ~5.6):

  | Chains | 15 s wall | 30 s wall | Peak footprint |
  | ---: | ---: | ---: | ---: |
  | 3 | 433 ms | 772-927 ms | 938 MB |
  | 2 | 412-636 ms | 926-977 ms | 782-787 MB |
  | 1 | 692 ms | 1,200 ms | 641 MB |

  - Default 3. Each stage-2 chain is three ~40 MB tensors.
  - The rest (shared stage tensors, 53 function models, the other stages) is
    ~500 MB.
  - Folded stays the low-RAM option (338-394 MB).
- **Next:**
  - per-function, FLOP-normalised ANE timing to find slow conv types
  - quality check of the hybrid on the phone
  - SDK hybrid wiring with an iOS background fallback
  - A14 when its Botnet worker is idle

### A14 (iPhone 12 Pro) end-to-end, staged policy (2026-10-22 00:10-00:30 PT)

First A14 run; the user released the device from Botnet duty. `ios-bench
--policy staged`, phase-major packages (core 224, rows 2/3/4/6/11), polynomial
Snake for folded (no ANE `sin`), Metal `sin` for the hybrid.

| Clip | Stage | Shipping | Folded | Hybrid |
| --- | --- | ---: | ---: | ---: |
| 30 s | total | 4,249 ms | 1,934 ms | 1,176 ms |
| 30 s | generator | 3,787 | 1,468 | 716 |
| 30 s | duration | 289 | 282 | 285 |
| 30 s | F0Ntrain | 93 | 102 | 102 |
| 30 s | hn-NSF (CPU) | 67 | 69 | 65 |
| 30 s | DecoderPre | 65 | 65 | 46 |
| 10 s | total | 1,424 | 632 | 349 |

- These used the bench's `staged` policy, which runs duration and F0Ntrain
  on CPU+GPU. The runtime does not: `PipelineConstants` puts both on the CPU
  on chips without native ANE `sin`. New `ios-bench --policy runtime` uses the
  runtime's own placement. With it (hybrid ≥ 7 s, folded 3 s):

  | Clip | Total | Generator | Duration | F0Ntrain | hn-NSF | DecoderPre |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | 3 s | **110 ms** | 80 | 17 | 5 | 6 | 5 |
  | 10 s | **291 ms** | 204 | 52 | 14 | 20 | 13 |
  | 30 s | **951 ms** | 733 | 113 | 41 | 63 | 45 |

  vs shipping 420 / 1,424 / 4,249 ms: 3.8x / 4.9x / 4.5x.
- A17 Pro rerun, same build and policy (00:50 PT): 3 s **62 ms** (generator
  45, duration 9, F0N 3), 10 s **188 ms**, 30 s **471 ms** (generator 330,
  duration 52, F0N 25, hn-NSF 47, DecoderPre 34). Shipping 201 / 2,510 ms.
- `ios-bench --hybrid-generator` alone falls back to the GPU generator at 3 s
  (below `minBucketSec`); pass `--folded-generator` too.
- Quality on the A14 not yet checked; peak RAM not captured by ios-bench.

### A14: ~10 ms per Neural Engine call after a GPU step (2026-10-22 01:45-02:45 PT)

- Per-function hybrid profile on the A14 (serial): ms = 10.1 + 0.237 ×
  GFLOP, i.e. ~10 ms fixed per call (52 calls) on top of 4.2 TFLOPS. The M1
  fit is 2.6 ms + 9.8 TFLOPS marginal.
- `tools/ane-lab` on the A14 (new options `--function`, `--io iosurface`,
  `--between-ms`, `--between-gpu`, `--rotate`, `--surface-sets`,
  `--metal-compat`) could not reproduce it: the stage-2 conv `rb3_c1_0` runs
  in 3.0 ms alone, 3.2 with plain I/O vs IOSurface (5.4 pointer-backed),
  ~5 ms after a 5 ms sleep or a full GPU compute rewrite of its input,
  6.1 ms mean rotating 12 programs, 3.3 ms over 24 distinct surface sets.
- In the pipeline the same call costs 15.3 ms, and an immediate repeat
  (`KOKORO_HYBRID_REPEAT=1`) 5.6 ms: a transition cost after the Metal step.
- **Fix: spin-wait on Metal command buffers on iOS** (instead of
  `waitUntilCompleted`). A14 30 s: generator 686-713 → 553-588 ms, end to end
  903-945 → 804-834 ms. QoS user-interactive alone: ~4%. M1: no clear
  difference, so macOS keeps sleeping. `KOKORO_HYBRID_SPIN=0/1` overrides.
  Not testable on the host (device timing).
- With spin-wait the A14 fit is ms = 7.4 + 0.192 × GFLOP (52 calls, 716 ms
  serial ANE; GPU busy ~380 ms serial). Lab after a GPU compute pass: 5.2 ms
  median for `rb3_c1_0` (3.0 ms without), the same when the GPU also touches
  the output surfaces. A 5 ms idle gap costs ~2 ms in the lab, and the
  pipeline's GPU phases last ~10 ms per step on the A14, so the rest looks like
  ANE power gating during GPU phases. Keeping the ANE busy would need more
  concurrent chains (RAM) or keep-alive calls (power); not pursued.

### A17 Pro end-to-end, staged policy (2026-10-21 22:19-22:58 PT)

- `ios-bench --mode ladder --policy staged --arms coreml --keys 3s,30s`,
  iPhone 15 Pro Max cabled; 3 warm-ups + 10 timed per key.
- Shipping uses the flexible GPU generator. Folded adds `--folded-generator`,
  which picks the native-`sin` package on iPhone16,x.

| Key | Folded wall | Folded generator | Shipping wall | Shipping generator |
| --- | ---: | ---: | ---: | ---: |
| 3 s | 100-119 ms | 56-60 ms | 196-208 ms | 149-157 ms |
| 30 s | 0.93-1.08 s | 0.72-0.84 s | 2.47-2.68 s | 2.19-2.38 s |

- 3 s is **1.85x**, 30 s **2.6x**.
- The shipping generator here was slower than October's receipt (1.50 s).
  Both arms ran back to back on the same phone, so the ratio holds.

### A17 Pro: hybrid and duration on the ANE (2026-10-21 23:20-23:40 PT)

Same protocol as above. Stage medians are from `e2e_*.json`.

| Arm | 3 s wall | 30 s wall | 30 s generator | 30 s duration |
| --- | ---: | ---: | ---: | ---: |
| Shipping (flexible GPU) | 201 ms | 2,505 ms | 2,222 ms | 181 ms |
| Folded | 109 ms | 958 ms | 742 ms | 114 ms |
| **Hybrid** (ANE convs + Metal, IOSurface) | **73 ms** | **553 ms** | **369 ms** | 105 ms |
| Folded, duration on CPU+ANE | 75 ms | 898 ms | 730 ms | **61 ms** |

- **Hybrid speedup over shipping on the A17:** 2.7x at 3 s, 4.5x at 30 s.
- **Duration on the ANE works on the A17**, even with the padded graph:
  9.2 ms at 3 s and 61 ms at 30 s. `durationComputeUnits` now picks CPU+ANE on
  A17-class iPhones and keeps other iPhones on the CPU.
- **Not yet verified on this run:** audio quality of the hybrid on the phone.
  The same packages scored 46.0 dB on the M1.

### M1 end-to-end, staged policy, every bucket (2026-10-21 22:00 PT)

`kokoro-bench --compute-units staged --batch`, M1 Mini, Botnet load 3-5, two
rounds alternated, 3 warm-ups + 10 timed per bucket. "Shipping" is the
pre-change binary (GPU generator, duration on CPU+GPU). "New" is the folded
poly generator (v4, plus v5 tuned rows for 7 s and 10 s) with per-size
duration units.

| Bucket | New wall (ms) | Shipping wall (ms) | Speedup |
| --- | ---: | ---: | ---: |
| 3 s | 158 / 168 | 380 / 355 | 2.2x |
| 7 s | 364-390 (v5 rows) | 762 / 638 | 1.8x |
| 10 s | 560-620 (v5 rows) | 931 / 1,049 | 1.6x |
| 15 s | 893 / 839 | 1,590 / 1,538 | 1.8x |
| 30 s | 1,732 / 1,500 | 3,090 / 3,300 | 2.0x |

Peak footprint: new 377-394 MB, shipping 1,595 MB.

**M2 Air**, same protocol (load 2.2-3.3, two rounds, 2026-10-21 22:17-22:26 PT):

| Bucket | New wall (ms) | Shipping wall (ms) | Speedup |
| --- | ---: | ---: | ---: |
| 3 s | 104 / 103 | 302 / 298 | 2.9x |
| 7 s | 248 / 238 | 591 / 590 | 2.4x |
| 10 s | 389 / 384 | 692 / 683 | 1.8x |
| 15 s | 526 / 536 | 1,146 / 1,175 | 2.2x |
| 30 s | 1,107 / 1,144 | 2,485 / 2,462 | 2.2x |

Peak footprint: new 368-450 MB, shipping 3.19 GB.

- **F0Ntrain on the CPU** (`PipelineConstants.f0ntrainComputeUnits`, test
  `testF0ntrainRunsOnCPU`). M2 Air warm, cpuOnly / cpuAndNeuralEngine /
  cpuAndGPU:

  | Bucket | CPU | CPU+ANE | CPU+GPU |
  | --- | ---: | ---: | ---: |
  | t120 | 3.6 ms | 3.6 ms | 26.4 ms |
  | t400 | 11.3 ms | 11.5 ms | 21.0 ms |
  | t1200 | 32.0 ms | 40.0 ms | 65.4 ms |

  The staged policy's CPU+GPU was the slowest at every size. Not yet in the
  A/B tables above.
- **Bucket-tuned rows** (v5): set the core so rows × core equals the bucket
  (7 s → 3 × 187, 10 s → 4 × 200) instead of rounding up to 240-frame rows.
  - Generator: 7 s 299-337 ms vs 395-474; 10 s 462-528 ms vs 521-589.
  - The runtime reads core and rows from the package shape.
- **Duration units by size** (`PipelineConstants.durationComputeUnits`, test
  `testDurationComputeUnitsBySize`):

  | Bucket | CPU | CPU+ANE | CPU+GPU |
  | --- | ---: | ---: | ---: |
  | t64 | 15.9 ms | 83 ms | 25.5 ms |
  | t128 | 28.8 ms | 18.7 ms | 38.1 ms |
  | t256 | 56 ms | 43 ms | 106 ms |
  | t512 | 115 ms | 100 ms | 230 ms |

  - macOS: CPU at ≤64 tokens, CPU+ANE above. iOS stays CPU (ANECCompile
    failures there; the SDK already uses CPU).
- **Rejected: duration split** into a BERT front (ANE) and LSTM back (CPU)
  (`scripts/export_duration_split.py`). Exact in PyTorch, but the front ran at
  60 ms (t64) and 496 ms (t512), slower than the whole model on the CPU. Also
  the LSTMs are not the CPU cost: removing them saves 4.6 of 14.3 ms (t64) and
  27 of 86 ms (t512). BERT dominates.
- **Duration split, ANE layout** (`scripts/export_duration_ane.py`; runtime:
  `SplitDuration.swift`, used by `kokoro-bench` staged mode when the packages
  are in the models dir, `ios-bench --split-duration`; `KOKORO_SPLIT_DURATION=0`
  turns it off). Same cut, but the PL-BERT front is rewritten in the
  ml-ane-transformers layout: (B, C, 1, S) tensors, 1x1 Conv2d for Linear,
  LayerNorm over dim 1, per-head attention with softmax over dim 1 and an
  additive key mask. The back is unchanged but fp16 (fp32 was 1.7x slower).
  - Compute plan: front 952 of 961 ops on the ANE (only mask prep and the
    gathers on the CPU). The whole model: 407 ANE, 80 CPU.
  - Parity: PyTorch fp32 front d_en ≤ 4.4e-5, pred_dur exact on 3 s/10 s/30 s.
    Core ML fp16 front + fp16 back: pred_dur exact on 3 s and 10 s, 1 of 476
    tokens off by one frame on 30 s. The shipping whole model has the same
    1/476 miss.
  - M2 Air, warm medians of 30, 2026-10-22 01:31 PT, load avg 2.9-3.3:

    | Bucket | Front ANE | Back CPU fp16 | Split total | Whole CPU | Whole CPU+ANE |
    | --- | ---: | ---: | ---: | ---: | ---: |
    | t64 | 3.2 ms | 3.4 ms | 6.6 ms | 13.7 ms | 11.1 ms |
    | t128 | 3.4 ms | 5.9 ms | 9.3 ms | 21.7 ms | 15.8 ms |
    | t256 | 5.1 ms | 11.3 ms | 16.4 ms | 40.3 ms | 29.1 ms |
    | t512 | 13.3 ms | 21.9 ms | 35.2 ms | 84.3 ms | 67.0 ms |

  - The split is 1.7-1.9x faster than the best whole-model placement at every
    size. Front alone: 13 ms vs 496 ms for the naive front at t512.
  - **Bug found in integration (fixed):** the fp16 back returns fp16 arrays,
    and the token-to-frame expansion (`alignTokenMajorToFrames`,
    `alignChannelMajorToFrames`) only had an fp32 pointer path: ~280 ms per
    30 s clip through NSNumber subscripting. fp16 pointer path added (test
    `testAlignHelpersReadPaddedFloat16Sources`).
  - **A14 (iPhone 12 Pro), runtime policy, ~02:05-02:20 PT, 2 rounds:** the
    front compiles and runs on the A14 ANE (front ~30 ms, back ~31 ms at
    t512). Duration: 3 s 17 → 10.4 ms, 10 s 54-65 → 25-36 ms, 30 s 127-172
    → 66-79 ms. But the generator ran 20-55 ms slower in both split rounds
    (another ANE program resident), so end to end only 3 s gained (109 →
    103 ms); 10 s and 30 s were within noise.
  - **M1 end to end** (ab10, 02:21-02:27 PT, load 2.8-3.4, hybrid + folded,
    full row ladder):

    | Clip | Whole duration | Split duration |
    | --- | ---: | ---: |
    | 3 s | 104-133 ms | 99-134 ms |
    | 10 s | 176-256 ms | 156-304 ms |
    | 15 s | 237-341 ms | 226-306 ms |
    | 30 s | 502-597 ms | **443-549 ms** |

    Duration at 30 s 85-95 → 44-47 ms; generator unchanged on the M1; peak
    RAM unchanged (927-954 vs 932-933 MB). The split is the default when its
    packages are present (`KOKORO_SPLIT_DURATION=0` to disable).
- **F0Ntrain split, ANE layout** (`scripts/export_f0ntrain_split.py`; runtime
  `SplitF0N` in `SplitDuration.swift`, picked up like the duration split;
  `ios-bench --split-f0n`; `KOKORO_SPLIT_F0N=0` disables). Same recipe: front = the masked shared BiLSTM, fp16,
  CPU; back = both AdainResBlk1d branches in (1, C, 1, T) layout, Conv2d (1, k),
  masked two-pass AdaIN, and the depthwise ConvTranspose upsample as two phases
  (`w1·x`, `w2·x + w0·x[i+1]`) interleaved by nearest upsample and 0/1 masks.
  - Compute plan: back 314 of 314 ops on the ANE at every bucket.
  - Parity: PyTorch fp32 split vs the shipping masked wrapper ≤ 2.3e-4 Hz F0,
    ≤ 7e-6 N (3 s/10 s/30 s). Core ML fp16 split vs PyTorch fp32: F0 mean
    0.22-0.27 Hz, p99 2.0-2.1 Hz, max 2.4-3.4 Hz (shipping whole fp16 CPU:
    mean 0.27-0.34, p99 1.0-1.3, max 1.3-2.7); N max ≤ 0.023 (whole ≤ 0.046).
    The max errors sit on voicing edges (ref ~0-90 Hz next to ~200 Hz).
    Voiced p99 relative error ~1% (17-27 cents) on ANE, 0.9-1.1% on CPU fp16.
  - M2 Air, warm medians of 30, 2026-10-22 02:50 PT, load avg 13-15 (1 min,
    raised by the build just before), interleaved in one process:

    | Bucket | Front CPU fp16 | Back ANE | Split total | Whole CPU |
    | --- | ---: | ---: | ---: | ---: |
    | t120 | 1.2 ms | 1.2 ms | 2.4 ms | 4.2 ms |
    | t280 | 3.1 ms | 1.9 ms | 5.0 ms | 9.4 ms |
    | t400 | 3.8 ms | 2.1 ms | 6.0 ms | 10.5 ms |
    | t600 | 5.5 ms | 3.0 ms | 8.8 ms | 15.0 ms |
    | t1200 | 11.0 ms | 5.8 ms | 17.0 ms | 29.3 ms |

  - 1.7-1.9x faster at every size. An fp32 front costs ~1.7x more (t1200
    19.1 ms) for no visible output gain. First ANE load of each back ~1.0-1.3 s
    (cached afterwards ~20-35 ms).
  - End to end, both splits vs neither (`ios-bench --policy runtime`, hybrid
    + folded, 03:00-03:10 PT, 2 rounds each):

    | Device | Clip | Whole | Split | Duration | F0Ntrain |
    | --- | --- | ---: | ---: | --- | --- |
    | A17 | 3 s | 62 ms | 60 ms | 9 → 8 ms | 3.0 → 2.0 ms |
    | A17 | 10 s | 160-165 ms | 146-149 ms | 26-27 → 16 ms | 8.4 → 6.2 ms |
    | A17 | 30 s | 417-419 ms | **394-399 ms** | 56-57 → 35 ms | 23.6 → 17.8 ms |
    | A14 | 3 s | 110 ms | 100 ms | 17 → 10 ms | 4.9 → 2.9 ms |
    | A14 | 10 s | 290-292 ms | 244-252 ms | 59 → 23-24 ms | 15.7 → 8 ms |
    | A14 | 30 s | 844-850 ms | **736-764 ms** | 128-129 → 58-59 ms | 46 → 22 ms |

    The A14 generator did not slow down this time (earlier duration-only
    rounds showed +20-55 ms). M1 (ab12, load 2.6-3.7): F0Ntrain 38.5 → 21 ms at
    30 s; the M1's end-to-end differences were within generator noise.

- **DecoderPre, ANE layout** (`scripts/export_decoder_pre_ane.py`, packages
  `kokoro_decoder_pre_ane_{3,7,10,15,30}s`; not wired into the runtime yet).
  Same recipe as F0Ntrain applied to the whole DecoderPre (F0/N stride-2
  convs kept as strided Conv2d (1, 3), encode, asr_res, 4 decode blocks with
  the polyphase upsample in the last). Runtime contract unchanged: same input
  names/shapes, fp32 I/O, x_pre (1, 512, 2T) fp32 with contiguous strides.
  - Compute plan (M2 Ultra): 284/284 ops on the ANE; the shipping package is
    also 250/250 ANE, so the win is layout, not placement.
  - Parity: PyTorch fp32 vs the shipping masked wrapper ≤ 4e-5 (|x| ≤ 70).
    Core ML fp16 vs PyTorch fp32 on the valid frames, 3/7/10/15/30 s:
    new max 0.05-0.13, mean 0.0023-0.0027; shipping max 0.05-0.14, mean
    0.0027-0.0045. Equal or better at every bucket.
  - Warm medians of 25-30, interleaved in one process, 90% valid mask,
    2026-10-22 03:16-03:20 PT:

    | Bucket | M2 Air new ANE | M2 Air shipping ANE | M1 new ANE | M1 shipping ANE |
    | --- | ---: | ---: | ---: | ---: |
    | 3 s | 2.3-2.8 ms | 2.6-2.7 ms | 2.9 ms | 4.2 ms |
    | 7 s | 3.7 ms | 5.2 ms | 4.9 ms | 8.1 ms |
    | 10 s | 5.3-5.8 ms | 7.3-7.5 ms | 6.6-7.9 ms | 11.2-11.6 ms |
    | 15 s | 6.4 ms | 12.1 ms | 9.2-10.3 ms | 17.3-17.6 ms |
    | 30 s | **15.9-16.2 ms** | 30.7-31.0 ms | **26.7-31.0 ms** | 43.4-44.9 ms |

    M2 Air load avg 3.7-4.7 (Botnet), M1 (irvine) 2.6-4.3, another agent's
    kokoro-bench overlapped one M1 pass. Shipping on .cpuAndGPU: 26-272 ms,
    not a contender. 30 s on the M2 Air is ~4.7 TFLOPS, near the ANE roof.
  - First ANE load ~1.2-1.7 s (cached afterwards ~30-55 ms).
  - **Integration:** same contract as the shipping package, so it ships under
    the shipping name (`coreml-dpre-ane/kokoro_decoder_pre_{N}s.mlpackage`,
    gitignored; `ios-bench/prepare_resources.sh` prefers it). No code change.
  - A14 end to end (03:25 PT, 2 rounds, both splits on): DecoderPre 45 →
    25 ms at 30 s; with the noise change below, 30 s 713-726 ms, 10 s
    234-242 ms, 3 s 97-98 ms. M1 (ab13, load 2.5-4.3): DecoderPre 50 → 32 ms.
- **hn-NSF noise made ahead of time.** The harmonic source's Gaussian noise
  (9 × 720k values at 30 s, ~28 ms single-threaded on an M1, its slowest
  task) depends only on the seed and the bucket length, so the executor now
  starts `harmonicNoise` when the bucket is chosen and it runs during
  F0Ntrain; `buildHar(noise:)` takes it without a copy (identical values; the
  bit-exact tests pass). A14 hn-NSF at 30 s 65 → 45-47 ms. hn-NSF (CPU) is
  now longer than DecoderPre (ANE) on the A14, so it is the next item on that
  path.
  - Rejected first: generating the uniforms sequentially into the buffer and
    the Box-Muller math in parallel chunks: 64-70 ms on the M1 (two passes over
    26 MB plus nested concurrency), slower than the original.
- **A14 generator: alone vs in the pipeline.** New `ios-bench --mode
  hybrid-check` (inputs copied to Documents/hc/in) runs the generator on the
  frozen 30 s inputs on the phone: **46.0 dB, corr 0.999987, same as the M1**,
  and 430-441 ms back to back vs 573-590 ms inside the pipeline. An idle gap
  before each check run reproduces the gap (150 ms: 518-551 ms; 500 ms:
  546-611 ms). A GPU keep-alive (short fixed kernels until the generator
  starts) fixed the 150 ms idle case in the check (448-483 ms) but not the
  pipeline (10 s 226-229 vs 230-233 ms, 30 s same), so it was removed. The
  remaining suspect is ANE program eviction: duration, F0Ntrain and
  DecoderPre run on the ANE just before the generator's 53 programs.
  (A GPU kernel spinning on a host-written flag tripped iOS's GPU watchdog:
  "Ignored (for causing prior/excessive GPU errors)". Do not do that.)
- **Eviction measured (A14, 04:20 PT):** with no other ANE work in the
  pipeline (duration and F0Ntrain whole on CPU, DecoderPre on the GPU via
  a temporary ios-bench switch, since removed) the generator is 155-159 ms at 10 s and 547-552 ms at
  30 s, vs 183-187 and 576-590 ms normally: ~30 ms. DecoderPre on the GPU costs
  249 ms, so it stays on the ANE. A GPU keep-alive from the start of synthesis
  changed nothing (30 s 683-705 vs 706-715 ms) and was removed.
- **Roofline and timeline (A14, 05:40-06:10 PT; second opinion
  `tmp/second-opinions/brief-20261022-051904/`).**
  - `tools/ane-lab` roofline: a stage-2 1×1 conv (84 MB in+out) 2.8 ms ≈ 30 GB/s,
    the k=3 stage-2 conv 2.9 ms (same: bandwidth-bound), k=11 6.35 ms ≈ 9.3
    TFLOPS. A tiny conv lands on the CPU (0.025 ms): no ms-scale dispatch cost.
  - The hybrid moves ~8 GB per 30 s call (GPU ~4.5 GB, ANE I/O ~3.5 GB), so on
    the A14 the generator's DRAM floor is ~270 ms at ~30 GB/s; back to back it
    runs 420-440 ms (~62%), in the pipeline 526-601 ms (~47%).
  - `KOKORO_HYBRID_TIMELINE=1` (host spans of ANE calls, Metal gpuStart/End):
    back to back span 420-439 ms, ANE busy 336-344, GPU busy 290-310, both
    223-241, neither ~17 ms. In the pipeline span 526-601, ANE busy 407-460,
    GPU 356-365, neither ~20 ms. The pipeline is slower because both engines'
    work takes longer, not because of gaps.
  - Rejected: GPU compute keep-alive and GPU memory-streaming keep-alive
    (16 MB blit copies) from the start of synthesis: 30 s 701-718 vs 710-712 ms.
    Public APIs give no control over the remaining power-state effect.
- **Stage-2 crop (kept, 06:00 PT).** The stage-0 halo (8 frames, 480 stage-2
  samples) is sized for the whole network; stage 2 needs only ~60-120 valid
  halo samples, so it now runs on columns [240, 14,640) of the host's rows
  (`CROP2`, plan `crop2`, `HybridPlan.w2s` = 13,920; Metal `unshuffle` writes
  at the offset, har/sm2/zm2 are cropped on the host, logits are expanded
  back). Exact: fp32 68.0 dB at 30 s and unchanged 63.5/60.3 dB at 10 s/3 s
  (same with and without the crop); Swift 46.1 dB on the M1 and the A14.
  240 is the most the 480-column alignment allows. Speed within noise (M1
  check 292-297 vs 292-298 ms; A14 check 421-432 vs 419-441 ms); stage-2
  tensors are 6.5% smaller.
- Qween findings (read-only review of /Users/mm/Documents/GitHub/Qween):
  the ANE+GPU concurrency tax there was 20-40% on a base M1 from shared DRAM
  bandwidth (ANE prefill 1,232 → ~730 positions/s beside the GPU), smaller on
  high-bandwidth chips; an idle M1 ANE powers off (first call after 5 s idle
  ~260 ms); standalone ANE elementwise ops stream at ~24 GB/s; GPU-touched
  no-copy buffers stay wired. These match the measurements here: the hybrid's
  remaining loss is bytes on a shared bus, not dispatch.
- **Rejected: int8 ANE inputs (halving the GPU→ANE bytes).** PyTorch fp32
  reference with every AdaIN+Snake output fake-quantized to int8 (symmetric,
  scale per row and channel): 30 s audio SNR 26.8 dB vs the unfolded model
  (per-channel scale: 17.0 dB). Far below 44 dB before any fp16 loss.
- **Rejected: freeing stage-1 tensors before stage 2** (and stage-2 branch
  tensors after each call), re-allocating per call: M1 generator-only peak
  743 MB → 1.87 GB and 294 → 323 ms. Dropping our references does not release
  the surfaces (Core ML / Metal keep them), so fresh allocations pile up.
- **Rejected: half-precision AdaIN+Snake apply on the GPU.** 45.0 dB (vs
  46.0) on the M1; A14 2 rounds: 10 s 240-245 vs 230-231 ms, 30 s 683-700 vs
  697-714 ms. No consistent gain.

### Time to first audio: streaming (2026-10-22 06:15 PT)

- `KokoroTTS.synthesizeStreaming(_:voice:options:onAudio:)` hands each
  chunk's audio to the caller as soon as it is synthesized. `PcmStreamJoiner`
  holds back the 5 ms crossfade tail until the next chunk, so the pieces
  concatenate to exactly `PcmJoiner.join`'s output (test
  `PcmStreamJoinerTests.testStreamConcatenationEqualsJoin`).
- `TextChunker` emits one chunk per sentence; `maxChunkSeconds` (10 s on iOS,
  15 s on macOS) only splits longer sentences. So first audio arrives after
  the first sentence (typically 2-6 s of speech: ~100-200 ms on the A14) and
  at most after a 10 s chunk (~230-250 ms), instead of after the whole text
  (A14, 30 s: ~700 ms). Estimated from the per-bucket numbers above, not a
  separate run.

### Distance to the physics floor (2026-10-22 04:30 PT)

Floor model for the 30 s bench input (27.4 s audio): every conv and matmul on
the ANE at its measured marginal rate, the elementwise/AdaIN traffic hidden
under it on the GPU, plus the sequential LSTMs on the CPU (~30 ms). Work:
generator 1.35 TFLOP (unfolded), BERT 0.063, DecoderPre 0.075, F0Ntrain
0.016 TFLOP on the ANE (≈ 1.50 TFLOP). ANE marginal rates come from the
per-function fits (`ms = a + b·GFLOP`): M1 9.8 TFLOPS, A14 4.2-5.2, A17
7-9 (per-primitive lab), M2 base ~13 (assumed: 15.8 vs M1 11 TOPS rating).

| Device | Floor (30 s) | Best now (30 s) | % of floor |
| --- | ---: | ---: | ---: |
| M1 | ~185 ms | 405 ms (quiet) | ~46% |
| M2 Air | ~145 ms | 473 ms | ~31% |
| A14 | ~320 ms | 683-726 ms | ~45% |
| A17 Pro | ~200 ms | 379 ms | ~53% |

- Where the rest goes, M1 generator (quiet, ~300 ms vs a ~138 ms compute
  floor): fixed cost per ANE call (fit 2.6 ms × 53 calls ≈ 140 ms), fold
  overhead (1.78 vs 1.35 TFLOP: halo, row padding), and ANE↔GPU hand-offs.
  A14: 7.4 ms × 52 per call plus ~110 ms more inside the pipeline than back
  to back (only ~30 ms of it is other ANE programs evicting the generator's:
  measured by moving DecoderPre to the GPU and duration/F0Ntrain to the CPU).
- RAM floor: weights ~164 MB fp16 (on iOS the ANE models do not count in the
  app footprint) plus a streamed working set of tens of MB. iOS: 3 s clips
  peak at 43-46 MB (at the floor); 10 s at 350 MB and 30 s at ~880 MB, set by
  the hybrid's IOSurface pool (~25 tensors, all GPU-touched, so they cannot be
  made purgeable). Macs: 0.9-0.96 GB process peak vs ~0.2-0.3 GB floor.
- Second opinion (both reviewers): this floor ignores shared DRAM traffic
  (~8 GB per 30 s call); the hybrid is DRAM-bound on the A14 (measured ~30 GB/s
  per call) and likely the M1. A two-part floor (hardware: max(FLOPs/peak,
  bytes/bandwidth); public API: plus measured per-handoff cost) puts the M1
  at ~240 ms and the A14 at ~380-400 ms for 30 s, i.e. ~55% today. Their
  estimate of what public APIs can reach with the exact model: 70-80% of that.
  >90% would need fewer bytes than the exact model moves (e.g. a distilled
  generator), which breaks the 44 dB-vs-fp32 contract.
- E3 bandwidth check (M1, 05:58 PT): a GPU blit hog alone moves 51 GB/s.
  Beside it the generator check went 300 → 401 ms (+34%) while the hog kept
  46.5 GB/s. The generator averages ~27 GB/s on its own (8 GB / 0.3 s): on the
  M1 it is partly bandwidth-bound and partly hand-off/latency-bound. (Confound:
  the hog also takes GPU time.)
- Decision (06:00 PT): keep the exact-match contract (CLAUDE.md: "redesign
  the pipeline, not the model"). A distilled or smaller generator, the only
  route reviewers saw past ~70-80% of a corrected floor, is out of scope.
- **Rejected: fusing the three parallel branches into one ANE call**
  (2026-10-22 12:15 PT, `tools/ane-lab`, IOSurface I/O, stage-2 shape
  11 × 128 × 13,920, conv + residual). One call with three convs (k3, k7, k11)
  vs three calls:

  | Phone | k3 | k7 | k11 | Sum | One fused call |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | A17 | 8.10 | 9.27 | 10.08 | 27.45 ms | 25.77 ms |
  | A14 | 5.96 | 6.86 | 9.18 | 22.00 ms | 21.03 ms |

  Program switching is free too (A14, 12:40 PT, one multifunction package,
  30 calls each): k3/k7/k11 alone average 7.34 ms, rotating the three
  7.37 ms, the fused call 7.01 ms per conv. With a GPU pass before each call:
  rotating 9.0 ms, fused 9.0 ms per conv.
  Fusing saves only 0.5-0.85 ms per removed call: the "2.6-7.4 ms per call"
  intercept is I/O, not dispatch. A k3 conv + residual moves ~117 MB in 6 ms
  (A14, ~20 GB/s): the ANE side is bound by its I/O bandwidth. Lockstep would
  also give up the ANE/GPU overlap, so it was not built.
- **Rejected: `specializationStrategy = .fastPrediction`** on the 53 hybrid
  function models (`KOKORO_HYBRID_FASTPRED=1`). A14 `--mode hybrid-check`,
  3 interleaved rounds: 419-429 ms vs 416-427 ms baseline.
- **Pipeline penalty:** the generator runs ~130 ms slower after any break
  from its own work; see "Cold generator after a break" below.
- **Private e5rt dispatch works, and buys ~nothing** (2026-10-22 15:30 PT).
  Core ML's own call sequence into Espresso's private `e5rt_*` C API was
  traced with lldb (breakpoints on `^e5rt_` while an `MLModel` loads and
  predicts). Replaying it directly (`e5rt_e5_compiler_compile` on the
  compiled package's `model.mil`, `retain_program_function`, a precompiled
  compute op, IOSurface buffer objects via
  `e5rt_buffer_object_create_from_iosurface(&buf, surface, 0x103)`,
  `execute_sync`) runs from a plain macOS CLI and inside a sandboxed,
  developer-signed iOS app, with output bit-identical to Core ML. Harness:
  `tools/ane-lab/Sources/E5Direct.swift`, `run.py --e5rt`.

  | Phone | Conv (stage-2 shape) | Core ML (IOSurface) | e5rt |
  | --- | --- | ---: | ---: |
  | A14 | k3 + residual | 5.91 ms | 5.70 ms |
  | A17 | k3 + residual | 4.44 ms | 4.46 ms |
  | A17 | k11 + residual | 7.98 ms | 7.76 ms |

  ~0.2 ms per call at most, ~10 ms per 30 s on the A14: the calls are bound
  by ANE I/O, which the private path does not change. Not wired into the
  generator; it also cannot ship in an App Store app.
- **Chunk size vs RAM on iOS** (A14, `--mode document`, pipelined, models
  kept resident, one row count per run, 15:10 PT):

  | Chunks | Audio | Total | Per audio second | First chunk | Peak footprint |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | 3 × 10 s (r4) | 28.9 s | 705-733 ms | 24.8 ms | 254-266 ms | 364 MB |
  | 4 × 7 s (r3) | 27.0 s | 722-738 ms | 27.0 ms | 194-200 ms | 272 MB |
  | 9 × 3 s (folded) | 25.2 s | 1,022-1,076 ms | 41 ms | 126-132 ms | 46 MB |

  7 s chunks cut peak RAM 25% and first audio 60 ms for ~9% more time per
  audio second.
- **Rejected: splitting a long first sentence at its first clause for
  earlier audio.** A 3 s first chunk runs on the folded generator and the
  rest on the hybrid; switching between them costs more than it saves
  (3s,7s,10s,10s pipelined: first 209-230 ms, total 1,063-1,133 ms vs
  10s × 3: first 253-275, total 699-736 ms on the A14). Code removed.
- Not reached: >90% of floor on any device. The structural limits found are
  Core ML's per-call cost (a public-API floor) and the fp16 tensor pool the
  hybrid needs to keep the ANE and GPU overlapped.

### Physics baseline (measured 2026-10-21)

- FLOPs for the 30 s bench input (27.4 s audio), torch `FlopCounterMode`:
  - total 1.52 TFLOP
  - generator 1348 GFLOP (89%)
  - DecoderPre ~75
  - duration ~76 (BERT 63)
  - F0Ntrain 21
- Generator share of wall time at 30 s: M1 83%, M2 Air 87%, A17 87%, A14 89%.
  - The "Generator" column in
    [paper-cs1-frozen-rerun-2026-10-20.md](paper-cs1-frozen-rerun-2026-10-20.md)
    is understated: it subtracts the other stages from `t_coreml_predict_s`,
    which is already generator-only (`swift/Sources/KokoroBenchmark/main.swift:468`).
- A17 Pro, per primitive, at stage-2 size (128 ch × 20 rows × ~8.2k, 20.9M
  elements, fp16), from `tools/ane-lab` micro models:

| Primitive | ANE | Note |
| --- | ---: | --- |
| conv k11 d1 | 7.6-8.5 ms | 7-9 TFLOPS |
| conv k11 d3 (native) | 13 ms | |
| conv k11 d5 (space-to-batch transposes) | 76 ms | do not use |
| conv k11 d5 (rows in batch, W reshaped to (W/5, 5), (11,1) kernel) | 14.7 ms | use this |
| elementwise add / per-channel affine | 5-8 ms | ~0.35 ns/elem/op whatever the layout |
| Snake, native `sin` | 11 ms | 9.2 ms in folded form `z + sin(z)²` |
| Snake, degree-5 polynomial | 29 ms | |
| AdaIN (two-pass masked means) | 17-20 ms | matmul-stats form 69 ms; do not use |
| resblock pair, stats as inputs, 4-row call | 1.23 ns/elem | vs ~2.7 ns/elem op by op |

- The same convs-only folded graph takes 514 ms on the A17 ANE and 1,677 ms on
  the A17 GPU. On the A17 the GPU is conv-bound. The ANE is bound by its
  elementwise ops, not by its convs.

### Results on the iPhone 15 Pro Max (A17 Pro), 30 s, core 120, halo 8, 20 rows

| Graph | ANE median | GPU median |
| --- | ---: | ---: |
| v1: rows in H axis, space-to-batch dilation, polynomial Snake | 1,705 ms | 1,510 ms |
| v1 ablation: convs only | 514 ms | 1,677 ms |
| v1 ablation: no Snake | 934 ms | |
| v1 ablation: no AdaIN | 1,281 ms | |
| v1 ablation: native `sin` Snake | 1,196 ms | |
| v2: rows in batch, reshape dilation, folded native Snake | 773 ms (764 rerun) | |
| v2 native, core 96 / 160 / **240** (halo 8) | 728 / 742 / **700 ms** | |
| v2 poly (degree 5 Snake, no `sin`), core 120 | 1,223 ms | |

- Wider rows are faster and lighter: core 240 (W2 = 15,365) peaks at 496 MB
  footprint vs 700 MB at core 120, and still compiles on the A17. On device it
  scores 45.2 dB, corr 0.999985.
- The polynomial Snake costs ~450 ms over native `sin` on the A17. Chips without
  ANE `sin` (M1, M2, A14) pay that. Degree 4 is free in quality (67.9 dB fp32)
  and saves 2 passes per Snake. Degree 3 gives 53.0 dB fp32.
- The poly package places 2,027 of 2,028 ops on an M2-generation ANE and scores
  44.2 dB there.
- **Macs: the first Python measurement was misleading; Swift vs Swift on the
  same host, the ANE wins.**
  - `mac_time.py` (Python, fp32 numpy I/O) showed M1 1,716 ms and M2 Air
    1,898 ms on the ANE. Those numbers are not comparable with the Swift
    receipts.
  - `kokoro-bench`, M1 Mini, 30 s, Botnet load ~3, rounds alternated
    fold/base/fold/base, 10 warm iterations each:

  | Arm | Wall | Generator | Duration | Tail/trim |
  | --- | ---: | ---: | ---: | ---: |
  | Shipping GPU (bucketed) | 2,807 / 2,927 ms | 2,522 / 2,693 ms | 140 ms | 5 ms |
  | **Folded ANE (poly deg 4, core 240)** | **1,730 / 1,863 ms** | **1,457 / 1,587 ms** | 142 ms | 11 ms |

  - Botnet load hurts the GPU more than the ANE: the quiet-host shipping
    generator is ~1,354 ms. Redo on a quiet host before claiming a number.
  - M1 ANE numerics against fp32: **44.5 dB, corr 0.999982**.
  - Fold vs the shipping GPU WAV is 37.2 dB, as expected for two independent
    fp16 approximations.
- **M2 Air**, same protocol (load 1.7-4.3, WindowServer and Botnet busy):

  | Arm | Wall | Generator |
  | --- | ---: | ---: |
  | Shipping GPU | 5,821 / 4,689 ms | 5,611 / 4,492 ms |
  | **Folded ANE** | **1,400 / 1,380 ms** | **1,192 / 1,158 ms** |

  The October quiet-host shipping wall was 1,176 ms, so on a quiet M2 the
  folded ANE path is ~15% slower. On the Macs the ANE is far more robust to GPU
  contention; on a quiet host it does not win by itself. Next: run ANE and GPU
  together (branch split).
- **M1 ANE time breakdown** (poly, core 240, `mac_time.py`, ablations):
  - full ~1,430 ms
  - no Snake 812 ms, no AdaIN 1,093 ms, convs only 327 ms
  - so Snake ~43%, AdaIN ~23%, convs ~23%
- **Leaner AdaIN (adopted):** take the residual mean from the same masked `d`
  used for the second moment, `var = E[d²] - m2²`, and fold `m2` into the
  per-channel shift. Two fewer full-tensor passes; M1 on-device 44.2 dB
  (unchanged).
- **Degree-3 Snake polynomial (adopted for chips without ANE `sin`):** 53.0 dB
  fp32, 44.0 dB on the M1 ANE, same as degree 4. M1 timing is too noisy under
  load to resolve the ~10% (best minimum 1,211 vs ~1,370 ms).
- **Host-code bug, fixed.** `floatValues(from:)` walks a strided fp16 ANE
  output element by element: about 1 s per 30 s clip on an M1, for both the
  logits unfold and the x_pre fold. Fixed with `readRun` (vImage
  fp16→fp32 per contiguous run); tail now 11 ms. Regression test:
  `FoldedGeneratorTests.testReadRunConvertsStridedFloat16Rows`.
- **End to end on the A17** (`ios-bench --folded-generator`, packages in
  `coreml-fold/`): the generator stage, including host fold, masks and Swift
  tail, takes **0.74 s at 30 s** (58 ms at 3 s). That run used ladder mode,
  whose first rung is `.all` on every stage (other stages ~1.07 s), so the
  end-to-end A/B under the staged policy is still to do. The phone kept dropping
  its CoreDevice connection (error 4016).
- **Mac options from the second opinions** (2026-10-21,
  `tmp/second-opinions/mac-generator-brief-*`):
  - Branch split: each stage's 3 parallel resblocks (k = 3/7/11) own their
    AdaIN stats, so a whole branch can run on the ANE while the GPU runs the
    others, with one join per stage.
  - Sentence pipelining: sentence n on the ANE while n+1 runs on the GPU.
  - Measure Swift vs Swift on the same host before any custom Metal.
  - The A14 shares the M1's ANE generation, but its GPU is much weaker; expect
    a win there.
- The executor now prefers `kokoro_generator_fold_{N}s.mlpackage` (loaded
  `.cpuAndNeuralEngine`) when it is present; `KOKORO_FOLDED_GENERATOR=0` turns
  it off.

---

## Issue: Cold generator after a break (A14) — Active

**First spotted:** 2026-10-22
**Status:** Active

### Summary

On the iPhone 12 Pro the hybrid generator takes 416-433 ms when runs follow
each other with no gap, but ~470-610 ms after 100 ms or more of anything
else. In the pipeline the generator runs right after the CPU-heavy front
stages, so it always pays this: 564-584 ms in the pipeline vs ~420 ms back to
back, about 20% of the 30 s end-to-end time. Keeping the ANE and the GPU busy
together during the break recovers most of it; keeping only one of them, the
CPU or memory busy does not.

### Symptom

`ios-bench --mode hybrid-check` (5 runs, last 3 kept), A14, 30 s input,
2-3 interleaved rounds per arm, 12:50-14:00 PT. Knobs (`BenchApp.swift`):
`KOKORO_CHECK_CPU_MS`, `KOKORO_CHECK_CPU_THREADS`, `KOKORO_CHECK_ALLOC_MB`,
`KOKORO_CHECK_MEMCPY_MS`, `KOKORO_CHECK_GAP_MS` (the ANE/GPU keep-busy arms
used a since-removed `keepANEBusy` hook).

| Before each run | Generator |
| --- | ---: |
| nothing (back to back) | 415-452 ms |
| 50 ms CPU busy loop | 415-433 ms |
| 100 ms CPU busy loop | 477-515 ms |
| 150 ms CPU busy loop, 1 / 2 / 4 threads | 494-515 / 471-556 / 489-532 ms |
| 150 ms CPU + 40 MB fresh writes | 471-532 ms |
| 150 ms CPU, then 30 / 100 / 400 ms idle | 500-522 / 514-552 / 568-617 ms |
| 150 ms memcpy on 2 threads (DRAM busy) | 488-522 ms |
| 300 ms idle | 542-615 ms |
| 300 ms idle, ANE kept busy (`post` conv loop) | 513-580 ms |
| 150 ms CPU, ANE kept busy | 463-550 ms |
| 300 ms idle, ANE and GPU kept busy | **456-523 ms** |
| spin-wait off (`KOKORO_HYBRID_SPIN=0`), back to back | 458-530 ms |
| spin-wait off, after 150 ms CPU | 545-587 ms |

### Root Cause

TBD. Not manually confirmed. It is time-based, not CPU-, cache- or
DRAM-demand-based: a register-only busy loop triggers it, memcpy traffic
does not prevent it, and longer idle makes it worse. Only keeping both
engines busy prevents most of it, which points at a power-state or clock
ramp shared by the ANE and the GPU (or the fabric between them) that needs
both busy. Public APIs expose no control over it.

### Related Guides

- [Kokoro A14 iPhone generator execution guide][a14-guide] - A14 generator
  placement and limits

### Fix

None yet. **Rejected: pre-warming in the pipeline** (14:20 PT). A
`startWarm()`/`stopWarm()` pair kept the ANE (`post` conv loop) and GPU (a
stage-2-sized copy) busy from the end of F0Ntrain until the generator
started. A14 end to end, 3 interleaved rounds:

| Arm | 10 s | 30 s | 30 s generator |
| --- | ---: | ---: | ---: |
| pre-warm | 238-254 ms | 768-784 ms | 607-617 ms |
| baseline | 232-242 ms | 722-759 ms | 592-611 ms |

The generator did not get faster, and the warm loop slowed the overlapping
DecoderPre/hn-NSF stages. Code removed. The ANE/GPU keep-busy knobs were
removed from `ios-bench` with it; the CPU, memcpy and gap knobs remain.

**Rejected: a realtime-audio work interval** (14:40 PT).
`AudioWorkIntervalCreate` (via `dlsym`; Swift marks it unavailable) with every
generator thread joined and a 300 ms deadline per run. A14 `--mode
hybrid-check`, 2 rounds: back to back 424-449 vs 418-441 ms; after 150 ms CPU
489-535 vs 481-503 ms. Code removed.

The effect is also on the A17, smaller: back to back 258-273 ms, after
150 ms CPU 275-306 ms, after 300 ms idle 289-304 ms (in the pipeline the
generator measures 295-310 ms).

M1 (`kokoro-bench --hybrid-check` now has the same `KOKORO_CHECK_CPU_MS` /
`KOKORO_CHECK_GAP_MS` knobs, 14:50 PT): inconclusive. Botnet load rose from
2.1 to 5.2 during the run and medians swung 488-608 ms in every arm
(minimums 390-482 ms), with no cold pattern visible above the noise.

**Diagnosis: what "cold" is** (A14, e5rt path, `KOKORO_HYBRID_CALLS=1`
prints every ANE call's start and duration, 18:00 PT). After 300 ms idle the
first ANE call started 48-52 ms into the run vs 16-19 ms warm: the host
setup (input crop, stride-6 space-to-depth in bounds-checked Swift loops,
~70 `makeBuffer` calls for AdaIN tables) ran ~3x slower on a cold CPU. After
that, calls were 1.1-1.6x slower throughout, with no early ramp: a
sustained state, not a warm-up.

**Kept: fast host setup** (`HybridInputs.swift`, tests `HybridInputsTests`):
vImage row conversion and pointer loops in parallel, AdaIN table buffers
allocated once. First ANE call now at ~4 ms warm, ~10.5 ms cold. A14 30 s
generator 407-427 → 390-405 ms warm, 517-533 → 482-528 ms cold.

**Kept: a 10-row package** (`ROWS` now 2,3,4,6,10,11). The "30 s" fixture
is 27.4 s = 2,190 frames = 10 rows of 224; it ran on r11 (9% more rows).

**Rejected: e5rt `skip_io_fences` and `prewire_in_use_allocations`**
(suggested by a Fable 5.1 review; config set with
`e5rt_execution_stream_set_config_options(stream, &cfg)`, by value traps).
A14 30 s: warm 404-435 vs 407-427 ms, cold 516-533 vs 517-533. Pre-wire
fails on aliased buffers ("Program IOSurfaces map failure (0x12)").

**A14 end to end after all of the above** (3 rounds, 18:20 PT): 30 s
**582-595 ms** with e5rt (Core ML path 638-658), 10 s 225-228 ms, peak
271 MB at 10 s (Core ML 357 MB). Was 683-726 ms this morning.

**Kept: parallel CPU tails around the generator** (exact; tests
`testTailChunksMatchSingleChunk`, `HarmonicSourceTests`). The iSTFT tail
runs in 4 frame chunks (A14 30 s: 16-19 → 7.6 ms); the hn-NSF STFT runs its
11 bins in parallel and drops two bounds-checked copy loops (45 → 36 ms,
now mostly hidden under DecoderPre's 28 ms). A14 30 s end to end with
e5rt: **578-587 ms** (3 rounds, 19:00 PT).

A17 end to end, Core ML path (its faster path), with all of the above
(3 rounds, 19:40 PT): 30 s **330-343 ms** (generator 254-260, was ~295;
was 371-383 end to end), 10 s 135-143 ms. Peak 1.89-1.94 GB at 30 s.

M1 end to end, 30 s, with all of the above (`kokoro-bench --batch`, 8 runs,
2 rounds, Botnet load ~3, 16:50 PT): e5rt 501-521 ms, peak 682 MB; Core ML
path 501-511 ms, peak 874-886 MB. Too loaded to rank speed (quiet best was
405 ms). The 10 s key failed in both arms: the M1's `models_v10/hybrid` still held
pre-crop r2-r6 packages (width 14,880) while `plan.json` (synced with r10)
says 14,400. Fixed by syncing the whole `coreml-hybrid/` folder. Guard:
`HybridGenerator.init` now compares the `post` output width with
`plan.w2s` and throws `badPlan` with a clear message. Not testable in unit
tests (needs packages); verified with an old r11 package and the current plan.

**Not pursued (estimated, not built):** a hand-written Accelerate LSTM for
the duration back (Core ML's BNNS LSTM ~31 ms on the A14; a sequential
476-step x 4-layer recurrence estimates ~38 ms), and deferring G2P of later
chunks for first audio (G2P ≤1.1 ms per chunk,
`iphone-release-build-mlx-comparison.md`).

**Kept: vectorized hn-NSF sine generation** (23:30 PT; second opinion
`tmp/second-opinions/brief-20261022-182525/`). A14 split at 30 s: sine
generation 22 ms, STFT 3.5 ms (`KOKORO_HNSF_PROFILE=1`), so a GPU STFT (one
reviewer's rank 1) cannot pay. The interpolation now runs in vDSP (ramp,
clip, `vDSP_vlintD`) and the sine in Float (`vvsinpif` on the phase reduced
to [0, 1) cycles in Double): max error 2e-6 on the merged source vs the
Double form (`testSineGenMatchesSequentialReferenceAcrossChunkBoundary`,
was bit-exact). Sine 22 → ~19 ms; the hn-NSF stage is still 32-35 ms (~10 ms
outside sine + STFT, unprofiled). M1 30 s audio: new build vs itself
45.9 dB, vs the previous build 46.0, previous vs itself 47.0. A14 end to
end 30 s **490-498 ms** (3 rounds), 10 s 189-193 ms.

**M2 Air was running stale front-stage packages** (19:20 PT). Its bench
folder (`~/ane-lab-tmp/models_v5`) had the unsplit duration, unsplit
F0Ntrain and GPU-layout DecoderPre; the M1's `models_v10` had the split and
ANE-layout ones. With the same set (`models_v10` built from `models_v5` plus
`coreml-dur-ane/`, `coreml-f0n-split/`, `coreml-dpre-ane/`, `coreml-hybrid/`),
Botnet paused, 30 s, 10 runs: **223.0 ms** (duration 34.1, F0Ntrain 16.5,
DecoderPre 17.8, hn-NSF 14.8 overlapped, generator 151.0) vs 272.5 ms with
the old set (59.5 / 31.0 / 31.8 / 15.9 / 147.1). Benchmarks must use the full
current package set; check the folder before comparing devices.

**Macs with Botnet paused** (2026-10-22 19:06-19:08 PT; paused with
`SupervisorCLI pause` over ssh, restored afterwards to the exact prior policy:
M1 default adaptive + manual override dedicated, M2 Air manual override
adaptive). Botnet's load is GPU work that `top` does not show as CPU; the
load average stayed ~2 for 8 minutes after pausing while running work
drained. `kokoro-bench --batch`, 30 s, 8 runs x 2 rounds:

| Mac | e5rt median | Core ML median | Peak (e5rt / Core ML) |
| --- | ---: | ---: | --- |
| M2 Air | 273-274 ms | 269-270 ms | 667-671 / 853-857 MB |
| M1 | 298-305 ms | 300-306 ms | 659 / 841 MB |

Every Mac latency recorded with Botnet running (including the 405 ms "quiet"
M1 and 473-493 ms M2 Air) was inflated by its GPU work.

**Macs on the final build, Botnet running** (2026-10-22 18:51-18:54 PT local clock on the
Macs, `kokoro-bench --batch`, 30 s, 8 runs x 2 rounds, Botnet busy):

| Mac (load) | e5rt median / min | Core ML median / min | Peak (e5rt / Core ML) |
| --- | --- | --- | --- |
| M2 Air (~2.1) | 398-469 / 376 ms | 411-432 / 395 ms | 667-671 / 864-870 MB |
| M1 (~2.8) | 453-523 / 392 ms | 469-474 / 450 ms | 665-668 / 835-857 MB |

Earlier: M2 Air 473-493 ms (similar load), M1 405 ms quiet. Needs a quiet
window for a fair latency number.

**Kept: apply passes walk rows and channels in reverse** (Opus idea; 23:55 PT).
The stats pass streams the tensor front to back, so its tail is still in the
system cache when the apply pass starts at the end. A14 back to back 329-342
ms (was 333-343), GPU busy ~245 (was ~256); end to end 30 s 488-497 ms, 10 s
187-191. Audio 46.1 dB.

**Kept: parallel Gaussian noise transform** (exact; `generateGaussianNoise`
draws the RNG stream in order, then runs log/sqrt/cos/sin per chunk in
parallel; noise tests unchanged). The missing ~10 ms in hn-NSF was the
hn-NSF thread waiting for the pre-generated noise, which ran single-threaded
past F0Ntrain on the A14. Wait now 0.5-0.8 ms; hn-NSF stage 32-35 → 24-26 ms,
now shorter than DecoderPre (28 ms), so end to end barely moves (A14 30 s
490-497 ms). That also makes "start the noise branches when `har` is ready"
(Opus) possible, but with ~3 ms of head start it is not worth building.

**Kept: last residual of each branch added in the branch sum** (22:50 PT;
join modes 4-6, `sumBranches(claim:)`, tests `testJoinAddsDeferredResidual`,
`testClaimRunsOnceForTheHolder`). The branch that becomes the accumulator
still joins its own residual; the others add theirs in the sum pass. A14:
back to back 333-343 ms (was 342-355), end to end 30 s **502-509 ms**,
10 s 190-192 ms. Audio 46.1 dB.

**Rejected: duration LSTM back on the GPU** (A14, 22:20 PT). End to end it
was faster (30 s 491-494 vs 502-505 ms: the generator ran 373 vs 387 ms,
apparently because real GPU work just before it softens the cold penalty),
but the GPU LSTM's fp16 numerics change per-token durations: same total
frames (1,096) yet the 30 s audio differs completely (-1.6 dB vs two CPU
runs agreeing at 47 dB, M1 `kokoro-bench --wav`). Breaks exact output.

**Rejected: F0Ntrain LSTM front on the GPU** (A14, 3 rounds): 30 s 511-520 vs
513-517 ms on the CPU; no effect.

**Rejected: DecoderPre on the GPU.** The ANE-layout DecoderPre takes 248 ms
on the A14 GPU at 30 s (27 ms on the ANE); 30 s end to end 788-797 ms.

**Kept: residual add on the GPU, fused into the next AdaIN stats pass**
(21:40 PT; second opinion `tmp/second-opinions/brief-20261022-173448/`, both
Opus 5.5 and GPT-6 Astra ranked it first; exporter `--resid-gpu`, plan
`resid_gpu`, Metal `stats_add`). With the tiled phase kernel the A14 became
ANE-bound (back to back: ANE busy 321-326 ms of a 366-394 ms span, GPU 222-261).
The residual input costs the ANE +3.7 ms (k3) / +4.4 ms (k11) per stage-2 call
after a GPU pass (`tools/ane-lab`, conv vs conv+residual, 11 x 128 x 14,400).
Now each pair's second conv is a plain conv; the next pair's stats pass
writes x = half(y + r) and takes its statistics in the same read; the last
pair of a resblock adds with a join.
- Separate join first: span 353-366 ms, ANE busy 268-272, GPU 273-275.
- Fused: span **342-355 ms**, ANE 264-265, GPU 264-266.
- Audio 46.0 dB (Core ML and e5rt paths). Bug on the way: the row shift was
  read from x[0] after other threads could overwrite it (19.4 dB); one
  shift per threadgroup now. `testStatsAddMatchesStatsOfTheSum` checks the
  math; the race itself does not reproduce in the unit harness (needs GPU
  scheduling as in the full generator), so the hybrid-check dB gate guards it.
- End to end, 3 rounds: A14 30 s **507-525 ms** e5rt / 522-526 Core ML path,
  10 s 193-196; A17 (Core ML path) 30 s **313-323 ms**, 10 s 134-143, peak
  1.33-1.37 GB at 30 s (was 1.89-1.94: one fewer input per conv).
- The previous packages (residual on the ANE) are kept outside the repo as a
  rollback; `coreml-hybrid/` now holds the `--resid-gpu` set (r2-r11, r10).

**Kept: vectorized tiled `apply_phase`** (20:40 PT; tests
`HybridPhaseKernelTests`, incl. new `testApplyPhaseAcrossTiles`). A GPU
roofline check (Opus's rank-5 idea) found the phase-layout AdaIN+Snake kernel
(dilated layers) at 3-5x the cost of the dense one. A14, GPU alone
(`KOKORO_HYBRID_CHAINS=1 KOKORO_HYBRID_SERIAL=1 KOKORO_HYBRID_PROFILE=1`,
GPU time per stage-2 call; dense `apply` 3.1 ms ≈ 25 GB/s):

| Version | d13 | d31 | d15 | d51 |
| --- | ---: | ---: | ---: | ---: |
| float[20] block per thread, lane-strided half4 reads (old) | 11.6 | 13.1 | 12.9 | 14.7 ms |
| per-sample scalar gathers | 20.5 | 17.1 | 17.0 | 18.5 ms |
| scalar threadgroup tile | 26.3 | 18.7 | 18.7 | 21.2 ms |
| **half4-only device I/O, shuffle in threadgroup memory** | **6.0** | **7.1** | **5.7** | **6.7 ms** |

A17 (Core ML path, 3 rounds, 21:00 PT): 30 s 319-330 ms (was 330-343), 10 s
128-135 ms; the A17 generator is ANE-bound, so the GPU fix moves it little.
e5rt on the A17: 30 s 324-325 ms at 1.09-1.31 GB peak (Core ML 1.89-1.94 GB),
10 s still ~25 ms slower (154-156 ms).

Lesson: on Apple GPUs device memory must be read and written as contiguous
half4 across threads; scalar half loads cost more than the strided pattern
they replace. Output bit-identical to the old kernel. A14 end to end
(3 rounds): 30 s **519-530 ms** with e5rt (was 578-587), Core ML path
537-553; 10 s 194-197 ms (was 221-228).

**Rejected: GPU ↔ ANE event chaining** (Fable 5.1's rank-6 idea, 20:00 PT).
It works: a `MTLSharedEvent` is an `IOSurfaceSharedEvent` subclass, so
`e5rt_async_event_create_from_iosurface_shared_event(&ev, "name", event)`
(3 args; the event is the third), `e5rt_async_event_set_active_future_value(ev, v)`,
`e5rt_execution_stream_operation_bind_dependent_events(op, &events, n)`,
`..._bind_completion_event(op, ev)` and `e5rt_execution_stream_submit_async(stream, block)`
let a command buffer gate an ANE op and the op gate the next command buffer
with no host involvement; output bit-identical to Core ML. But on the A14
(`tools/ane-lab` `run.py --e5chain 12`, 12 × GPU pass + stage-2 k3 conv,
8 rounds) chained 111.2 ms vs host-synchronous 109.7 ms median: host round
trips cost nothing measurable. Not built into the generator.

**Rejected: e5rt stream QoS values** (`KOKORO_E5_QOS`, A14, 30 s after
300 ms idle, 17:40 PT): 33 (Core ML's 0x21) 490-551 ms, 21 (Orion) 516-551,
47 506-521, 9 541-573. Priority does not explain the cold penalty.

**Finding: the standalone check measures boost, the pipeline measures
sustained speed** (A14, e5rt, 30 s, 2026-10-22 19:40-21:00 PT). The
pipeline generator ran ~380 ms (r10) and 402-432 ms (r11 forced) against
333-347 ms for `--mode hybrid-check` at r11, and an immediate second run
inside the pipeline was no faster. Ruled out in turn: thread QoS (both
`Task.detached(.userInitiated)`), input data (the check's own inputs run
inside the pipeline process: 379-404 ms on a cool phone, check 329-343 in
the same session), resident models (a pipeline run first, models kept or
freed: +5-15 ms, within noise), memory layout (generator allocated before
the pipeline run: no change), leftover CPU work (process CPU time per run
equal). Then `KOKORO_CHECK_RUNS=25` (back to back, no gaps): 427 (first),
334, 343, 347 … 372, 386, 381, 385, 383, 375 ms. The check's 5-run window
is the chip's boost; ~10 s of sustained work settles near 370-385 ms (r11),
which is the regime the ladder benchmark (warm-ups plus iterations back to
back) measures. So the A14 generator is power- or heat-limited when run
continuously, and energy per run (DRAM traffic) is the lever there, not
host overhead. Compare device numbers only within one regime.

**Second opinion, 21:00 PT** (`tmp/second-opinions/brief-20261022-204757-*`,
Opus 5.5 and GPT-6 Astra): both ranked exact GPU-side changes first (tiled
`unshuffle`, cheaper `apply_phase`, shared first-AdaIN statistics) and a
sustained-regime recheck of the spin-wait. All measured on the A14 the same
night, old and new apps installed alternately, 3 rounds, 30 s:

- **Spin-wait under sustained load: keep.** `KOKORO_CHECK_RUNS=25`, 2 rounds:
  spin median 357-358 ms, no spin 367-368.
- **Rejected: tiled `unshuffle`** (phase-by-phase contiguous loads into
  threadgroup memory, half4 writes; bit-identical, test written). End to end
  459-469 vs 459-470 ms; `metal.unshuffle` 16.3-16.4 ms both ways. That
  timer includes waiting behind other chains' command buffers on the shared
  queue, so it was never a memory-pattern problem. Reverted.
- **Rejected: half-precision `apply_phase` tile** (exact: values round to
  half on output anyway). Per-kernel GPU times equal within noise in
  `KOKORO_HYBRID_CHAINS=1 KOKORO_HYBRID_SERIAL=1 KOKORO_HYBRID_PROFILE=1`.
  Reverted.
- **Rejected: one statistics pass shared by a stage's three branches**
  (`statsOnly` before the branches, each branch finalizes with its own g, b;
  audio 46.1 dB both paths, same max logit error). End to end 459-466 vs
  453-469 ms. Removing ~100 MB of GPU reads changes nothing: neither GPU time
  nor DRAM bytes bind. Reverted.

**Rejected: two concurrent branch chains** (`KOKORO_HYBRID_CHAINS=2`, A14,
2 rounds, 22:10 PT). 30 s 537-554 vs 486-493 ms, 10 s 215-216 vs 189-192;
peak 788 vs 736-745 MB, because the layout adds per-stage accumulators and
keeps all ten stage-2 slots. RAM picture at 30 s (r10, e5rt, A14): the arena
is ~385 MB (ten 37 MB stage-2 slots plus inputs and logits) of a 642 MB peak;
~83 MB is base process and models, up to ~170 MB front-stage temporaries.
Wider rows cannot shrink the halo share: the ANE's 16,384 axis cap allows a
core of at most 248 frames (overhead 6.5% vs 7.1% now).

**Where the ANE waits** (`KOKORO_HYBRID_CALLS=1`, r11 check, run 334 ms):
setup to first call 4.7 ms; stage-1 noise chains 15-100 ms (nr0, nr1),
stage-1 branches 98-173, `ups1` 175-180, stage-2 branches 191-331, `post`
339-343. ANE gaps at the boundaries total ~28 ms (stage-2 entry 11.5 ms:
unshuffle, noise join, first AdaIN of three branches; final branch sum
8 ms). Each boundary waits on a whole-clip AdaIN statistic, so it cannot be
pipelined without changing the math. Inside the branch phases the ANE is
~80% busy, with calls I/O-bound. The generator is close to what this layout
allows; the remaining exact lever is the layout's own bytes (the 8-frame
halo per 224-frame row is 7% extra width).

**Rejected: AdaIN statistics computed on the Neural Engine** (21:30 PT).
Idea: the conv that feeds each pair's second AdaIN also emits shifted,
masked means of y and y² (`(y - y[..., :1]) * m`, `reduce_mean`), so half
the GPU stats passes disappear. Precision is workable (macOS ANE, random
data: variance error 0.06% median, 0.7% worst), but the ANE does not fuse
the reductions into the conv. A14, `tools/ane-lab`, 11 x 128 x 14,880 k3
conv, IOSurface I/O: conv alone 2.9 ms, conv + stats 12.5 ms (7 ANE ops),
against ~3-4 ms for a GPU stats pass of the same tensor. The ANE is the
bottleneck already, so this would be slower.

**Rejected: GPU warm-up while the front stages run** (`KOKORO_GPU_WARM=alu`
register-only kernel or `copy` 16 MB blits, started at duration, F0Ntrain or
DecoderPre, stopped before the generator; A14 30 s, 3 rounds). Generator
353-364 vs 366-373 ms, end to end 462-473 vs 471-481 ms (`copy` slowed
hn-NSF 25 → 34 ms). About 1.5% bought with wasted GPU power, which works
against the sustained limit above and against battery. Code removed.

**Rejected: spinning on an e5rt completion event instead of
`execute_sync`** (`submit_async`, a `MTLSharedEvent` completion event, host
spin on `signaledValue`; audio 46.1 dB). A14 30 s end to end 484-490 vs
479-483 ms. The ANE wake-up path is not where the time goes. Code removed.

**Rejected: duration back as two concurrent CPU models** (M2 Air, Botnet
paused, 30 s, 2 × 10 runs per arm, 19:20 PT). The text-encoder LSTM
(`t_en`) does not feed the durations, so the back split cleanly into
`back_pred` (predictor text encoder, duration LSTM, projection) and
`back_ten` (one LSTM). Outputs were bit-identical on CPU (t64-t512, 5 random
inputs each). The back fell from 21.4 to 18.5 ms (the predictor half is most
of it), the duration stage from 34 to 31.5 ms, but wall time stayed
219-222 ms in both arms (generator noise ±2 ms). Eight extra packages for a
gain the clip cannot show; code reverted. Padding is not the lever either:
the 30 s input fills 476 of its 512 tokens.

**Kept: two chunks in flight for multi-chunk text** (14:00 PT).
`executeKokoroSynthesisPipelined` (`PipelinedSynthesis.swift`) runs chunk
i+1's duration stage while chunk i runs its bucket stages (F0Ntrain,
DecoderPre, hn-NSF, generator). The executor's `beforeBucketStages` hook is
the hand-off point, placed before `prepareForBucket` so a chunk never loads
or evicts bucket models while the previous chunk uses them. Model vending
goes through `LockedModelProvider` (the providers' caches are plain
dictionaries). The SDK's `synthesize` and `synthesizeStreaming` use it.
`ios-bench --mode document --keys 10s,10s,10s`, 5 interleaved rounds:

| Phone | One chunk at a time | Pipelined | First chunk (seq / pipe) |
| --- | ---: | ---: | ---: |
| A14 | 748-766 ms | 709-735 ms | 244-260 / 252-267 ms |
| A17 | 413-478 ms | 390-405 ms | 140-159 / 148-156 ms |

Max sample difference 0.004 between the two (the branch sum adds in
completion order, so fp16 rounding differs run to run).

- **Bug (fixed):** the first version created its semaphores with value 1;
  libdispatch traps (`_dispatch_semaphore_dispose`) when a semaphore is freed
  below its initial value. Regression test:
  `PipelinedChunksTests.testPipelinedChunksReturnInOrderAndFreeCleanly`.
- **Trap for benchmarks:** a document whose chunks need different hybrid row
  counts (10 s r4, 7 s r3) reloads all 53 function models per switch: 10s,
  7s, 10s took 2.2 s either way on the A14. The SDK ships one row count on
  iOS, so it never switches.

### Verification

`PipelinedChunksTests` (order, serial second stages with overlap, failure
stops later chunks, a chunk that skips the gate). The single-clip 30 s
number is unchanged by this; the penalty there has no public-API fix found.

---

## Issue: Private e5rt arena for the hybrid generator — Active

**First spotted:** 2026-10-22
**Status:** Active (opt-in `KOKORO_HYBRID_E5=1`; not for App Store builds)

**Build flag (2026-10-23):** the e5rt code is compiled out by default (`#if KOKORO_E5`), so SDK and app binaries carry no private-framework strings. To reproduce the plan 014 `hybrid-e5` column, build with
`swift build -c release --package-path swift -Xswiftc -DKOKORO_E5` and run with `KOKORO_HYBRID_GENERATOR=1 KOKORO_HYBRID_E5=1`. iPhone bench (no project change; the setting reaches the KokoroPipeline package target):
`xcodebuild -project ios-bench/KokoroIPhoneBench.xcodeproj -scheme KokoroIPhoneBench -configuration Release -destination 'generic/platform=iOS' -allowProvisioningUpdates 'OTHER_SWIFT_FLAGS=$(inherited) -DKOKORO_E5' build`, then launch with `-e '{"KOKORO_HYBRID_E5":"1"}'` plus `--hybrid-generator`. Without the flag, `KOKORO_HYBRID_E5=1` fails with "needs a build with -Xswiftc -DKOKORO_E5": a benchmark run (`KOKORO_HYBRID_GENERATOR=1` or `--hybrid-generator`) stops, and an app falls back to folded with a stderr line. The plan 014 apps predate the gate (e5rt always compiled in); same code path.

### Summary

Core ML needs every tensor to be its own IOSurface of exactly the tensor's
shape, so the hybrid's ~25 pooled tensors can never share memory. Espresso's
private `e5rt_*` C API (the layer Core ML itself calls) binds any byte range
of a buffer to an ANE port. `HybridGenerator` can now run its 53 convs
through e5rt with all tensors in one lifetime-planned arena: stage-1 tensors
and `poly1` live inside the two stage-2 branch chains, which are idle until
stage 2. Output matches Core ML (70 dB between paths, 46.1 dB audio vs the
PyTorch reference on the A14). Peak RAM falls 19-60%; speed is better on the
A14 and 5-10% worse on the A17 (open).

### Symptom

Hybrid peak footprint on phones: A14 10 s chunk ~360 MB, 30 s ~880-920 MB;
A17 generator alone at 30 s 1.77 GB with Core ML.

### Root Cause

Core ML I/O rules (one exact-shape IOSurface per tensor, no offsets), plus,
on the A17, a large per-model charge for the 53 function `MLModel`s.

### Related Guides

- `README/Papers` copies in other repos: ANEForge companion guide
  (`whisper-fine-tuner-macos/README/papers/apple-neural-engine-architecture-programming-performance.md`,
  listing 12) and Orion (`Qween/README/papers/orion-characterizing-programming-apple-neural-engine-llm.md`).
  Their argument orders disagree; the sequence below is traced from Core ML.

### Fix

- **Call sequence (traced with lldb, breakpoints on `^e5rt_` while an
  `MLModel` loads and predicts; every create returns through its first
  argument, `release(&handle)`):** `e5_compiler_create`,
  `e5_compiler_options_create`, `..._set_compute_device_types_mask(opts, 4)`,
  `..._set_mil_entry_points(opts, names, n)`,
  `e5_compiler_compile(compiler, "<mlmodelc>/model.mil", opts, &lib)`
  (all 53 functions in one compile), then per function
  `program_library_retain_program_function`,
  `precompiled_compute_op_create_options_create_with_program_function`,
  `execution_stream_operation_create_precompiled_compute_operation_with_options`,
  `execution_stream_create`. Per call (Core ML's order): `stream_reset`, bind
  ports, `prepare_op_for_encode`, `set_quality_of_service(stream, 0x21)`,
  `encode_operation`, `execute_sync`.
- **Buffers:** `buffer_object_create_from_iosurface(&b, surface, 0x103)`,
  then `buffer_object_create_as_alias(&a, b, offset, length)` per tensor. All
  53 functions' ports are dense (no row padding, even the 240-wide `ups0`).
- **Arena:** `HybridArenaLayout.swift` (offsets), `E5Runtime.swift` (library,
  operations, arena). One IOSurface per stage-2-sized slot, wrapped as one
  untracked `MTLBuffer` each; `SharedTensor` carries a byte offset for Metal.
- **Gotchas found on the way:**
  - One ~445 MB arena surface made the ANE ~50% slower after any idle gap
    (A14 30 s after 300 ms idle: 729-963 ms vs Core ML 553-641). Splitting
    into slot surfaces fixed it (506-535 ms, better than Core ML).
  - Without `release` calls, an r4 → r11 switch kept the old programs and
    arena alive (slower, higher peak).
  - Setting the QoS once at stream creation is not what Core ML does; it sets
    it every call (did not by itself fix the cold penalty).

### Verification

A14 (iPhone 12 Pro), 2026-10-22 ~16:00 PT:

| Run | e5rt arena | Core ML |
| --- | --- | --- |
| Generator only, 30 s, back to back | 400-413 ms, 576-585 MB | 416-429 ms, 723-744 MB |
| Generator only, 30 s, after 300 ms idle | 506-535 ms | 548-590 ms |
| End to end 10 s (3 rounds) | 233-237 ms, peak 292 MB | 232-239 ms, peak 359-360 MB |
| End to end 30 s (3 rounds) | **663-674 ms** | 728-736 ms |

A17 (iPhone 15 Pro):

| Run | e5rt arena | Core ML |
| --- | --- | --- |
| Generator only, 30 s, back to back | 288-311 ms, 697-713 MB | 264-295 ms, 1,765-1,768 MB |
| End to end 10 s | 161-171 ms, peak 487-490 MB | 140-157 ms, peak 870-889 MB |
| End to end 30 s | 381-387 ms, peak 1.02-1.19 GB | 378-421 ms, peak 2.08-2.11 GB |

- A17 10 s end to end, 4 more interleaved rounds: generator 126-131 vs
  106-112 ms (e5rt ~20 ms slower), peak 490 vs 886-889 MB. Core ML's device
  mask 5 and its `EnableLowEffortCPAllocation=true` ANE option
  (`KOKORO_E5_MASK`, `KOKORO_E5_ANE_OPTIONS`) changed nothing (generator only
  262-306 ms across arms vs Core ML 262-292).
- M1 (`kokoro-bench --hybrid-check`, 30 s, Botnet load 2.7-3.1, 15:21 PT):
  e5rt 465-483 ms median, peak 562 MB; Core ML 443-457 ms, peak 769-770 MB.
- With chunking (A14, `--mode document`, pipelined, 16:30 PT): arena 3 × 10 s
  657-709 ms, first 242-262 ms, peak 294 MB; arena 4 × 7 s 715-724 ms, first
  192-201 ms, peak 223 MB (Core ML 3 × 10 s: 705-733 ms, 254-266 ms, 364 MB).
- **Where the A17 slowdown comes from** (generator only, 30 s, 3 interleaved
  rounds, 16:50 PT): e5rt with every tensor in its own slot surface
  (`KOKORO_E5_NOSHARE=1`) 253-264 ms, peak 841 MB; shared arena 267-335 ms,
  646-661 MB; Core ML 250-261 ms, 1,764-1,769 MB. Dropping Core ML's 53
  `MLModel`s saves ~920 MB on the A17 at no cost; sharing saves ~190 MB more
  for 10-15 ms. Best per device: A14 shared arena, A17 separate surfaces.
- **Per-device mode (default since 17:00 PT):** `HybridArenaLayout.sharesMemory`
  picks separate surfaces on iPhone16,x and newer, the shared arena elsewhere
  (`KOKORO_E5_SHARE=0/1` overrides; test `testSharingChoicePerDevice`). The
  accumulator slots are allocated only with fewer than 3 chains. 30 s
  generator, 2 interleaved rounds:

  | Device | Mode | e5rt | Core ML |
  | --- | --- | --- | --- |
  | A14 | shared | 401-408 ms, 534-538 MB | 417-432 ms, 745-748 MB |
  | A17 | separate | 257-269 ms, 840 MB | 253-261 ms, 1,764-1,766 MB |
  | M1 (load ~3) | shared | 453-483 ms, 568 MB | 454-480 ms, 778-779 MB |
  | M1 | separate | 430-447 ms, 748 MB | |
  | M2 Air (load ~2.6) | shared | 362-369 ms median, 568 MB | 319-389 ms median, 779 MB |

- Trade: A14 faster and lighter; A17 and M1 27-60% less RAM for 5-20 ms more
  per call set. Kept opt-in.
- Tests: `HybridArenaLayoutTests` (live-together tensors never overlap at
  any row count, no tensor crosses a slot, arena < 80% of separate tensors).
  `sumBranches` moved to `HybridBranchSum.swift` (HybridGenerator back to
  980 lines).
- Pitfall: remote runs on the Macs go through zsh, which does not word-split
  `$a` in `set -- $a`; the first M1 "e5" arm silently ran Core ML. Use
  `bash -s`.

---

## Issue: Peak RAM — Active (folded path fixed on M1)

**First spotted:** 2026-10-21
**Status:** Active. The M1 is fixed; iPhones not yet measured.

### Summary

On the M1 Mini, `kokoro-bench --batch` running five 30 s calls on one input
(`/usr/bin/time -l` peak footprint, `KOKORO_FOOTPRINT_LOG=1` per stage):

| Pipeline | Steady footprint | Process peak |
| --- | ---: | ---: |
| Shipping (fixed-bucket GPU generator) | ~1.27 GB | 1.79 GB |
| **Folded ANE generator** | **~380 MB** | **408 MB** |

Per-model attribution (`tools/ane-lab/fpcheck.swift`, one model per process,
footprint after compile, load and 3 predicts):

| Model | Units | Total |
| --- | --- | ---: |
| `kokoro_decoder_har_post_30s` | GPU | **1,032 MB (+999 MB on first predict)** |
| `kokoro_generator_fold_30s` | ANE | 145 MB |
| `kokoro_duration_t512` | all / GPU / ANE | 51 / 89 / 36 MB |
| `kokoro_f0ntrain_t1200` | all | 46 MB |
| `kokoro_decoder_pre_30s` | ANE | 31 MB |

ANE weights live in wired memory, which `phys_footprint` does not count
(Qween notes). Sample `vm_stat` wired pages before claiming a system-level
number.

### Root Cause

1. The MPSGraph GPU generator allocates ~1 GB of intermediates at 30 s.
2. The executor's warm-up loaded and ran the GPU generator even when the
   folded package was in use.
3. Core ML outputs were autoreleased and never drained in a loop: footprint
   grew ~120 MB (folded) or ~50 MB (shipping) per 30 s call.

### Fix

- `warmModels` warms only the folded generator when one exists.
- `executeKokoroSynthesis` wraps each call in `autoreleasepool`.
- Not testable: both need real Core ML packages, and the test suite has no
  model-provider mock. Verified by the footprint log above (flat across five
  calls) and the M1 peak dropping from 2.43 GB to 408 MB.

---

## Issue: Hybrid generator (ANE convs + Metal AdaIN/Snake) — Active (revived)

**First spotted:** 2026-10-21
**Status:** Active. IOSurface-backed I/O removed the Core ML per-call
overhead. The 30 s generator now runs in **443 ms on an M1** (Botnet load ~3),
vs 1.3-1.5 s folded all-ANE and ~2.7 s shipping GPU, at **46.0 dB** vs fp32.

### The fix that revived it (2026-10-21 23:00 PT)

`scratchpad overhead.swift` on the M1 times an ANE conv (10,128,1,15360) fp16
over 10 calls:

| I/O | nothing between | sleep 5 ms | CPU pass | Metal |
| --- | ---: | ---: | ---: | ---: |
| pointer-backed (`MLMultiArray(dataPointer:)`) | 10.2 ms | 12.1 ms | 11.3 ms | 9.9 ms |
| IOSurface-backed (`MLMultiArray(pixelBuffer:)`, OneComponent16Half, 64-byte rows) | **5.7 ms** | 6.0 ms | 5.9 ms | 5.9 ms |

- The "wake-up" penalty blamed below was Core ML repacking pointer-backed
  arrays on every call, not ANE power gating.
- `SharedTensor` is now a CVPixelBuffer/IOSurface. Metal reads the same
  memory through `makeBuffer(bytesNoCopy:)`.
- The rows need no padding: W·2 bytes must be a multiple of 64, so the
  exporter's stage-2 width is now 60·W0 + 160 = 15,520.
- Do not hold the pixel-buffer lock across a prediction: Core ML rejects a
  locked output backing.
- Profile (serial chains, ms per 30 s): ANE stage-2 convs 232, stage-1 176;
  Metal AdaIN+Snake stage 2 140, stage 1 87; joins 20.
- Parallel chains bring the total to 443 ms; serial is 753 ms.
- Peak footprint went from 4.3 GB to 979 MB. Cutting buffers next: 3 per
  chain, no accumulators.

### Phase-major dilated convs, vectorized Metal (2026-10-21 23:50 - 10-22 00:05 PT)

- **Layout:** the 12 dilated `c1` convs (d = 3, 5) read and write rows
  phase-major: natural sample w = q·d + p is stored at p·(W/d) + q. Inside
  the ANE function a contiguous reshape to (rows, C, d, W/d) turns the
  dilated conv into a plain conv, with no transposes. Metal converts layout
  inside the passes it already runs (`apply_phase`, phase-aware `stats`).
  - Exporter: `PhaseConv`, `plan.json` `phase` map. Geometry now core 224,
    W0 240, W2 14,880, rows (2, 3, 4, 6, 11), every W/d a multiple of 4.
  - x_pre and the ups0 output are not 64-byte rows; `SharedTensor` falls
    back to plain shared memory for them.
  - Tests: `HybridPhaseKernelTests` (both kernels vs a CPU reference),
    `testUnalignedWidthFallsBackToPointerMemory`.
- **Dilated convs on the M1 ANE:** 20-28 ms → 12-15 ms each.
- **Coalesced phase kernel (02:47 PT).** The first `apply_phase` read (or
  wrote) with stride d, so each memory transaction used 1/d of its bytes: A14
  stage-2 apply ran ~9 GB/s (9.4 ms per call). Now each thread owns q in
  [4t, 4t+4) for every phase: D contiguous half4s on the natural side, one
  half4 per phase row. M1 `--hybrid-check`: 299-307 ms median (was 360-430),
  stage-2 GPU apply 57 ms (was 70-128). 46.0 dB.
- **Metal kernels vectorized to half4** (stats and phase apply). GPU time per
  30 s (serial, M1): stage 2 157 → 128 ms, stage 1 ~60 → 40 ms. On a quiet
  moment stage 2 reached ~2.1 ms per apply and ~1 ms per stats pass, about
  the memory-bandwidth floor.
- **30 s generator on the M1** (`--hybrid-check`, 8 iterations, interleaved,
  load ~3): previous packages min 402-434 ms; phase-major + half4 min
  349-379 ms, median 360-405 ms, with 11 rows instead of 10. Quality 46.0 dB,
  corr 0.999988.
- **Where the M1 sits now:** ANE convs take ~300-330 ms for 1.78 TFLOP
  (11 folded rows), near the ANE's measured conv rate. The Metal passes run
  alongside. The generator is ANE-bound.

### Mixed-length text, reloads and RAM (2026-10-22 00:10-00:45 PT)

Second opinion (GPT-6 Astra, Opus 5.5;
`tmp/second-opinions/brief-20261022-000859/`). Both put exact rows, phase-major
k3 dilated convs and pool lifetime first; both flagged the bucket-switch
reload as unmeasured. Their 30-60 ms host-prep estimate was wrong: prep
measures 2.4 ms (`hybrid synth:` line, `KOKORO_HYBRID_PROFILE=1`).

- **Bucket switches reload the hybrid.** One generator stays resident, so a
  new row count reloads 53 function models: ~1.4 s on an M1 with a serial
  loop, ~0.35-0.5 s with concurrent loading (`DispatchQueue.concurrentPerform`
  in `HybridGenerator.init`). A mixed 12-request document (30/15/10/7 s) took
  11.1-12.6 s on the full row ladder, with every return to 30 s costing
  ~1.2-1.4 s instead of ~0.6 s.
  - Fewer packages means fewer switches. `HybridGenerator.rows` now picks the
    smallest packaged row count that fits (test
    `testHybridUsesSmallestPackagedRowCountThatFits`).
- **Bench bug (fixed):** `kokoro-bench` recompiled the folded generator after
  every bucket eviction, ~10 s per switch. It now keeps the compiled URL.
  The SDK already caches compiled models on disk.
- **Rejected: volatile pool tensors.** Marking idle IOSurfaces purgeable
  (`IOSurfaceSetPurgeable`) takes untouched surfaces out of `phys_footprint`
  (5 × 42 MB: 203 → 3 MB), but not once the GPU has used them: after one blit
  they stay counted whether wrapped as a no-copy `MTLBuffer` (also with the
  buffer's own purgeable state set) or as an IOSurface texture (208 MB
  before and after). In the pipeline the footprint stayed at the whole pool
  (M1 30 s: 525 MB after stage 1, 844 after stage 2, 894 on the next call).
  Code removed.
- **Rejected: phase-major k3 dilated convs** (rb0/rb3 `c1_1`, `c1_2`, dense
  width-7/11 kernels today, ~6% zero taps). 1.78 → 1.67 TFLOP and 46.1 dB, but
  M1 `--hybrid-check` (interleaved, 3 × 8): medians 378-411 ms vs 364-417 ms,
  mins worse. Kept dense.
- **Rejected: stage-1 tensors as views into idle stage-2 memory.** Stage 1's
  tensors (~150 MB at 30 s) never overlap the two spare stage-2 chains in
  time, so they were carved out of that memory as pointer-backed views. Core
  ML then allocates and keeps its own staging buffers per model for
  non-IOSurface I/O: M1 generator-only peak 743 MB → 1.40 GB, and slower.
  Reverted. (The same applies, in small, to x_pre and the ups0 output, which
  are pointer-backed because their 240-wide rows are not 64-byte multiples.)
- **A14 peak RAM** (`ios-bench --policy runtime`, one launch per clip, new
  `footprint_mb` / `peak_footprint_mb` fields, ~02:30-02:40 PT):

  | Clip | New latency / peak | Shipping latency / peak |
  | --- | --- | --- |
  | 3 s | 103 ms / **46 MB** | 419 ms / 422 MB |
  | 10 s | 260 ms / **349 MB** | 1,467 ms / 396 MB |
  | 15 s | 370 ms / 504 MB | 2,085 ms / 440 MB |
  | 30 s | 766 ms / 888 MB | 4,259 ms / 635 MB |

  - On iOS the folded model's memory does not count in the app footprint;
    the hybrid's IOSurface pool does. From 15 s up the hybrid is heavier
    than shipping.
  - Fewer chains (A14): 2 chains 15 s 566 ms / 443 MB, 30 s 1,089 ms /
    771 MB; 1 chain 659 ms / 351 MB, 1,280 ms / 608 MB. A bad trade.
  - Throughput per audio second is about the same at 10 s and 15 s chunks
    (26 vs 25 ms per second of audio), so 10 s SDK chunks on phones would keep
    peak RAM under shipping with earlier first audio. Product call (chunk
    length changes prosody context).
- **Footprint breakdown** (M1, 30 s, `KOKORO_FOOTPRINT_LOG=1`): 103 MB after
  the hybrid's 53 models and the other stages load, 199 after DecoderPre,
  525 after stage 1, 844 after stage 2, ~900 peak. The models are cheap; the
  tensor pool is the RAM.
- **Row ladder for products** (M1, ab9, 00:37-00:47 PT, load ~4-6):

  | Packages | 7 s | 10 s | 15 s | 30 s | Peak | Mixed 12-request doc |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | r2, r3, r4, r6, r11 | 186-259 ms | 253-317 | 303-372 | 603-675 | 955 MB | 13.0-13.2 s |
  | r6, r11 | 251-325 | 301-375 | 323-381 | 600-608 | 900 MB | 12.7-12.8 s |

  - Every row-count switch still costs ~0.6-0.8 s, so the ladder should match
    the chunker. The SDK chunks at 15 s by default: ship r6 only and every
    7-15 s chunk shares one resident generator (no reloads, half the pool).
    Botnet's 30 s workers: r11.
- **No separate branch accumulators** with 3 chains: the first finished
  branch's output holds the sum (one tensor and one copy pass less per stage).
  - **Bug found and fixed (01:10 PT):** the first version returned that
    branch's chain to the free list. `parallel` (`concurrentPerform`) may run
    the branches one after another, and the next branch then reused the chain
    and overwrote the sum: stage 2 got garbage input (-4.6 dB, every stage-2
    branch wrong while stage 1 and the noise branch matched PyTorch). A 46.1 dB
    M1 run before that was thread-timing luck. The chain holding the sum now
    stays out of the free list (`HybridGenerator.sumBranches`, test
    `HybridBranchSumTests`, which fails with the old behaviour).
- **Rejected: residual add on Metal** (`c2` convs without the `+ r` input). On the M1 ANE an elementwise
  add costs about a conv: per-function ANE time showed every `c2` at
  1.8-4 TFLOPS vs 5.5 average, and serial ANE time fell 322 → 247 ms with the
  add moved. The add runs as the first dispatch of the next pair's AdaIN
  command buffer (a separate join only after the last pair).
  - M1 generator, interleaved 4 × 8: mean of medians 418 ms vs 399 ms with
    the residual on the ANE; within noise. The chains alternate ANE and GPU, so
    a lighter ANE does not shorten the wall unless the GPU side keeps up.
  - Phones (`ios-bench --policy runtime`, 01:25-01:45 PT, one run each):

    | Device | Clip | Residual on ANE | Residual on Metal |
    | --- | --- | ---: | ---: |
    | A14 | 10 s | 286 ms | 314 ms |
    | A14 | 30 s | 928 ms | 892 ms |
    | A17 | 10 s | 177 ms | 189 ms |
    | A17 | 30 s | 460 ms | 458 ms |

  - No consistent win, so the simpler two-input `c2` stays. The bug fix and
    `sumBranches` stay.
- M1 check: 46.0 dB, generator-only peak 743 MB.

### Integration gotchas (fixed)

- **Reading an IOSurface-backed `MLMultiArray`'s `dataPointer` locks the pixel
  buffer.** Core ML then rejects it as an output backing on the next call
  ("pixel buffer ... has been locked"). Host readers use
  `SharedTensor.hostView`, a pointer view of the same memory.
  - Test: `HybridSharedTensorTests`.
- **Several hybrid generators resident at once** (1-row and 3-row, 106
  function models) made the next load fail with a misleading
  "`functionName` must be nil unless ML Program". Alone, the 4-row package
  loads and runs (M1 10 s: 226 ms wall).
  - Providers keep one resident generator.
  - Not testable without models; see the M1 run log.
- **Compile once.** `HybridGenerator` caches compiled package URLs per
  process (a compile took ~7 s).
- **Provider wiring:**
  - `KokoroModelProvider.hybridGenerator(bucketSec:)`, opt-in with
    `KOKORO_HYBRID_GENERATOR=1` and `<models>/hybrid/`.
  - `ios-bench --hybrid-generator` reads `plan.json` and `adain.bin` from the
    bundle root.

### Summary

- **Idea:** let each engine do what it is good at. The ANE runs every conv
  (`scripts/export_hybrid_generator.py`: 53 functions in one multifunction
  package, fixed shapes per row count). Metal runs exact fp32 AdaIN
  statistics, the affine and native-`sin` Snake, plus the branch joins
  (`swift/Sources/KokoroPipeline/HybridGenerator.swift`; check with
  `kokoro-bench --hybrid-check DIR`).
- **Exactness:** the hybrid step order in fp32 PyTorch is exact (68.0 dB). On
  the M1 the output scores **46.1 dB, corr 0.999988**, the best of any ANE
  variant.
- **Micro test (M1):** ANE k11 conv 11.3 ms + Metal AdaIN+Snake 5.4 ms ≈
  17 ms per stage-2 unit, vs ~45 ms all-ANE. The physics case holds.
- **Whole generator (M1, loaded):**

  | Version | Median | Min |
  | --- | ---: | ---: |
  | Strict alternation | 2.6-2.9 s | |
  | Parallel chains (noise branch during stage 1, the three branches together) | 1.8-2.1 s | 1.25-1.31 s |
  | Folded ANE-only, for comparison | | ~1.4 s |

### Root Cause (superseded 23:00 PT: per-call cost was pointer-backed I/O, see above)

- Each Core ML ANE call costs 10-20 ms beyond the conv when Metal work runs
  between calls: ~16 ms back to back vs ~27 ms after a Metal step (stage-2
  convs, `KOKORO_HYBRID_REPEAT=1`). Wake-up and copies of the
  pointer-backed I/O arrays both look likely.
- Each Metal command buffer costs ~5 ms round trip (`waitUntilCompleted`).
- Peak footprint was 4.3 GB: each per-function `MLModel` maps the whole
  multifunction program. An `autoreleasepool` around predictions did not
  change it.

### Gotchas found on the way (all fixed in the exporter)

- `EnumeratedShapes` for the row axis drops ANE placement: `ups0` took
  522 ms. Export one fixed-shape package per row count instead.
- A native k=3 dilation-3 conv ran at 516 ms on the M1 (off the ANE). The fix
  is the equivalent dense width-7 kernel; `ane_conv` now makes any dilated
  kernel with (k-1)·d+1 ≤ 11 dense.

### Next step if revisited

Needs zero-copy IOSurface-backed I/O, fewer and larger ANE calls, and one
`MLModel` instead of 53. Otherwise only a lower-level ANE runtime gets the 17 ms
units.

---

## Issue: fp16 precision of AdaIN on the ANE — Resolved

**First spotted:** 2026-10-21
**Status:** Resolved (30 s audio 38.7 → 44.4 dB host ANE, 45.2 dB A17)

### Summary

The first ANE build scored 38.7 dB against 47.1 dB for the same graph on the
GPU in fp16. Per-block local SNR pinned the loss on AdaIN: removing AdaIN made
the ANE match the GPU (64.7 vs 66.7 dB).

### Root Cause

- Some channels have |mean| ~3 but std ~0.08.
- The ANE's fp16 masked mean is about 1e-3 relative off (GPU: 2e-4), which
  shifts the whole normalised channel by ~0.04 σ.
- This is systematic per channel, so it is not averaged out.

### Fix

Two-pass centring in `AdaIN2d`:
`c = x - mean(x); c = c - mean(c); var = mean(c²)`. This brought resblock 5
from 50.1 to 59.8 dB locally.

### Rejected

- **Pre-scaling `c` by 1/16 before squaring**, meant to avoid a feared
  c² > 32k saturation that was never observed. The noise branch has std ~0.1,
  so `(c/16)²` lands in fp16 subnormals, which the ANE flushes to zero. Variance
  became 0 and the composite output fell to 16 dB.
  - Blocks tested alone looked fine, so this only showed up in composition.
- **Two-step (width, then rows) reductions alone:** no change.
- **Moving `conv_post` and the tail to fp32:** the loss is already in the body
  (ANE stage-2 tap plus fp32 `conv_post` still gives 39.3 dB).

---

## Issue: ANE op admission for the generator — Resolved

**First spotted:** 2026-10-21
**Status:** Resolved

### Summary

Bisection on an M2-generation ANE (MLComputePlan, one op per model and in
conv chains):

- Rejected to CPU: dilation-5 conv, both ConvTranspose upsamplers, the
  stride-6 noise conv, `sin`/`cos`, reflect pad.
- Placed on the ANE: dilation 1/3, AdaIN reductions, `round`, `floor`,
  `leaky_relu`, `exp`, `tanh`.
- `sin` places on the A17 Pro ANE; it does not on the M2 generation.

### Fix (all exact in fp32: 68 dB vs the unfolded generator)

- **Fold:** rows of `core` x_pre frames plus `halo` frames each side. The
  receptive field is ~8 frames; halo 8 works.
  - AdaIN uses a stats mask (each global index counted once, by its owner row)
    and a zero mask (valid audio, reproducing the unfolded zero padding).
  - The residual stream is zeroed outside valid audio at stage boundaries.
- **ConvTranspose** → polyphase conv to Cout·s channels plus a width shuffle.
- **Stride-6 conv** → space-to-depth.
- **Dilation 5** → rows in the batch axis plus a reshape of W to (W/5, 5) and a
  (11,1) kernel. The stage-2 width is 60·W0+5 so it is divisible by 5.
- **Snake** → folded: the AdaIN emits `a·y`, the next conv carries `1/a` in its
  weights, so the activation is `z + sin(z)²`.
  - Without `sin`: `f = u - round(u)`, `u = a·y/π`, and an even degree-5
    polynomial for `sin²(πf)`.
- **Model output** stops at the `conv_post` logits; the exp/sin/iSTFT tail runs
  in Swift (`foldedGeneratorAudio`, matches `CustomSTFT.inverse` to 2e-5).
- **Width limit:** a single 15,605-wide row failed ANECCompile for the whole
  graph (H-axis layout); 8,165 compiles.

---

## Issue: Sampled AdaIN statistics — Resolved (rejected)

**First spotted:** 2026-10-21
**Status:** Resolved

### Summary

The idea was to estimate every AdaIN's statistics from every k-th row (a pilot
pass), then stream rows with the statistics as inputs, which would bound RAM
and allow fused small calls. It fails: on 30 s the audio SNR is 1-4 dB for
k = 2-10 (k = 1 reproduces exact at 68 dB). Low-variance, high-mean channels
make the statistics fragile. Exact global statistics are required.

## Related Guides

- [Kokoro A14 iPhone generator execution guide][a14-guide] - the 16,384 axis
  cap and the earlier re-chunking options table
- [Core ML compute-unit ablation](coreml-compute-unit-ablation.md) - every
  earlier generator ANE attempt and why it failed
- [Core ML ANE compiler failure triage](../Guides/apple-silicon/CoreML-ANE-compiler-failure-triage-guide.md)

[a14-guide]: ../Guides/apple-silicon/Kokoro-A14-iPhone-generator-execution-guide.md
