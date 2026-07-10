# Kokoro M1 Graph Surface Target

June 6, 2026

This note turns the laishere-vs-first-party Core ML graph comparison into an
implementation target. It is not a general Core ML guide. It is the current
strict-parity frontier for making Config F faster than laishere on Irvine M1.
For external research on the unresolved kernel/partition behavior, use
`README/Kokoro-M1-kernel-partition-deep-research-prompt.md`.
For the narrower paper-frontier short-bucket target, use
`README/Kokoro-M1-paper-frontier-3s-7s-deep-research-prompt.md`.

## Current Short-Bucket Paper Target

The latest source/body plus HAR-post rewrite combined frontier is tracked in:

- `outputs/external_bakeoff/lower_end_mac_win_gate.md`
- `outputs/external_bakeoff/irvine_paper_frontier_path.md`

After adding the HAR-post rewrite and `decoder_pre`/HnSF overlap projections,
source/body still leaves Irvine `3s/7s` `27.0 ms` and `29.0 ms` short of the
paper-facing laishere rows. Do not treat newer profile-only wins as
paper-frontier wins.

| Bucket | Combined projected Config F | Paper frontier | Extra save needed |
| --- | ---: | ---: | ---: |
| `3s` | `203.3 ms` | `176.3 ms` | `27.0 ms` |
| `7s` | `423.6 ms` | `394.6 ms` | `29.0 ms` |

The useful next implementation target is therefore not a small surface cleanup.
It must change package boundary, tensor layout, runtime synchronization, or
source/body contract enough to remove tens of milliseconds on M1 short buckets.

## Current Surface Delta

Command used for the latest local refresh:

```bash
uv run --no-sync python scripts/compare_coreml_graph_surface.py \
  --model ours3=coreml/kokoro_decoder_har_post_3s.mlpackage \
  --model laishere_vocoder=outputs/external_bakeoff/laishere_packages/KokoroVocoder.mlpackage \
  --model laishere_noise=outputs/external_bakeoff/laishere_packages/KokoroNoise.mlpackage \
  --model laishere_tail=outputs/external_bakeoff/laishere_packages/KokoroTail.mlpackage \
  --output outputs/graph_surface/laishere_vs_local_generator_refresh.json
```

Summary:

| Model | Spec | Size | Ops | Conv | ConvT | InstNorm | ReduceMean | Tile | Sin | Cos | LUT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| first-party `GeneratorFromHar 3s` | 7 | `39.7 MB` | `2207` | `51` | `4` | `0` | `88` | `96` | `50` | `1` | `0` |
| laishere `KokoroVocoder` | 8 | `49.1 MB` | `1534` | `54` | `3` | `42` | `1` | `0` | `0` | `36` | `101` |
| laishere `KokoroNoise` | 8 | `4.7 MB` | `529` | `16` | `0` | `12` | `0` | `0` | `13` | `0` | `26` |
| laishere `KokoroTail` | 8 | `0.1 MB` | `41` | `1` | `2` | `0` | `0` | `0` | `2` | `1` | `0` |

The important point is not only operation count. The first-party strict fused
generator has manual AdaIN lowering with `88` reductions and `96` tiles, while
laishere's vocoder has native `instance_norm`, no tiles, and LUT-backed weight
decompression. Existing first-party native-InstanceNorm and cos-Snake probes
matched small pieces of this surface but did not reproduce laishere's runtime
benefit.

## Target

Find a strict single-package surface that eliminates the manual AdaIN
`reduce_mean`/`tile` footprint without creating the split-boundary sync penalty
seen in decoder+vocoder and generator-stage splits.

The next useful candidate must do at least one of these:

- preserve the `GeneratorFromHar` call boundary while replacing manual AdaIN
  lowering with native `instance_norm` and no materialized time-axis tiles;
- combine native `instance_norm` with a weight-compression surface that remains
  strict and avoids the previous palettization/linear-quantization failures;
- change the tensor layout enough that Core ML can select a laishere-like
  mixed CPU/Neural Engine partition without increasing call count;
- prove that a smaller Swift-produced source/HAR tensor can be consumed inside
  one package without the existing source/STFT strict-path regression.

## Non-Targets

Do not spend more time on these as standalone probes:

- native InstanceNorm alone;
- cos-Snake alone;
- iOS17/spec8 alone;
- broadcast AdaIN alone;
- style specialization;
- HAR trim alone;
- fused `GeneratorFromHar` fp16 input dtype alone;
- fused native `instance_norm` plus fp16 input dtype without eliminating tiles;
- fused native `instance_norm` plus broadcast AdaIN plus fp16 input dtype as a
  standalone surface-only change;
- fused native `instance_norm` plus broadcast AdaIN plus fp16 input dtype plus
  8-bit palettization as a standalone surface-only change;
- adding more package boundaries.

Each one has already been measured as slower, noise-sized, or quality-failing.

## Latest Rejection

Fused `GeneratorFromHar` fp16 input dtype was tested directly at the existing
single-package boundary:

```bash
uv run --no-sync python scripts/probe_generator_cos_snake.py \
  --no-cos-snake \
  --input-dtype fp16 \
  --label 3s_fused_input_dtype \
  --report-name report_fp16_inputs_cpu_gpu.json \
  --warmup 3 \
  --iterations 10
```

Result:

- strict pass: corr `1.0` vs fused trimmed, SNR `142.94 dB`, max abs `0`;
- warmed local M2 Studio CPU+GPU: fused `26.434 ms`, fp16-input candidate
  `26.453 ms`, speedup `-0.07%`;
- graph surface: `2207 -> 2201` ops, but still `88` reductions and `96` tiles.

Decision: reject as standalone. It does not remove the manual AdaIN/tile
surface and is not worth Irvine timing.

## Latest Partial Surface Repair

Fused native `instance_norm` plus fp16 input dtype was also tested inside the
same `GeneratorFromHar` package boundary:

```bash
uv run --no-sync python scripts/probe_generator_cos_snake.py \
  --no-cos-snake \
  --native-instance-norm-adain \
  --input-dtype fp16 \
  --deployment-target ios17 \
  --label 3s_native_in_fp16_inputs \
  --report-name report_ios17_native_in_fp16_inputs_cpu_gpu.json \
  --warmup 3 \
  --iterations 10
```

Result:

- strict pass: corr `0.9999942588867583` vs fused trimmed, SNR `49.84 dB`,
  max abs `0.00256348`;
- warmed local M2 Studio CPU+GPU: fused `26.349 ms`, candidate `26.316 ms`,
  speedup `0.12%`;
- graph surface: `2207 -> 1725` ops, `88 -> 0` reductions,
  `0 -> 44` native `instance_norm`, but still `96` tiles and no LUT
  decompression.

Decision: partial surface repair, not material. Do not promote to Irvine by
itself. A useful next candidate must also eliminate the tile footprint or
otherwise create a measurable local win before remote timing.

## Latest Near-Surface Match

Fused native `instance_norm`, broadcast AdaIN, and fp16 input dtype was tested
inside the same `GeneratorFromHar` package boundary:

```bash
uv run --no-sync python scripts/probe_generator_cos_snake.py \
  --no-cos-snake \
  --native-instance-norm-adain \
  --broadcast-adain \
  --input-dtype fp16 \
  --deployment-target ios17 \
  --label 3s_native_in_broadcast_fp16_inputs \
  --report-name report_ios17_native_broadcast_fp16_inputs_cpu_gpu.json \
  --warmup 3 \
  --iterations 10
```

Result:

- strict pass: corr `0.9999942588867583` vs fused trimmed, SNR `49.84 dB`,
  max abs `0.00256348`;
- warmed local M2 Studio CPU+GPU: fused `26.424 ms`, candidate `26.402 ms`,
  speedup `0.08%`;
- graph surface: `2207 -> 1533` ops, `88 -> 0` reductions, `96 -> 0` tiles,
  `0 -> 44` native `instance_norm`, but still no LUT decompression and no
  material local runtime gain.

Decision: this is the closest first-party fused graph surface to laishere so
far, but it proves that removing reductions and tiles is not sufficient by
itself. Do not promote to Irvine unless combined with a change that creates a
larger local win, changes placement, or adds laishere-like weight
decompression without quality/runtime failure.

## Latest Full Surface Match

Fused native `instance_norm`, broadcast AdaIN, fp16 input dtype, and 8-bit
palettization was tested inside the same `GeneratorFromHar` package boundary:

```bash
uv run --no-sync python scripts/probe_generator_cos_snake.py \
  --no-cos-snake \
  --native-instance-norm-adain \
  --broadcast-adain \
  --input-dtype fp16 \
  --deployment-target ios17 \
  --palettize \
  --label 3s_native_broadcast_fp16_pal8 \
  --report-name report_ios17_native_broadcast_fp16_pal8_cpu_gpu.json \
  --warmup 3 \
  --iterations 10
```

Result:

- strict threshold pass but thinner margin: corr `0.9998795313806623` vs fused
  trimmed, SNR `36.55 dB`, max abs `0.00868225`;
- warmed local M2 Studio CPU+GPU: fused `26.333 ms`, candidate `27.077 ms`,
  speedup `-2.83%`;
- graph surface: `2207 -> 1533` ops, `88 -> 0` reductions, `96 -> 0` tiles,
  `0 -> 44` native `instance_norm`, and `0 -> 101` LUT decompression ops.

Decision: reject as standalone. This is a strong negative result because it
matches the visible laishere-like surface most closely so far, yet loses local
warmed runtime and reduces quality margin. The remaining win is not explained
by surface counts alone; it likely depends on placement, layout, external
runtime scheduling, or a boundary effect outside this fused package.

## Latest Placement Check

`MLComputePlan` confirms that the visible graph-surface match still does not
produce laishere-like ANE residency. Under `cpuAndNeuralEngine`, laishere's
vocoder has `597` Neural-Engine-preferred ops and about `0.56` NE cost weight.
The first-party near-surface fused generator has the same order of operations
(`1533` vs laishere `1534`) and hundreds of NE-supported ops, but Core ML still
prefers `0` ops on the Neural Engine and assigns `0` NE cost weight:

| Package | Units | Ops | Preferred CPU | Preferred GPU | Preferred NE | Unknown | CPU cost | GPU cost | NE cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| laishere vocoder | CPU+NE | 1534 | 58 | 0 | 597 | 879 | 0.409 | 0 | 0.560 |
| first-party HAR-post baseline | CPU+NE | 2207 | 1038 | 0 | 0 | 1169 | 0 | 0 | 0 |
| first-party native-IN+broadcast+fp16 | CPU+NE | 1533 | 677 | 0 | 0 | 856 | 0 | 0 | 0 |
| first-party native-IN+broadcast+fp16+pal8 | CPU+NE | 1533 | 677 | 0 | 0 | 856 | 0 | 0 | 0 |
| first-party native-IN+broadcast+fp16 | CPU+GPU | 1533 | 0 | 678 | 0 | 855 | 0 | 1.000 | 0 |

Reports:

- `outputs/graph_surface/compute_plan_laishere_vocoder_cpu_ne.json`
- `outputs/external_bakeoff/compute_plan/ours_har_post_3s_cpu_ne.json`
- `outputs/graph_surface/compute_plan_generator_native_broadcast_fp16_3s_cpu_ne.json`
- `outputs/graph_surface/compute_plan_generator_native_broadcast_fp16_pal8_3s_cpu_ne.json`
- `outputs/graph_surface/compute_plan_generator_native_broadcast_fp16_3s_cpu_gpu.json`

Decision: do not keep optimizing by only matching op counts, AdaIN lowering, or
palettization. The current first-party fused package is GPU-friendly but not
ANE-preferred on M1. The next useful target is whatever changes Core ML's
placement decision: tensor layout, operation dialect, package boundary,
conversion target/toolchain, or a narrower laishere-style vocoder contract.

## Latest Cos-Snake Placement Check

The previous fp16/native-broadcast probes used `--no-cos-snake`, leaving
`50` `sin` ops and `48` `pow` ops. A combined cos-Snake, native
`instance_norm`, broadcast AdaIN, fp16-input package was tested to isolate
whether those ops were blocking ANE placement:

```bash
uv run --no-sync python scripts/probe_generator_cos_snake.py \
  --native-instance-norm-adain \
  --broadcast-adain \
  --input-dtype fp16 \
  --deployment-target ios17 \
  --label 3s_cos_native_broadcast_fp16_inputs \
  --report-name report_ios17_cos_native_broadcast_fp16_inputs_cpu_gpu.json \
  --warmup 3 \
  --iterations 10
```

Result:

- strict pass: corr `0.999994092360851` vs fused trimmed, SNR `49.71 dB`,
  max abs `0.00226212`;
- warmed local M2 Studio CPU+GPU: fused `26.416 ms`, candidate `26.348 ms`,
  speedup `0.26%`;
- graph surface: `1629` ops, `51` conv, `4` conv-transpose,
  `44` native `instance_norm`, `0` reductions, `0` tiles, `2` sin, `49` cos,
  `0` pow, `0` LUT ops;
- CPU+NE placement: `725` CPU-preferred ops, `0` NE-preferred ops,
  `904` unknown, `0` NE cost weight;
- CPU+GPU placement: `726` GPU-preferred ops, `903` unknown, GPU cost weight
  `1.0`.

Reports:

- `outputs/generator_cos_snake/3s_cos_native_broadcast_fp16_inputs_broadcast_adain_native_in_fp16_inputs_ios17/report_ios17_cos_native_broadcast_fp16_inputs_cpu_gpu.json`
- `outputs/graph_surface/fused_cos_native_broadcast_fp16_3s.json`
- `outputs/graph_surface/compute_plan_generator_cos_native_broadcast_fp16_3s_cpu_ne.json`
- `outputs/graph_surface/compute_plan_generator_cos_native_broadcast_fp16_3s_cpu_gpu.json`

Decision: reject as standalone. Removing the original `sin`/`pow` footprint
does not unlock ANE placement for the fused HAR-post package. The remaining M1
loss is still a placement/layout/boundary problem, not a single visible op-class
problem.

## Latest Strict Positive Candidate

Rewriting the two main generator upsample `ConvTranspose1d` layers as
zero-insertion plus normal `conv1d`, while keeping cos-Snake, native
`instance_norm`, broadcast AdaIN, fp16 inputs, and iOS17 export, is the first
strict local candidate in this sequence with a material warmed CPU+GPU win
across all runtime buckets:

```bash
uv run --no-sync python scripts/probe_generator_cos_snake.py \
  --native-instance-norm-adain \
  --broadcast-adain \
  --input-dtype fp16 \
  --rewrite-ups-conv-transpose \
  --deployment-target ios17 \
  --label 3s_cos_native_broadcast_fp16_inputs \
  --report-name report_ios17_cos_native_broadcast_fp16_ups_as_conv_cpu_gpu.json \
  --warmup 3 \
  --iterations 10
```

Local M2 Studio CPU+GPU results:

| Bucket | Strict | Fused | Candidate | Speedup | Corr | SNR | Max abs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3s | yes | 26.375 ms | 25.203 ms | 4.45% | 0.999994 | 49.65 dB | 0.00217 |
| 7s | yes | 53.897 ms | 52.053 ms | 3.42% | 0.999995 | 50.74 dB | 0.00208 |
| 10s | yes | 72.972 ms | 70.371 ms | 3.56% | 0.999995 | 50.27 dB | 0.00256 |
| 15s | yes | 106.084 ms | 102.349 ms | 3.52% | 0.999995 | 50.40 dB | 0.00244 |
| 30s | yes | 204.624 ms | 197.992 ms | 3.24% | 0.999995 | 50.37 dB | 0.00244 |

3s graph and placement:

- graph surface: `1629 -> 1651` ops versus the prior cos/native/broadcast/fp16
  candidate, `conv_transpose 4 -> 2`, `conv 51 -> 53`, `44` native
  `instance_norm`, `0` reductions, `0` tiles;
- CPU+GPU placement: `734` GPU-preferred ops, `917` unknown, GPU cost weight
  `1.0`;
- CPU+NE placement: `733` CPU-preferred ops, `0` NE-preferred ops,
  `918` unknown, `0` NE cost weight.

Reports:

- `outputs/generator_cos_snake/3s_cos_native_broadcast_fp16_inputs_broadcast_adain_native_in_fp16_inputs_ups_as_conv_ios17/report_ios17_cos_native_broadcast_fp16_ups_as_conv_cpu_gpu.json`
- `outputs/generator_cos_snake/7s_cos_native_broadcast_fp16_inputs_broadcast_adain_native_in_fp16_inputs_ups_as_conv_ios17/report_ios17_cos_native_broadcast_fp16_ups_as_conv_cpu_gpu.json`
- `outputs/generator_cos_snake/10s_cos_native_broadcast_fp16_inputs_broadcast_adain_native_in_fp16_inputs_ups_as_conv_ios17/report_ios17_cos_native_broadcast_fp16_ups_as_conv_cpu_gpu.json`
- `outputs/generator_cos_snake/15s_cos_native_broadcast_fp16_inputs_broadcast_adain_native_in_fp16_inputs_ups_as_conv_ios17/report_ios17_cos_native_broadcast_fp16_ups_as_conv_cpu_gpu.json`
- `outputs/generator_cos_snake/30s_cos_native_broadcast_fp16_inputs_broadcast_adain_native_in_fp16_inputs_ups_as_conv_ios17/report_ios17_cos_native_broadcast_fp16_ups_as_conv_cpu_gpu.json`
- `outputs/graph_surface/fused_cos_native_broadcast_fp16_ups_as_conv_3s.json`
- `outputs/graph_surface/compute_plan_generator_cos_native_broadcast_fp16_ups_as_conv_3s_cpu_gpu.json`
- `outputs/graph_surface/compute_plan_generator_cos_native_broadcast_fp16_ups_as_conv_3s_cpu_ne.json`

Decision: promote to Irvine M1 timing when the machine is quiet. This is not an
ANE unlock and will not by itself prove the paper claim, but it is a real
strict GPU-path optimization and the first local signal large enough to justify
remote timing.

## Production Exporter Smoke

The upsample rewrite is now available in the production `decoder-har` exporter
behind `--rewrite-ups-conv-transpose`; defaults are unchanged until Irvine M1
timing validates the candidate.

```bash
uv run --no-sync python -m export_synth.main \
  --output_dir outputs/export_rewrite_smoke \
  --buckets 3s \
  --mode decoder-har \
  --rewrite-ups-conv-transpose
```

The exported package has the intended production-default graph delta:

| Model | Ops | Conv | ConvT | InstNorm | ReduceMean | Tile | Sin | Cos |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| shipped `kokoro_decoder_har_post_3s` | 2207 | 51 | 4 | 0 | 88 | 96 | 50 | 1 |
| production rewrite smoke | 2229 | 53 | 2 | 0 | 88 | 96 | 50 | 1 |

All five runtime-bucket packages export and benchmark against the shipped
`coreml/` packages on real Swift generator tensor dumps. The first multi-bucket
export exposed an idempotence bug because the exporter reuses one in-memory
generator across buckets; rerunning the rewrite on an already rewritten
`ZeroInsertConvTranspose1d` layer failed. `rewrite_generator_ups_conv_transpose`
now skips already rewritten layers, so one process can export `3s/7s/10s/15s/30s`
safely. The 30s synthetic export gate still reports the known FP16 synthetic-HAR
non-finite warning, but the real-dump warmed benchmark output is finite and
strict-like.

Local M2 Studio CPU+GPU production-package results, warmed only:

| Bucket | Shipped | Rewrite | Speedup | Corr | SNR | Max abs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3s | 26.278 ms | 25.154 ms | 4.28% | 0.999995708 | 51.10 dB | 0.002197 |
| 7s | 53.827 ms | 52.134 ms | 3.15% | 0.999996566 | 52.17 dB | 0.001953 |
| 10s | 73.072 ms | 70.753 ms | 3.17% | 0.999995896 | 51.41 dB | 0.002426 |
| 15s | 105.842 ms | 103.086 ms | 2.60% | 0.999996419 | 51.98 dB | 0.002563 |
| 30s | 204.765 ms | 200.253 ms | 2.20% | 0.999995368 | 50.88 dB | 0.002930 |

Reports:

- `outputs/export_rewrite_smoke/report_all_buckets_cpu_gpu.json`
- `outputs/export_rewrite_smoke/report_3s_cpu_gpu.json`
- `outputs/graph_surface/production_rewrite_ups_as_conv_3s.json`

Decision: production integration path is open and opt-in. The local all-bucket
package proof is positive but smaller than the probe-only result because it uses
the shipped production graph surface rather than the fully rewritten
cos/native-IN/broadcast/fp16 probe surface. Next proof must be Irvine M1 warmed
timing and, if positive, regenerating the actual five shipped
`coreml/kokoro_decoder_har_post_{3s,7s,10s,15s,30s}.mlpackage` packages.

The end-to-end Config F overlay also moves in the right direction locally:
redirecting only the five HAR-post generator packages to the rewrite export
improves warmed M2 Studio medians by `1.97%` `3s`, `1.79%` `7s`, `1.62%`
`10s`, `1.22%` `15s`, and `2.58%` `30s` versus the current best local
`vector_noise_batch` result. Treat that as local implementation proof, not a
paper frontier update, until Irvine M1 repeats it under quiet warmed conditions.
The projection artifacts at
`outputs/external_bakeoff/rewrite_candidate_impact.md` and
`outputs/external_bakeoff/overlap_rewrite_candidate_impact.md` show why this
cannot be the final answer alone: overlap + rewrite likely closes M2 Air and
Irvine `15s` profile rows, but still leaves Irvine `3s/7s/10s` profile gaps of
`27.0 ms`, `28.0 ms`, and `12.8 ms`. The rewrite is a keeper, but it must
combine with another strict source/body gain to prove absolute fastest on
Irvine M1.

The larger cos/native-IN/broadcast/fp16/ups-as-conv probe packages were also
tested as a Swift overlay and are contract-compatible, but they do not beat the
simpler production-rewrite overlay overall. Local M2 Studio full-surface medians
were `49.532/94.128/123.617/183.980/376.200 ms` for
`3s/7s/10s/15s/30s`; that is only better than the production rewrite on `3s`
and effectively tied on `10s`, while slower on `7s/15s/30s`. Decision: do not
promote the full probe surface as the production package set.

## Deep Research Request

A useful external deep-research guide would be narrower than "Core ML
optimization." Ask for:

> How can an MLProgram for a 1D vocoder on M1 replace manual AdaIN
> reduce/tile lowering with native instance normalization or an equivalent
> ANE-friendly pattern, while preserving fixed-shape Core ML input/output
> contracts and avoiding CPU/NE synchronization regressions? Include concrete
> PyTorch/coremltools rewrite patterns, expected MIL ops, and ways to verify
> residency with `MLComputePlan`, Instruments, or Core ML performance reports.

The guide should specifically address why a graph with visible
Neural-Engine-preferred ops can still lose warmed runtime, because our strict
decoder+vocoder split already demonstrates that placement alone is not enough.
