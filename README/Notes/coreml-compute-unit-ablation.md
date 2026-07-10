# Core ML Compute-Unit Ablation Notes

Institutional memory for isolating Core ML `.all`, `.cpuAndGPU`,
`.cpuAndNeuralEngine`, and `.cpuOnly` behavior in the Swift Kokoro pipeline.

**Quick filter:** `grep -n "— Active" README/Notes/coreml-compute-unit-ablation.md`

---

## Issue: Swift Core ML `.all` Is Slower Than `.cpuAndGPU` On M2 Air-Class Hardware — Active

**First spotted:** 2026-04-17
**Status:** Active

### Summary

The first F/G ablation shows that allowing ANE via Core ML `.all` does not help
on the local Apple M2 24 GB machine. Config G (`.cpuAndGPU`) beats Config F
(`.all`) at every benchmark length, especially 15s and 30s, where the slowdown
is concentrated in `GeneratorFromHar`.

Latency alone does not prove `.all` actually used ANE. It proves only that
allowing ANE did not help. The next ablations must distinguish a bad ANE path
from a bad `.all` mixed execution plan and from GPU doing the useful work.

### Current Evidence

Completed run:

```bash
BAKEOFF_SKIP_SMOKE=1 uv run --no-sync python scripts/bakeoff_harness.py run \
  --configs f,g \
  --iterations 5 \
  --order-seed 0 \
  --machine-id ane_ablation_fg_local
```

Result file:

- `outputs/bakeoff/results_ane_ablation_fg_local.json`

Machine:

- Apple M2, 24 GB unified memory
- macOS 15.7.5
- Git commit `9738030122ac00fe5fbe25930c094c467e70552a`
- Dirty tree: true

Median wall time:

| Input | F `.all` | G `.cpuAndGPU` | G/F |
| --- | ---: | ---: | ---: |
| 3s | 227.5 ms | 175.7 ms | 0.77 |
| 7s | 478.8 ms | 384.2 ms | 0.80 |
| 15s | 2051.9 ms | 802.3 ms | 0.39 |
| 30s | 4803.5 ms | 1592.2 ms | 0.33 |

Median `GeneratorFromHar` time:

| Input | F `.all` | G `.cpuAndGPU` |
| --- | ---: | ---: |
| 3s | 195.3 ms | 131.7 ms |
| 7s | 421.6 ms | 302.3 ms |
| 15s | 1911.9 ms | 638.8 ms |
| 30s | 4512.6 ms | 1273.7 ms |

### Interpretation Boundary

This result does **not** prove ANE execution was bad, because `.all` is a
scheduler request, not proof of active ANE placement. It establishes a narrower
but important claim:

> On this machine and artifact set, Core ML `.all` is slower than excluding ANE
> with `.cpuAndGPU` for the Swift decomposed pipeline.

Three mechanisms remain possible:

1. **ANE path hurts:** the ANE placement itself is slow for these bucket shapes.
2. **Mixed `.all` plan hurts:** `.all` creates a bad CPU/GPU/ANE partition or
   synchronization pattern, while ANE-only plus CPU might be fine.
3. **ANE fallback hurts:** `.all` attempts ANE, then spills or falls back into a
   worse execution path than explicitly excluding ANE.

### Required Ablations

Add two Swift Core ML compute-unit controls to the harness:

| Human label | Suggested harness id | Core ML compute units | Purpose |
| --- | --- | --- | --- |
| G-prime | `gne` | `.cpuAndNeuralEngine` | Force CPU + ANE only; exclude GPU. |
| G-double-prime | `gcpu` | `.cpuOnly` | Exclude both GPU and ANE. |

The full compute-unit matrix:

| Config | Core ML compute units | Meaning |
| --- | --- | --- |
| F | `.all` | Let Core ML choose CPU, GPU, and ANE. |
| G | `.cpuAndGPU` | Exclude ANE. |
| G-prime | `.cpuAndNeuralEngine` | Exclude GPU. |
| G-double-prime | `.cpuOnly` | CPU baseline for Swift Core ML packages. |

Decision logic:

| Result pattern | Interpretation |
| --- | --- |
| G-prime is catastrophic like F at 15s/30s | ANE path is likely the problem, or `.all` and CPU+ANE share the same bad fallback. |
| G-prime is fast while F is slow | The problem is specifically the `.all` mixed plan. |
| G-double-prime is about 2x slower than G | GPU is doing useful work in G. |
| G-double-prime is near G | The win is mostly "not ANE"; GPU is not carrying much useful work. |

### Cross-Machine Gates

Before rewriting the paper framing, run F/G at minimum on Ultra and Mini with
the same artifacts and harness shape.

| Scenario | Paper implication |
| --- | --- |
| G beats F on Ultra, Air, and Mini | Thesis is inverted: Swift + decomposed Core ML on CPU+GPU beats PyTorch-on-MPS, and ANE does not help this workload. |
| G beats F on Air, but F beats G on Ultra | Most interesting result: ANE behavior is hardware-dependent for generative audio, and consumer-tier chips can be penalized by `.all`. |
| F beats G on Ultra and Mini, but G beats F only on Air | Air may be an outlier due to thermal state, memory pressure, or a chip-specific Core ML plan. Paper can survive with a caveat. |

### Related Guides

- [Core ML compute-unit scheduling](../Guides/apple-silicon/CoreML-Compute-Unit-Scheduling-guide.md) - explains `.all`, `.cpuAndGPU`, `.cpuAndNeuralEngine`, silent fallback, and powermetrics verification.
- [Bakeoff results v2](bakeoff-results-v2.md) - current cross-machine A/D/E/F benchmark context before this F/G ablation.
- [Performance notes](performance-notes.md) - historical timing and Core ML performance observations.

### Verification Gap

`powermetrics` was not captured during the first F/G run because non-interactive
sudo was unavailable:

```log
sudo: a password is required
```

For publication claims about ANE participation, latency comparisons must be
paired with telemetry:

```bash
sudo powermetrics --samplers cpu_power,gpu_power,ane_power -i 1000
```

If the sampler names differ on the target macOS version, use the ANE-only form
from the scheduling guide:

```bash
sudo powermetrics -i 1000 --samplers ane
```

### Next Steps

- [ ] Add Swift Core ML `G-prime` (`.cpuAndNeuralEngine`) to the harness.
- [ ] Add Swift Core ML `G-double-prime` (`.cpuOnly`) to the harness.
- [ ] Run F/G/G-prime/G-double-prime on the local M2 24 GB machine.
- [ ] Run F/G on M2 Ultra.
- [ ] Run F/G on M1 Mini.
- [ ] Capture powermetrics during steady-state F and G loops.
- [ ] Update [bakeoff results v2](bakeoff-results-v2.md) only after cross-machine data is collected.

### Investigation Log

**2026-06-05**

- **Generator-isolation harness:** `kokoro-bench --generator-input-dump` now
  accepts `--warmup` and `--iterations`, so the generator can be timed against
  a dumped Swift tensor boundary (`x_pre_padded`, `ref_s`, `har_padded`) without
  rerunning duration, F0, decoder-pre, HnSF, trimming, or JSON orchestration.
- **Generator compute-unit result:** CPU+GPU remains the fastest policy for the
  GPU-preferred `GeneratorFromHar` graph on the losing machines. N=10 medians
  after three discarded warmups:

  | Machine | Input | `cpuAndGPU` | `.all` | `cpuAndNeuralEngine` | `cpuOnly` |
  | --- | --- | ---: | ---: | ---: | ---: |
  | m2-studio | 3s | 27.2 ms | 27.0 ms | 1535.5 ms | 100.9 ms |
  | m2-studio | 7s | 59.5 ms | 60.3 ms | not rerun | not rerun |
  | m2-air | 3s | 120.1 ms | 155.4 ms | not rerun | not rerun |
  | m2-air | 7s | 277.6 ms | 426.2 ms | not rerun | not rerun |
  | irvine-m1 | 3s | 168.9 ms | 172.8 ms | not rerun | not rerun |
  | irvine-m1 | 7s | 384.7 ms | 394.2 ms | not rerun | not rerun |

- **Decision:** Do not spend more time trying to make the existing generator
  package faster by changing `MLComputeUnits`. The path to beating laishere on
  M2 Air/M1 short and medium rows is generator graph work: splitting the graph
  into noise/vocoder/tail stages, or rewriting the ANE-hostile operator surface
  (`conv_transpose`, long harmonic temporal axes, and Snake activations lowered
  to `sin`).
- **Exact-generator-only geometry rejected:** `scripts/probe_generator_exact_geometry.py`
  exported exact 2.8s and 6.75s `GeneratorFromHar` packages from the dumped
  Swift tensor boundary. They ran, but failed parity against the current
  trimmed bucket reference: 2.8s corr `0.927`, SNR `8.87 dB`, max abs `0.240`,
  median `27.1 ms`; 6.75s corr `0.952`, SNR `10.73 dB`, max abs `0.389`,
  median `55.7 ms`. A shorter HAR-post package fed by cropped bucket tensors is
  not a production-safe optimization. If exact-duration generator packages are
  revisited, they need an end-to-end exact graph plus listening/quality gates.
- **Final-tail split rejected:** `scripts/probe_generator_split.py` exported a
  body package ending at pre-tail logits plus a tiny tail package for
  `exp`/`sin`/iSTFT. Parity passed, but the split was slower on local M2 Studio:
  3s fused `28.3 ms` vs split `29.1 ms`; 7s fused `57.5 ms` vs split `58.4 ms`.
  The tail costs only `0.8-1.2 ms`, so separating it does not address the body
  bottleneck.
- **HAR-noise split rejected as a direct static-HAR optimization:** `scripts/probe_generator_noise_split.py`
  exported `ref_s + har -> x_source_0/x_source_1` and a separate body package.
  Output matched the fused package exactly on 3s and 7s, proving the boundary is
  semantically safe, but total latency got worse: 3s fused `28.9 ms` vs split
  `32.5 ms`; 7s fused `57.6 ms` vs split `63.4 ms`. The body package shrank
  materially (`19.5 ms` at 3s and `37.7 ms` at 7s), but the noise package
  (`13.1 ms` / `25.8 ms`) plus dispatch overhead outweighed it. Forcing the
  noise-split body to CPU+ANE was catastrophic (`~237-248 ms` body at 3s), so
  the direct split does not recreate laishere's ANE residency.
- **Actual laishere lesson:** its own README says tail split alone failed. Its
  working path combines a separate fp32 noise stage, a dual-output vocoder with
  a discarded anchor to influence scheduling, a separate fp32 tail, cos-form
  Snake, and palettization where the audio output is discarded. The next graph
  experiment should port that scheduler trick or rewrite the body operator
  surface; more naive split boundaries are not supported by the data.
- **Dual-output anchor trick rejected for the current HAR-post graph:** `scripts/probe_generator_dual_anchor_split.py`
  ports the remaining visible laishere ingredients onto the static Swift dump
  boundary. On local M2 Studio `3s`, mean-anchor + cos-Snake + fp32 tail on
  CPU+ANE still spent `236.7 ms` in the vocoder and `249.8 ms` total versus
  `29.3 ms` fused. The audio-anchor variant was also slow (`242.3 ms` total,
  N=3 rejection run), and int8 palettizing the vocoder did not help
  (`252.3 ms` total, N=3 rejection run, with lower parity margin). The CPU+GPU
  variants passed parity but stayed slower than fused (`32.0-33.0 ms` split
  versus `28.1-28.9 ms` fused). This closes the "maybe laishere's dual output
  alone fixes scheduling" hypothesis for our current generator boundary.
- **Revised decision:** stop adding generator split boundaries unless a new
  hypothesis changes the operator surface. The remaining path to beat
  laishere's chain-only M2 Air/M1 short buckets is to reduce generator math or
  rewrite the GPU/ANE-hostile operators themselves (`conv_transpose`, AdaIN
  reductions/broadcasts, and Snake lowering), or to evaluate a larger
  end-to-end graph reshape rather than the already-rejected static HAR-post
  splits.
- **Visible laishere graph-surface rewrites rejected:** `scripts/probe_generator_cos_snake.py`
  now patches the actual dynamically loaded export module
  (`export_synth.wrappers.kokoro_istftnet`) and can test iOS17/CoreML7 target
  packages, broadcast AdaIN, native `nn.InstanceNorm1d` AdaIN, and cos-form
  Snake. `scripts/compare_coreml_graph_surface.py` records the MIL histogram.
  The strongest combined 3s candidate drops the fused graph from `2207` ops to
  `1635`, removing explicit `tile`, explicit reduction-based normalization,
  and nearly all Snake `sin` ops. It still does not speed prediction:
  M2 Studio `30.08 ms` fused vs `30.68 ms` candidate, M2 Air `120.51 ms` vs
  `120.65 ms`, and Irvine M1 `167.83 ms` vs `167.60 ms`. Parity passed
  (`corr >= 0.999994`, `SNR >= 49.7 dB`). This rejects the simple hypothesis
  that laishere wins because of visible MIL op cleanup alone.
- **Palettized fused generator rejected:** the same probe can now apply 8-bit
  k-means Core ML palettization. The strongest visible-surface candidate
  (native InstanceNorm + broadcast AdaIN + cos Snake + pal8) reproduced
  laishere's visible LUT surface (`101` `constexpr_lut_to_dense` ops) and cut
  the 3s package from `38M` to `19M`, but local M2 Studio predict-only latency
  worsened from `29.64 ms` fused to `31.31 ms` and the row missed the existing
  max-abs gate (`0.01007` vs threshold `0.01`). This rejects "palettize the
  fused final-waveform package" as the explanation for laishere's M1 lead.
  Laishere's palettized vocoder output is discarded before its separate tail,
  so its quantization error does not map directly onto our fused final
  waveform output.
- **Linear-quantized fused generator rejected:** `scripts/probe_generator_cos_snake.py`
  now also supports Core ML linear weight quantization via
  `--linear-quantize {int8,uint8,int4,uint4}`. A plain fixed-shape 3s int8
  candidate compressed the package from `39.7 MB` to `20.2 MB` while preserving
  the same visible MIL op histogram (`2207` ops, `51` conv, `4` conv_transpose,
  `88` reduce_mean, `96` tile, `50` sin). Both macOS13 and iOS17 CPU+GPU runs
  crashed during runtime specialization with
  `MPSGraphExecutable.mm:5070: failed assertion 'Error: MLIR pass manager failed'`.
  The saved package loads under CPU-only, but CPU-only predict is slow
  (`93.27 ms` vs `97.43 ms` fused CPU-only) and fails the waveform gate
  (`corr 0.999051`, SNR `27.62 dB`, max abs `0.03387`). This rejects Core ML
  linear weight compression for the final-waveform generator path.
- **coremltools 9 conversion-only path rejected:** the probe now records
  conversion toolchain versions. A plain iOS17 fused-generator export with
  `coremltools==9.0` preserved the same visible MIL surface as CT8 (`2207`
  ops; `51` conv, `4` conv_transpose, `88` reduce_mean, `96` tile, `50` sin).
  Same-process local 3s timing was a tie: shipping fused `30.07 ms`, CT8 iOS17
  `29.76 ms`, CT9 iOS17 `29.81 ms`. Remote 3s timing also tied on the losing
  machines: M2 Air shipping `120.803 ms` vs CT9 `120.816 ms`; Irvine M1
  shipping `167.900 ms` vs CT9 `167.947 ms`. The 7s local CT9 candidate was
  slower (`60.49 ms` vs `59.87 ms`). Do not pursue a CT9-only migration for the
  current fused final-waveform package without a new compute-plan signal.
- **RangeDim input contract rejected for the fused generator:** laishere's
  vocoder uses flexible ranges, so `scripts/probe_generator_cos_snake.py` now
  supports `--input-shape-mode range` for `x_pre` and `har`. Plain RangeDim
  converted but ran catastrophically (`1561.07 ms` vs `49.73 ms` fused) and
  emitted E5RT `tile` shape-propagation failures. The tile-free
  native+broadcast+cos graph still failed quality and speed on both CT8 and
  CT9 (`343.61-343.96 ms` candidate vs `31.91-33.05 ms` fused) with E5RT
  dynamic `add` broadcast failures. The flexible-shape advantage, if any, is
  tied to laishere's different split boundary; it is not a safe drop-in for the
  current fused final-waveform package.
- **Style-specialized generator rejected:** `scripts/probe_generator_style_specialization.py`
  bakes the dump's `ref_s` into the generator and replaces all `AdaIN1d`
  projections with fixed gamma/beta constants. The 3s MIL graph shrank from
  `2207` to `1625` ops and removed all `linear`/`reshape`/`split`/`tile` ops,
  but latency got worse on every tested Mac: M2 Studio `31.3 ms` fused vs
  `31.9 ms` specialized, M2 Air `120.8 ms` vs `123.0 ms`, and Irvine M1
  `167.8 ms` vs `170.8 ms`. It also bloats artifacts (`315 MB` for 3s,
  `690 MB` for 7s). Per-voice generator packages are not a speed path.
- **Per-stage generator split rejected as a production split and useful as a
  profiler:** `scripts/probe_generator_stage_split.py` splits the current
  static HAR-post generator into noise, first upsample/resblock stage, and
  second upsample/resblock stage plus tail. CPU+GPU parity passed, but the split
  is slower than fused because of extra package dispatch: `3s` fused `28.6 ms`
  vs split `33.0 ms`; `7s` fused `58.5 ms` vs split `66.5 ms`. The 3s stage
  medians are noise `12.6 ms`, stage0 `9.0 ms`, and stage1+tail `11.3 ms`.
  CPU+ANE is not viable for any substage: noise `67.0 ms`, stage0 `37.8 ms`,
  and stage1+tail `93.1 ms` when each is isolated with the other stages on
  CPU+GPU. The 3s MIL operation distribution is broad: fused `2207` ops, noise
  `562`, stage0 `807`, stage1+tail `856`. There is no hidden ANE island inside
  the current generator; the next real optimization must remove/rewrite work
  across the generator regions instead of only repartitioning them.
- **Remote stage-placement check:** the same 3s stage packages and tensor dump
  were copied to `m2-air` and `irvine-m1` and run predict-only. CPU+GPU parity
  passed, with M2 Air stage medians noise `51.2 ms`, stage0 `31.2 ms`,
  stage1+tail `44.1 ms`, and Irvine M1 medians noise `74.4 ms`, stage0
  `44.9 ms`, stage1+tail `64.6 ms`. CPU+ANE substage placement is invalid on
  both losing machines: M2 Air stage0 CPU+ANE corr `0.403806`, stage1+tail
  CPU+ANE corr `0.120825`; Irvine M1 stage0 CPU+ANE corr `0.403829`,
  stage1+tail CPU+ANE corr `0.121443`. Do not use ANE for any current generator
  substage on these hosts.
- **Laishere stage profile narrowed the remaining loss:** `scripts/external_bakeoff/profile_laishere_stages.py`
  times laishere's seven-package chain by stage while preserving its public
  timing boundary. On M2 Air, laishere's noise+vocoder+tail portion is
  effectively tied with our isolated generator (`3s` `123.7 ms` vs `120.1 ms`;
  `7s` `279.0 ms` vs `277.6 ms`). Source audit later showed this is not a
  pure same-boundary comparison: laishere's `KokoroVocoder` includes F0/N conv,
  decoder encode/decode, and the generator body, then emits `x_pre` for a
  separate fp32 tail. On Irvine M1, laishere remains faster at that broader
  boundary (`3s` `145.1 ms` vs our generator `168.9 ms`; `7s` `340.4 ms` vs
  `384.7 ms`) and also has faster upstream stages. The next useful comparison
  is an exact laishere-style decoder-plus-generator boundary probe against the
  Swift dumps, not more `MLComputeUnits` experiments or static HAR-post
  boundary splitting.
- **Laishere-style decoder+vocoder boundary rejected on cross-machine 3s:** `scripts/probe_decoder_vocoder_split.py`
  ports the broader laishere boundary onto the Swift tensor dumps: checked-in
  `decoder_pre` + fused HAR-post generator as the baseline versus a candidate
  chain of HAR-noise, decoder encode/decode plus generator body with a discarded
  anchor, and an fp32 tail. Parity passed against the fused baseline, but the
  boundary did not win on any tested Mac. M2 Studio CPU+ANE emitted an ANE
  compiler failure and lost (`119.5 ms` vs `32.9 ms` baseline); M2 Air CPU+ANE
  was stopped after more than `110s` in `ANECompilerService`; Irvine M1 CPU+ANE
  lost (`314.6 ms` vs `176.3 ms`). CPU+GPU removed the ANE catastrophe but still
  lost on every machine: M2 Studio `38.0 ms` vs `33.2 ms`, M2 Air `138.8 ms` vs
  `123.7 ms`, and Irvine M1 `199.3 ms` vs `174.6 ms`. This rejects a simple
  "move our split to laishere's decoder+generator boundary" explanation. The
  remaining laishere gap is either from its full chain/runtime details, a
  hardware-specific Core ML compile plan, or work reduction outside this
  boundary; do not ship this split.
- **F0-noise exact-shape path is the next real target, but not quality-safe yet:**
  feeding the Swift dumps into the pinned laishere `KokoroNoise`/`KokoroVocoder`/
  `KokoroTail` packages shows the missing speed ingredient is not another
  HAR-post split; it is removing the HAR input and using exact dynamic lengths.
  On Irvine M1, natural-shape `3s` (`asr=112`, `F0=240`) took `135.4 ms` for
  noise+vocoder+tail versus the current `DecoderPre + Swift HnSF + Generator`
  stack at `199.3 ms`. Natural-shape `7s` (`asr=270`, `F0=540`) took
  `337.4 ms` versus the current stack at `443.1 ms`. Padded shapes are much
  slower (`245.1 ms` for 3s), so exact dynamic shape is part of the speed story.
  This path is not shippable yet: waveform parity versus the current dump is
  poor (`3s` corr `0.699830`, SNR `0.56 dB`; `7s` corr `0.701953`, SNR
  `0.64 dB`). Next work should build a first-party F0-noise exact-shape probe
  and run listening/quality recovery; do not spend more time repartitioning the
  current HAR-post package.
- **First-party F0-noise exact-shape probe confirms speed but rejects quality:**
  `scripts/probe_f0_noise_exact_shape.py` exports the F0-noise, body, and tail
  packages from local weights. On M2 Studio 3s natural shape (`asr=112`,
  `F0=224`), candidate runtime tied/slightly beat the baseline (`32.7 ms` vs
  `33.4 ms`) but failed parity (corr `0.814046`, SNR `5.08 dB`). Padded shape
  (`asr=120`, `F0=240`) improved parity (corr `0.931896`, SNR `9.19 dB`) but
  lost speed (`33.7 ms` vs `33.5 ms`). On Irvine M1 natural shape, the
  candidate was materially faster (`153.3 ms` vs `172.0 ms`) but failed the
  same parity threshold (corr `0.814046`, SNR `5.08 dB`). The 7s probe keeps the
  speed-positive signal: M2 Studio natural shape is `56.5 ms` vs `63.1 ms`
  (`+10.4%`), M2 Studio padded is `58.8 ms` vs `63.0 ms` (`+6.7%`), Irvine M1
  natural is `349.8 ms` vs `398.4 ms` (`+12.2%`), and Irvine M1 padded is
  `358.9 ms` vs `390.8 ms` (`+8.2%`). The PyTorch candidate metrics are
  similarly poor for natural shape (`corr ~0.796`, SNR `~4.33-4.54 dB`), so the
  issue is inherent source-path drift rather than Core ML conversion drift. Next
  work must recover or validate audio quality before any runtime integration.
- **F0-source candidates are listening-ready but not approved:** `scripts/create_f0_source_listening_pack.py`
  renders saved F0-source probe reports into WAVs, waveform plots, quality
  reports, and fillable listening reviews without ASR/Whisper. The local 3s and
  7s natural and padded F0-source candidates all pass the machine
  waveform-health gate as `needs_listening`, not `reject_without_listening`, but
  strict waveform parity still rejects them (`3s` natural corr `0.814034`, SNR
  `5.08 dB`; `3s` padded corr `0.931895`, SNR `9.19 dB`; `7s` natural corr
  `0.796791`, SNR `4.77 dB`; `7s` padded corr `0.962251`, SNR `11.51 dB`).
  Treat this as a human listening gate or source-formulation research path, not
  a runtime integration approval.
- **HAR-source boundary is not the escape hatch:** `scripts/probe_f0_source_variants.py`
  shows the F0-source downsample shortcut is not the main quality loss:
  `avg_pool` and linear interpolation are effectively tied against dumped Swift
  `har_source` (`3s` corr `0.93978`, `7s` corr `0.96731`). Recomputed STFT from
  dumped source is magnitude-exact and phase-equivalent modulo `2*pi`, and a
  PyTorch sensitivity check remains high parity (`3s` waveform corr `0.99881`,
  `7s` corr `0.99846`). `scripts/probe_har_source_noise_split.py` then exports a
  temporary exact-source boundary (`har_source + style -> x_source_*`) and
  rejects it as the winning path: 3s is slower before source-generation cost
  (`36.5 ms` vs `35.2 ms`) and 7s has only a small pre-source speedup
  (`62.8 ms` vs `67.4 ms`), while both fail strict waveform parity (corr
  `~0.98`, SNR `~13.1 dB`).
- **Fused HAR-source graph is fast but not parity-safe:** `scripts/probe_har_source_fused.py`
  avoids the lossy body/tail split by exporting one temporary
  `x_pre + ref_s + har_source -> waveform` graph. The package is materially
  faster than the current `GeneratorFromHar` package on local warmed inference
  (`3s` fp16 `26.4 ms` vs `30.3 ms`; `7s` fp16 `51.2 ms` vs `60.9 ms`), but it
  still fails waveform parity (`3s` corr `0.980656`, SNR `13.10 dB`; `7s` corr
  `0.979271`, SNR `13.06 dB`). fp32 conversion of the native-`atan2` graph does
  not fix it (`3s` corr only `0.981718`), and CPU-only Core ML execution still
  fails (`3s` corr `0.979581`, SNR `12.61 dB`). Treat this as a source/STFT
  contract recovery target, not an ANE scheduling issue or production
  replacement.
- **Converted `atan2` is the first bad op in fused HAR-source:** `scripts/probe_coreml_stft_semantics.py`
  exports only `har_source -> magnitude, phase, real, imag`. Core ML matches
  PyTorch/Swift for `real`, `imag`, and `magnitude` (all effectively corr
  `1.0`, SNR `~63 dB`), but converted `torch.atan2(imag, real)` phase fails even
  on `cpu_only` (corr `0.818405`, SNR `4.67 dB`). `acos` and manual-quadrant
  `atan` formulas improve the fused waveform to corr `~0.987`, but still fail
  strict parity because raw `+pi/-pi` branch choices are ordinary feature values
  to the downstream convs. The follow-up `scripts/probe_har_source_fused_debug.py`
  proves fp32 manual `atan` is Core ML-safe (Core ML vs PyTorch waveform corr
  `1.000000`, SNR `72.24 dB`) but the PyTorch/Core ML source-boundary waveform
  is still only corr `0.987820` vs the Swift dump. Padding the recomputed HAR
  back to the shipping 3s `28801`-frame input restores most quality (corr
  `0.998808`, SNR `26.44 dB`) but loses the package speed edge (`27.2 ms`
  candidate vs `26.9 ms` baseline) and still fails strict parity. The remaining
  raw phase mismatch is isolated to the Nyquist phase channel (`10`): channels
  `0-9` match Swift, and replacing only channel `10` with the dumped Swift phase
  restores PyTorch waveform parity to corr `0.999991`, SNR `47.76 dB`. Future
  work should avoid raw phase discontinuities or reproduce the exact Swift HAR
  contract before revisiting the fused source speed path.
- **Swift-float DFT basis is not the Nyquist fix:** Recomputing the DFT basis
  with Swift-like float32 trigonometry preserves magnitude (SNR `124.76 dB`) but
  worsens Nyquist raw branch parity (`2871` raw `2*pi` errors, channel-10 corr
  `0.139881`). The issue is not merely NumPy-double constants in
  `CustomSTFT`; it is the raw Nyquist phase convention reaching learned conv
  features.
- **Nyquist neutralization is not quality-safe:** `scripts/probe_nyquist_phase_contribution.py`
  shows the Nyquist phase channel is low-weight-mass but still quality-sensitive
  in `noise_convs`. At padded shipping length, copying the dumped Nyquist phase
  recovers parity (`3s` corr `0.9999909`, `7s` corr `0.9999933`), but zero,
  mean, `+pi`, and `-pi` substitutes all fail strict waveform parity. At compact
  natural HAR length, even exact dumped HAR remains around corr `0.986-0.988`,
  proving a separate natural-vs-padded geometry loss.
- **Generator compute-unit switch is not the M2 Air fix:** Swift generator-input
  isolation on local M2 Studio 3s with five warmups and twenty measured calls
  gives CPU+GPU `28.289 ms`, `.all` `28.071 ms`, CPU-only `99.673 ms`, and
  CPU+NE `1517.266 ms`. The CPU+NE run also printed
  `MILCompilerForANE error: failed to compile ANE model using ANEF`. Keep the
  staged production policy's generator on CPU+GPU; the remaining short-bucket
  gap requires graph/package changes, not a compute-unit flag flip.
- **Current lower-end gap is source/vocoder contract, not prefix:** Comparing
  corrected Config F HAR-direct-pad stage medians with laishere's stage-profile
  records shows M2 Air is effectively tied at 3s/7s, while Irvine M1 still loses
  in the combined HnSF/generator region. On M1 3s, Config F HnSF+generator is
  `194.9 ms`; laishere noise+vocoder+tail is `145.1 ms`. On M1 7s, Config F is
  `434.7 ms`; laishere is `340.4 ms`. Do not spend more turns on Duration,
  F0Ntrain, DecoderPre, compute-unit flags, or generic compression unless a new
  profile contradicts this split.
- **HAR input-tail trimming is too small:** `scripts/probe_generator_har_input_trim.py`
  keeps the bucketed `x_pre` shape and current Swift HAR source, but exports a
  temporary `GeneratorFromHar` with a shorter static `har` axis. The aggressive
  local 3s trim to `har_time=27601` was only `1.07%` faster and failed strict
  parity (corr `0.999827`, SNR `35.05 dB`, max `0.02661`). The first strict
  quality-safe trim, `har_time=28561`, was slower on M2 Studio (`30.41 ms` vs
  `30.07 ms`) and only `0.43%` faster on Irvine M1 (`167.64 ms` vs
  `168.36 ms`). This rejects "slightly shorter HAR padding" as the missing
  laishere/fastest-everywhere ingredient.
- **Laishere-style fp16 vocoder inputs are not sufficient:** The source audit
  found that laishere's `KokoroVocoder` declares fp16 body inputs and applies
  int8 palettization, while our earlier `scripts/probe_f0_noise_exact_shape.py`
  and `scripts/probe_decoder_vocoder_split.py` declared the same body boundary
  as fp32. Added `--body-input-dtype` and tested the closer F0-source split on
  local 3s. fp16 body inputs made the candidate catastrophically slower:
  baseline `34.37 ms`, candidate `223.14 ms`, with `213.99 ms` inside the body
  and failed parity (corr `0.931413`, SNR `9.17 dB`). Adding body palettization
  still failed: baseline `32.69 ms`, candidate `223.07 ms`, body `214.56 ms`,
  corr `0.930939`, SNR `9.14 dB`. Graph surface confirms the palettized body
  shrank `97.8 MB -> 49.2 MB` and added `101` LUT ops, but did not change the
  runtime failure. Do not attribute laishere's M1/M2 Air advantage to the
  vocoder interface dtype or generic weight palettization in our static 3s
  export.
- **iOS17 target lowering does not rescue the laishere-style body:** Added
  `--deployment-target` to `scripts/probe_f0_noise_exact_shape.py` and reran the
  local 3s F0-source split with `ios17`, fp16 body inputs, body palettization,
  and `CPU_AND_NE` body execution. The candidate got slower: baseline
  `34.34 ms`, candidate `265.89 ms`, body `256.57 ms`, corr `0.930895`, SNR
  `9.13 dB`. This rejects "laishere is faster only because iOS17 lowering
  unlocks a better static body plan" for the current probe.
- **Cos Snake + residual-scale rewrite recovers speed, not quality:** The
  laishere-style math patch (`--cos-snake --patch-resblock-scale`) makes the
  first-party F0-source split speed-positive again with the body on CPU+GPU:
  local 3s padded is `30.63 ms` candidate versus `30.88 ms` baseline, and local
  7s padded is `57.03 ms` candidate versus `61.26 ms` baseline (`+6.9%`). This
  still fails strict quality (`3s` corr `0.931895`, SNR `9.19 dB`; `7s` corr
  `0.962251`, SNR `11.51 dB`). Keep this as the fastest local F0-source
  research branch, but do not integrate it until the source-quality contract is
  fixed or a human listening gate explicitly approves it. A 3s rerun with
  `--include-torch-reference` confirms the remaining miss is not primarily Core
  ML conversion drift: the shipped baseline versus dump is corr `0.999996`, SNR
  `51.60 dB`, while the PyTorch F0-source candidate itself is only corr
  `0.939812`, SNR `9.57 dB` against the dump.
- **F0-source phase-mode variants do not recover quality:** Added
  `--phase-mode {atan2,acos,atan_manual,atan_swift}` to
  `scripts/probe_f0_noise_exact_shape.py` and reran local 3s with the known-fast
  `--cos-snake --patch-resblock-scale` branch. All variants remain
  quality-rejected. Baseline `atan2` is corr `0.931895`, SNR `9.19 dB`;
  `atan_swift` is worse at corr `0.915815`, SNR `7.44 dB`; `atan_manual` is
  corr `0.938613`, SNR `9.47 dB`; `acos` is the best alternate at corr
  `0.949566`, SNR `10.34 dB` but still far below strict parity. This closes
  raw phase branch selection as the missing F0-source quality fix; the remaining
  gap is source formulation and/or human-listening acceptance, not a Core ML
  `atan2` lowering issue.
- **Swift-like source explains the F0-source quality gap:** Extended
  `scripts/probe_f0_source_variants.py` with a Python `swift_like_seeded`
  implementation of Swift `HarmonicSource.swift`: xorshift64 initial phase,
  Box-Muller Gaussian noise, linear interpolation, and Double phase
  accumulation. It matches dumped Swift `har_source` essentially exactly
  (`3s` corr `1.000000`, SNR `138.15 dB`; `7s` corr `1.000000`, SNR
  `139.65 dB`). The laishere/CoreML-friendly deterministic source remains only
  corr `0.93978`/`0.96731` against that boundary. This proves the speed-positive
  F0-source branch is trading away the exact seeded/Double source contract; it
  cannot become strict-parity production without either porting that contract or
  accepting the audio drift by listening review.
- **Exact Swift HAR + laishere body split is quality-safe but slower:** Reran
  `scripts/probe_decoder_vocoder_split.py` with exact dumped `har_padded` plus
  `--cos-snake --patch-resblock-scale`. Local 3s passes strict quality
  (`corr 0.9999908`, SNR `47.76 dB`) but is slower (`36.76 ms` candidate vs
  `32.07 ms` baseline, `-14.6%`). Local 7s also passes quality
  (`corr 0.9999907`, SNR `47.81 dB`) but is slower (`67.72 ms` vs `60.88 ms`,
  `-11.2%`). This rejects the simple production path of keeping the current
  Swift HnSF/HAR contract while splitting only laishere's noise/body/tail
  packages.
- **10s F0-source speed branch matches the earlier pattern:** Captured a fresh
  `10s` Swift tensor dump with `exact_t156`, then reran
  `scripts/probe_f0_noise_exact_shape.py` with `--cos-snake
  --patch-resblock-scale --include-torch-reference`. Local padded shape
  (`asr=400`, `F0=800`) is faster (`79.02 ms` candidate vs `87.55 ms`
  baseline, `+9.7%`) but fails quality (`corr 0.955085`, SNR `10.86 dB`).
  Natural shape (`asr=384`, `F0=768`) is faster again (`76.37 ms` vs
  `86.01 ms`, `+11.2%`) but worse on metrics (`corr 0.866976`, SNR
  `6.55 dB`). Copied the temporary packages to Irvine M1 and reran with
  `--skip-export`; the same speed-positive/quality-negative pattern holds:
  padded `509.02 ms` vs `565.71 ms` baseline (`+10.0%`, corr `0.955223`),
  natural `487.11 ms` vs `563.90 ms` (`+13.6%`, corr `0.867049`). PyTorch
  references are already divergent locally, so this remains source
  formulation/listening acceptance, not Core ML conversion drift. Rendered the
  no-ASR 10s listening pack at
  `outputs/f0_source_listening/10s_speed_branch/README.md`.
- **15s F0-source speed branch strengthens the duration trend:** Captured a
  fresh `15s` Swift tensor dump with `exact_t219` and reran the same cos/residual
  F0-source probe locally. Padded shape (`asr=600`, `F0=1200`) is faster
  (`109.20 ms` candidate vs `129.98 ms` baseline, `+16.0%`) but still fails
  strict quality (`corr 0.956701`, SNR `10.99 dB`). Natural shape
  (`asr=556`, `F0=1112`) is faster again (`101.41 ms` vs `129.88 ms`,
  `+21.9%`) and worse on metrics (`corr 0.838603`, SNR `5.73 dB`). The local
  PyTorch reference is already divergent (`corr 0.949135` padded,
  `0.818057` natural), so this remains the same source-formulation/listening
  gate. Rendered the no-ASR 15s listening pack at
  `outputs/f0_source_listening/15s_speed_branch/README.md`.
- **30s F0-source completes the runtime-bucket sweep:** Captured a fresh `30s`
  Swift tensor dump with `exact_t476` and reran the same cos/residual F0-source
  probe locally. Padded shape (`asr=1200`, `F0=2400`) is faster (`211.43 ms`
  candidate vs `269.86 ms` baseline, `+21.7%`) but still fails strict quality
  (`corr 0.949790`, SNR `10.40 dB`). Natural shape (`asr=1095`, `F0=2190`) is
  faster again (`191.31 ms` vs `268.94 ms`, `+28.9%`) and worse on metrics
  (`corr 0.794801`, SNR `4.78 dB`). The local PyTorch reference is already
  divergent (`corr 0.943165` padded, `0.776711` natural), so the completed
  `3s`/`7s`/`10s`/`15s`/`30s` sweep keeps pointing at source
  formulation/listening acceptance rather than Core ML conversion drift.
  Rendered the no-ASR 30s listening pack at
  `outputs/f0_source_listening/30s_speed_branch/README.md`.
- **Vectorized Swift Gaussian noise is a quality-preserving HnSF win:** Replaced
  the scalar Box-Muller transcendental loop in `HarmonicSource.swift` with
  vectorized `vForce`/`vDSP` math while preserving the same seeded RNG draw
  order. Matched local M2 Studio scalar-control vs vector runs
  (`KOKORO_USE_EXACT_DURATION_MODELS=1`, staged, warmup 2, iterations 5) show
  HnSF medians improve by `17.5-20.6%` and wall time improves by `0.7-9.0%`
  across `3s`/`7s`/`10s`/`15s`/`30s`. The `30s` parity check against the
  pre-change tensor dump passed through `waveform_full`; final waveform corr is
  `0.999997`, and HnSF boundary tensors are corr `1.0`. This is production-safe
  host-DSP cleanup, not a model-architecture escape from the generator
  bottleneck.

**2026-05-17**

- **Powermetrics result:** Config F/reference single-stream was not ANE-bound
  on Studio (`ANE Power: 0 mW`) even when the 3s path stayed fast. Irvine
  single-stream showed only tiny ANE readings while GPU power dominated.
- **Production-shaped result:** Irvine `.all` could stall in
  `ANECompilerService` before emitting workload traces. The same shape completed
  when forced to `.cpuAndGPU`.
- **Compute-plan result:** `kokoro_decoder_pre_3s.mlpackage` is
  NeuralEngine-preferred, but `kokoro_decoder_har_post_3s.mlpackage`
  (`GeneratorFromHar`) is GPU-preferred. The generator contains ANE-hostile
  structure: exported `conv_transpose`, >16k temporal axes in the harmonic
  branch, and Snake activations lowered to `sin` ops.
- **Prototype tried:** Rewriting `conv_transpose1d` as zero-insertion plus
  `conv1d` removed MIL `conv_transpose`, but the compute plan remained
  GPU-preferred. Cropping the harmonic branch before residual convs reduced the
  >16k conv surface but changed output because AdaIN normalizes over the full
  time axis. That path was not shipped.
- **Runtime decision:** The Swift pipeline now uses explicit stage placement:
  duration/F0 on `.cpuAndGPU`, `decoder_pre` on `.cpuAndNeuralEngine`, and
  `GeneratorFromHar` on `.cpuAndGPU`. This is a manual partition in the sense of
  the compute-unit scheduling guide: run the ANE-eligible static conv island on
  ANE, and keep the generator away from the ANE compiler until the model
  architecture changes.
- **Verification after commit `cdc4f86`:** Studio and Irvine both built the
  Swift package. Irvine's staged full workload completed without the earlier
  `.all` `ANECompilerService` stall and powermetrics showed nonzero ANE samples
  (p95 47 mW, max 82 mW). Studio still reported 0 mW ANE even when an isolated
  `kokoro_decoder_pre_3s` loop was loaded with `CPU_AND_NE`; the compute plan
  for that exact model remains 100% NeuralEngine-preferred, so Studio
  powermetrics is non-confirming rather than proof of a different plan.
- **Decoder-pre latency control:** Isolated `kokoro_decoder_pre_3s` was faster
  with `CPU_AND_NE` than `CPU_ONLY`, and `CPU_AND_NE` matched `.all`: Studio
  p50 3.231ms (`CPU_ONLY`) vs 2.476ms (`CPU_AND_NE`) vs 2.469ms (`.all`);
  Irvine p50 5.024ms vs 2.734ms vs 2.741ms. This is the strongest confirmation
  that the staged runtime is preserving the ANE-eligible decoder-pre island
  while deliberately keeping the ANE-hostile generator on CPU+GPU.

**2026-04-17**

- **Hypothesis:** Config F's speedup over Config A may not isolate ANE, because
  F changes host language, orchestration, and the number of Core ML models.
- **Tried:** Added Config G as Swift + Core ML with `.cpuAndGPU`, preserving the
  same Swift pipeline and model packages as Config F.
- **Outcome:** Config G was faster than F on the local Apple M2 24 GB machine.
  The result invalidates an ANE-latency-win claim for this machine, but does not
  yet identify whether `.all` used ANE, mixed a bad plan, or fell back poorly.
