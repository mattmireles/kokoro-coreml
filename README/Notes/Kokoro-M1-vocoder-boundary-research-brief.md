# Kokoro M1 Vocoder Boundary Research Brief

June 6, 2026

> **Scope:** This is a targeted research brief for making the first-party
> Swift+Core ML Kokoro path faster than laishere on lower-end Apple Silicon,
> especially Irvine M1 `3s`. It is intentionally narrower than a generic Core
> ML optimization guide. Do not repeat broad `.all`, palette, fp16, or
> native-InstanceNorm experiments unless the proposed change creates a different
> runtime boundary or a different Core ML partition.

## Current Conclusion

The remaining real loss is not MLX. After warmed-inference correction, Config F
beats or ties MLX on the validated Mac cells. The remaining strict competitor is
laishere on Irvine M1 short and medium buckets. The live frontier is tracked by:

- `outputs/external_bakeoff/goal_frontier_status.md`
- `outputs/external_bakeoff/frontier_freshness.md`
- `outputs/external_bakeoff/irvine_next_targets.md`
- `outputs/external_bakeoff/irvine_3s_placement_target.md`
- `outputs/external_bakeoff/candidate_frontier_matrix.md`
- `outputs/external_bakeoff/remote_host_quiet_latest.md`
- `README/Kokoro-M1-graph-surface-target.md`
- `README/Kokoro-M1-kernel-partition-deep-research-prompt.md`
- [Core ML vs MLX vocoder scheduling (ConvTranspose / iSTFT)](../Guides/apple-silicon/Core%20ML-MLX-Scheduling-1D-ConvTranspose-ISTFTNet-vocoders-guide.md) —
  synthesized field guide: fixed-cost diagnosis, handoff collapse, dual-output
  anchor, and staged recommendations for Irvine M1 short buckets.
- [Apple Silicon warmed-inference benchmark hygiene](../Guides/apple-silicon/Apple-Silicon-warmed-inference-benchmark-hygiene-guide.md) —
  publishable warmed-only methodology and quiet-host gate.
- [Kokoro M1 vocoder runtime boundary guide](../Guides/apple-silicon/Kokoro-M1-vocoder-runtime-boundary-guide.md) —
  runtime boundary strategy distilled from the restarted guide run.
- [Kokoro M1 vocoder partition and boundary guide](../Guides/apple-silicon/Kokoro-M1-vocoder-partition-boundary-guide.md) —
  output-backing, multi-function, and placement claims filtered against current
  Core ML API facts.

For `irvine-m1/3s`, the warmed profile gap is:

| Runtime | Warm median |
| --- | ---: |
| Config F | `233.6 ms` |
| laishere | `195.0 ms` |
| Gap | `38.5 ms` |

The best saved quality-fail source branch still leaves `19.8 ms` against warmed
laishere, and even an optimistic combination of known positive estimates leaves
`6.2 ms`. Existing strict-pass probes do not close the gap.

## What Is Proven

### MLX Is Not the Bottleneck

The apparent MLX win came from polluted comparisons: cold compile/cache work,
padding, `.all` behavior, and HAR overhead mixed into non-equivalent timings.
Paper-facing comparisons must use warmed inference only. Under that rule, the
real Mac-side gap is laishere on Irvine M1, not MLX.

### Compute-Unit Flags Are Not Enough

Core ML placement data for Irvine M1 `3s`:

| Plan | Compute units | Ops | Preferred counts | NE cost |
| --- | --- | ---: | --- | ---: |
| First-party HAR-post | `cpuAndNeuralEngine` | `2207` | `cpu=1038`, `unknown=1169` | `0.0%` |
| First-party HAR-post | `cpuAndGPU` | `2207` | `gpu=1041`, `unknown=1166` | `0.0%` |
| laishere vocoder | `cpuAndNeuralEngine` | `1534` | `cpu=58`, `neuralEngine=597`, `unknown=879` | `47.5%` |
| First-party exact decoder+vocoder body | `cpuAndNeuralEngine` | `1546` | `cpu=64`, `neuralEngine=599`, `unknown=883` | `48.7%` |

The last row is the trap: we can already create a strict first-party body with
laishere-like NE placement, but it is slower. Therefore the target is not "get
partial ANE placement." The target is a runtime-positive boundary and partition.

The graph surface delta is now explicit: first-party `GeneratorFromHar 3s` has
`2207` ops with `88` `reduce_mean` ops and `96` `tile` ops from manual AdaIN
lowering; laishere `KokoroVocoder` has `1534` ops with `42` native
`instance_norm` ops, `0` tiles, and `101` LUT decompression ops. The next strict
surface should eliminate the manual AdaIN/tile footprint without adding the
measured split-boundary synchronization penalty.

### Existing Strict Splits Lose

Saved Irvine M1 `3s` strict-equivalent results:

| Family | Best relevant result | Decision |
| --- | --- | --- |
| HAR input trim | `+0.7 ms` | Too small. |
| Native-IN/broadcast/cos fused surface | `+0.2 ms` or `.all` `-141.2 ms` | Not material; `.all` is harmful. |
| Generator noise split | `-11.5 ms` CPU+GPU, CPU+NE fails quality/speed | Split overhead exceeds graph savings. |
| Generator stage split | `-15.5 ms` | Split overhead exceeds graph savings. |
| Exact decoder+vocoder split | `-24.8 ms` CPU+GPU, `-138.3 ms` CPU+NE | Boundary too broad or sync-heavy. |
| Exact iOS17/native-IN decoder+vocoder split | `-29.3 ms` CPU+GPU, `-116.3 ms` CPU+NE | Matching laishere's visible surface is insufficient. |
| Exact decoder+vocoder split with fp16 body inputs | local `3s` `-10.1 ms` | Halving `x_source_*` transfer made body slower. |
| Style-specialized fused generator | Irvine `3s` `-3.0 ms`; M2 Air `3s` `-2.2 ms` | Freezing `af_heart` AdaIN projections is slower remotely. |
| Native-IN style-specialized fused generator | local `3s` `+0.07 ms`; CPU+NE fails | Noise-sized, not a frontier candidate. |
| Native-IN style-specialized fused generator + HAR trim | local `3s` `+0.06 ms` | Strict but only `0.22%`; do not promote to Irvine. |
| Plain cos-Snake/iOS17 fused generator | local `10s -0.16%`, `15s -0.27%` | Strict but slower on remaining medium buckets. |
| Fused `GeneratorFromHar` fp16 inputs | local `3s -0.07%`; graph still has `88` reductions and `96` tiles | Does not change the actual bad surface. |
| Fused native-IN + fp16 inputs | local `3s +0.12%`; `88 -> 0` reductions, `44` instance_norm, still `96` tiles | Partial surface repair, not material. |
| Fused native-IN + broadcast + fp16 inputs | local `3s +0.08%`; `88 -> 0` reductions, `96 -> 0` tiles, `44` instance_norm | Near-surface match but no material speed win. |
| Fused native-IN + broadcast + fp16 + pal8 | local `3s -2.83%`; `101` LUT ops, no reductions/tiles | Full visible surface match still loses and reduces quality margin. |
| Fused cos-Snake + native-IN + broadcast + fp16 + upsample ConvT rewrite | local `3s +4.45%`, `7s +3.42%`, `10s +3.56%`, `15s +3.52%`, `30s +3.24%` | Strict local win; promote to Irvine when quiet. |
| Generator output-backed prediction | local CPU+GPU `3s -0.077 ms`, `7s +0.414 ms`; waveform dump bit-identical | Harness is valid, but below `1 ms` promotion gate; do not productionize. |
| HAR-source fused strict path | after Swift STFT credit: `+0.051 ms` 3s, `+1.326 ms` 7s, `+2.231 ms` 10s, `+14.977 ms` 30s | Source/STFT boundary is not a net win unless a Core ML call boundary or body cost is also removed. |

Do not spend research budget on another broad split of the current
`GeneratorFromHar` package unless it removes a Core ML call boundary or changes
the actual partitioning/synchronization behavior.

### Existing Speed-Positive Branches Change Quality

The speed-positive candidates are F0/source simplifications:

| Candidate | Irvine `3s` speed signal | Quality |
| --- | ---: | --- |
| `3s_natural_asr_cos_rsqrt` | `+18.7 ms` | corr `0.813995`, SNR `5.08 dB` |
| native-IN no-palette F0/source | `+12.8 ms` | corr `0.931801`, SNR `9.19 dB` |
| iOS17 native-IN no-palette F0/source | `+10.9 ms` | corr `0.931840`, SNR `9.19 dB` |

These branches are useful evidence but not strict parity. They require either a
source formulation recovery or explicit human listening acceptance before they
can support a time-to-parity claim.

## Research Question

Find a first-party Core ML graph boundary for Kokoro's source/body/vocoder path
that satisfies all of the following:

1. Preserves the current Swift HAR/source semantics or produces an accepted
   quality-equivalent waveform.
2. Avoids adding extra Core ML prediction calls on the `3s` hot path unless the
   saved compute is larger than the call/sync overhead on M1.
3. Produces a mixed CPU/Neural Engine plan only when it improves warmed runtime,
   not merely because the compute plan reports NE-preferred ops.
4. Beats warmed laishere on Irvine M1, with priority order `3s`, `7s`, `10s`,
   `15s`, then `30s`.

The shortest useful formulation is:

> Design an M1 MLProgram source/STFT/vocoder body that preserves current Swift
> HAR/source semantics, keeps strict waveform parity, and shifts the expensive
> conv/add/mul/instance_norm body work into a laishere-like mixed CPU/NE plan
> without the existing split-boundary synchronization penalty or a `3s` warmed
> regression.

For the next implementation pass, use the constrained handoff prompt in
`README/Kokoro-M1-source-body-deep-research-prompt.md`.
For a narrower external-research pass on the current M1 partition paradox, use
`README/Kokoro-M1-kernel-partition-deep-research-prompt.md`.
The numeric post-rewrite target budget is generated at
`outputs/external_bakeoff/strict_win_budget_after_overlap_rewrite.md`; update it
after any new strict candidate before changing the research prompt. The candidate
frontier matrix at `outputs/external_bakeoff/candidate_frontier_matrix.md`
summarizes the current strict wins, strict rejections, quality-fail speed
branches, and iPhone launch gate; update it whenever a candidate changes status.
Before any lower-end Mac promotion run, regenerate
`outputs/external_bakeoff/remote_host_quiet_latest.md` with
`scripts/external_bakeoff/check_remote_host_quiet.py` and run timing only when
the target host reports `quiet=yes`.

## High-Value Directions

### 1. Single-Package Body Reshaping

The most promising strict direction is not another multi-package split. It is a
single package that changes the operator surface inside the existing
`GeneratorFromHar` call boundary while preserving the same runtime inputs and
outputs.

Research targets:

- Can native InstanceNorm, broadcast AdaIN, cos Snake, HAR trim, and residual
  scale changes be fused into one `GeneratorFromHar` package without changing
  the externally visible boundary?
- Can the zero-insert plus `conv1d` upsample rewrite replace main
  `ConvTranspose1d` layers in the production exporter and preserve the current
  local `3.2-4.5%` strict win on Irvine M1? The production exporter now has an
  opt-in `--rewrite-ups-conv-transpose` flag, and the all-bucket production
  package smoke improves warmed local M2 Studio CPU+GPU medians by `+4.28%`
  `3s`, `+3.15%` `7s`, `+3.17%` `10s`, `+2.60%` `15s`, and `+2.20%`
  `30s` against the shipped packages. Projection says this is not enough alone
  for Irvine M1, but it likely combines well with any independent source/body
  or upstream/runtime win.
- Can the graph be rewritten so M1 chooses useful NE partitions without
  crossing CPU/NE/GPU boundaries repeatedly?
- Can layout be changed to reduce ANE padding or memory movement while keeping
  the Swift input contract stable?

Acceptance:

- Strict waveform gate passes against the shipped fused package.
- Irvine M1 `3s` improves by at least `20 ms` warmed median, or a smaller `3s`
  win combines with measured upstream/runtime savings that fully close the gap.
- M2 Studio does not regress materially.

### 2. In-Package HAR Source Consumption

Current strict source/STFT split paths lose because they add package boundaries
or recompute source/STFT with drift. A useful path may be to consume a smaller
Swift-prepared tensor inside the same package boundary rather than exporting a
standalone source/noise package.

Research targets:

- Identify the minimal Swift-produced tensor that preserves source quality but
  avoids the large HAR-post input cost.
- Determine whether passing `x_source_0/x_source_1` directly into a single
  body+tail package can be faster than fused `GeneratorFromHar` once call
  overhead is eliminated or amortized.
- Validate whether the tail can stay fused with the body without causing ANE
  fallback or CPU sync.

Acceptance:

- No extra Core ML call on the hot path unless timing proves it is profitable.
- Strict parity or listening-approved quality.
- Same bucket contract: `3s`, `7s`, `10s`, `15s`, `30s`.

### 3. Source-Quality Recovery for the Fast F0 Branch

The non-strict branch is the only saved branch with enough speed signal to beat
laishere for `7s`, `10s`, and `15s`, and nearly enough for `3s` after upstream
savings. It is worth researching, but it must be labeled separately from strict
parity.

Research targets:

- Explain why the deterministic/laishere-style source changes waveform
  character relative to the seeded Swift Double-accumulator HnSF source.
- Find a Core ML-friendly source formulation that improves correlation/SNR
  without losing the speed signal.
- Treat the sine-source equation as mostly solved: the five-bucket
  `swift_like_seeded` probe matches dumped `har_source` at SNR `138.15-140.33 dB`.
  The unsolved part is HAR/STFT recreation from source, which remains near SNR
  `8.11-8.23 dB` even when recomputing from the exact dumped source.
- Treat Nyquist phase as a proven sub-blocker, not the full fix. With the
  corrected `waveform_raw_trimmed` reference, dumped Nyquist phase plus padded
  shipping HAR geometry repairs strict source-boundary parity across
  `3s/7s/10s/15s/30s`, but natural geometry still fails and prior Core ML timing
  shows the padded strict path loses the speed edge.
- If objective parity remains impossible, define a listening-review protocol
  that can support a "quality-equivalent" paper claim without Whisper/ASR.

Acceptance:

- Either strict waveform gate passes, or human listening decisions explicitly
  accept the exact speed-branch WAVs under
  `outputs/f0_source_listening/irvine_exact_speed_branch/`.

## Low-Value Directions

Do not prioritize these unless new evidence changes the premises:

- More `.all` toggles. `.all` is a request, not proof, and already produced
  severe slowdowns on strict partial-NE candidates.
- Palette-only changes. They reduce package size but did not deliver the needed
  runtime win.
- Linear quantization of the final-waveform fused generator. Int8 crashed on
  CPU+GPU runtime specialization; int4 requires iOS18 and was slower while
  failing quality locally.
- fp16-input-only changes. Tested and rejected for the current static body.
- iOS17/spec8-only changes. Helpful for metadata matching, not sufficient.
- More exact decoder+vocoder multi-package splits. The matching mixed CPU/NE
  body exists and still loses.
- fp16 body inputs for the exact decoder+vocoder split. Local strict quality
  passed, but candidate total regressed from `30.242 ms` to `40.316 ms`.
- Style specialization for the fixed `af_heart` voice. Plain style-specialized,
  native-IN style-specialized, and native-IN style-specialized plus HAR trim are
  all slower or noise-sized and do not justify Irvine timing.
- Extending the plain cos-Snake/iOS17 fused generator to medium buckets. Local
  `10s` and `15s` are strict but slower, so do not spend Irvine timing on this
  surface by itself.
- Fused `GeneratorFromHar` fp16 inputs. The direct single-package probe was
  strict but slower locally and left the manual AdaIN `reduce_mean`/`tile`
  footprint intact.
- Fused native-IN plus fp16 inputs. The direct single-package probe removes
  reductions but leaves the `96` tiles and only improves local `3s` by `0.12%`.
- Fused native-IN plus broadcast AdaIN plus fp16 inputs. The direct
  single-package probe removes both reductions and tiles, but only improves
  local `3s` by `0.08%`; surface similarity alone is not enough.
- Fused native-IN plus broadcast AdaIN plus fp16 inputs plus 8-bit
  palettization. It adds the laishere-like LUT surface but regresses local `3s`
  by `2.83%` and thins quality margin.
- More graph-surface-only chasing. `MLComputePlan` shows the
  native-IN+broadcast+fp16 fused package has `1533` ops, but still receives
  `0` Neural-Engine-preferred ops and `0` NE cost weight under CPU+NE. Laishere's
  `1534`-op vocoder gets `597` NE-preferred ops and about `0.56` NE cost
  weight. The missing win is placement/layout/boundary, not just surface count.
- Cos-Snake as a placement unlock. The combined cos-Snake + native-IN +
  broadcast-AdaIN + fp16-input package removes the original `sin`/`pow`
  footprint and is strict with a small local CPU+GPU win, but CPU+NE still gives
  it `0` Neural-Engine-preferred ops and `0` NE cost weight.
- More broad generator noise/stage splits. The extra Core ML call boundary is
  currently more expensive than the saved graph work.

## Required Measurements

Every candidate must report:

- exact git SHA and command line;
- model paths and whether packages were freshly exported or reused;
- deployment target, precision, input dtypes, compute units;
- warmup count and warm iteration count;
- cold latency separately from warm median;
- waveform metrics against the same tensor dump or emitted reference;
- `MLComputePlan` preferred-device counts and cost weights;
- local M2 Studio timing before remote Irvine timing;
- Irvine M1 timing only when the machine is quiet enough for publishable data.

Do not promote a candidate from M2 Studio alone. M2 Studio is useful as a fast
rejection filter, but the remaining loss is on Irvine M1.

## Current External-State Blocks

- The iPhone 12 Pro is visible and paired, but app launch fails while the device
  is physically locked. Unlock the phone before iOS runner execution.
- Irvine M1 timing is not publishable while background indexing or
  `mediaanalysisd` consumes CPU. Use saved artifacts until the host is quiet.

## Useful Commands

Regenerate the target summaries:

```bash
uv run --no-sync python scripts/external_bakeoff/summarize_competitive_frontier.py \
  --output outputs/external_bakeoff/competitive_frontier.md \
  --json-output outputs/external_bakeoff/competitive_frontier.json

uv run --no-sync python scripts/external_bakeoff/summarize_frontier_freshness.py \
  --output outputs/external_bakeoff/frontier_freshness.md \
  --json-output outputs/external_bakeoff/frontier_freshness.json

uv run --no-sync python scripts/external_bakeoff/summarize_irvine_3s_placement_target.py
```

Run the reusable gates:

```bash
uv run --no-sync pytest tests/test_external_bakeoff_tools.py -q
uv run --no-sync python scripts/external_bakeoff/verify_external_bakeoff_completion.py
```

Check iPhone state:

```bash
xcrun devicectl list devices
xcrun devicectl device process launch \
  --device F383FC46-FD64-5346-AEC6-59E3E2F8C9CA \
  --console --terminate-existing \
  com.kokoro.externalbakeoff.ConfigFIOSRunnerManual
```

Check Irvine load before remote timing:

```bash
ssh mattmireles@irvine-m1.local \
  'uptime; ps -axo pcpu,pid,comm | sort -nr | head -12'
```
