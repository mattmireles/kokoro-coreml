# HAR/STFT Phase Contract Bisection

Collected: 2026-06-06.

This note records the current strict source-boundary bisection for the Kokoro
HAR/STFT path. It exists to stop repeated probes that treat phase as a circular
quantity when the trained generator consumes raw phase values.

## Findings

The standalone Core ML forward-STFT subgraph is not the bug:

- `coreml_vs_torch_magnitude`: SNR `156.10 dB`, max abs `0`.
- `coreml_vs_torch_phase`: SNR `152.00 dB`, max abs `2.38e-7`.
- `torch_magnitude_vs_swift_dump`: SNR `128.13 dB`.

The mismatch is raw phase branch convention versus the dumped Swift HAR phase:

- `atan_manual` fp32: raw phase SNR `8.12 dB`, max abs `2*pi`, but wrapped phase
  max error only `0.08724`.
- Branch errors are isolated to the Nyquist bin (`bin 10`): `2331` branch
  mismatches on the 3s dump, with `1898` at `-2*pi` and `433` at `+2*pi`.
- All branch-error samples have effectively zero imaginary component and
  negative real component; a simple `real < 0 && abs(imag) < threshold` rule
  catches them but also catches `4901` false positives.
- `atan_swift` fp32 is worse, not better: raw phase SNR `-0.89 dB`, max abs
  `2*pi`, wrapped max error `1.47`.

The generator is sensitive to the raw phase branch. Although the STFT is
mathematically equivalent modulo `2*pi`, the first noise convolutions do not
consume phase modulo `2*pi`. Therefore wrapped-phase equivalence is not a
strict waveform-parity proof for this trained generator.

## Strict Source-Boundary Candidates

| Candidate | Quality | Speed signal | Decision |
| --- | --- | --- | --- |
| natural `har_source -> fused generator` | quality fail, about `16-17 dB` SNR | speed-positive vs generator-only | reject for strict paper claim |
| `har_source + dumped Nyquist + padded HAR` fused generator | replacement quality good versus current generator: SNR `48-50 dB` | after crediting removed Swift STFT, still no net win: `+0.051 ms` 3s, `+1.326 ms` 7s, `+2.231 ms` 10s, `+14.977 ms` 30s | reject as direct replacement |
| `har_source + dumped Nyquist + padded HAR` source/noise split | quality good: corr `0.999986975`, SNR `46.25 dB` | slower than decoder-pre+generator: `34.4 ms` vs `30.4 ms` 3s, `-13.0%` | reject |
| oracle-fitted affine Nyquist repair | padded waveform SNR only `26.46/27.69/27.64/27.04/26.36 dB` for `3s/7s/10s/15s/30s` versus `50.06/49.14/49.87/49.21/48.42 dB` with dumped Nyquist | PyTorch-only sensitivity probe; no production timing because quality fails | reject scalar/affine calibration |
| exact Swift Float Nyquist `atan2` repair | matches dumped-Nyquist oracle on padded geometry: SNR `50.06/49.14/49.87/49.21/48.42 dB` for `3s/7s/10s/15s/30s`; branch-only Swift basis fails at `25-27 dB` | same direct speed envelope as dumped-Nyquist padded path; no standalone net win after Swift STFT credit | keep as strict source-contract unlock, not current production replacement |
| exact Swift Nyquist plus shorter padding | first strict points: `3s` HAR `28561` and `7s` HAR `66601`, saving only `0.83%` and `0.89%` of full padded HAR frames | too small to close lower-end Mac losses; 3s matches prior `har28561` trim neighborhood | reject as standalone tail-trim path |
| embedded exact Swift Nyquist inside fused Core ML graph | first attempt failed because Core ML collapsed DC phase channel `11` to `0` where Swift/PyTorch require `real < 0 ? pi : 0`; explicit DC branch plus exact Swift Nyquist repairs Core ML HAR to `147.21 dB` vs PyTorch | corrected no-side-input fused package is strict but slower: 3s `27.53 ms` vs `26.98 ms` (`-2.05%`), 7s `58.36 ms` vs `56.40 ms` (`-3.47%`) | keep correctness fix; reject as current speed win |
| embedded exact Swift Nyquist plus upsample rewrite, fp16 | local M2 Studio `3s` candidate-vs-dump SNR only `33.00 dB` | slightly faster than production rewrite: `28.07 ms` vs `28.27 ms` (`+0.73%`) | reject: speed-positive but not strict |
| embedded exact Swift Nyquist plus upsample rewrite, fp32 | local M2 Studio `3s` candidate-vs-baseline SNR `49.87 dB`, candidate-vs-dump SNR `47.75 dB` | slower than production rewrite: `28.70 ms` vs `27.79 ms` (`-3.25%`) | reject: strict but slower |

## Net Source-Boundary Timing

`kokoro-hnsf-bench` measures the shipping Swift HnSF source/STFT boundary for
all five runtime buckets. The strict fused `har_source` candidate removes only
Swift STFT. Swift source generation still remains required, so it cancels out
of the net replacement delta.

| Bucket | Swift source still required | Swift STFT removable | Strict fused generator delta | Net after STFT credit |
| --- | ---: | ---: | ---: | ---: |
| `3s` | `5.059 ms` | `0.518 ms` | `+0.569 ms` | `+0.051 ms` |
| `7s` | `11.802 ms` | `1.293 ms` | `+2.619 ms` | `+1.326 ms` |
| `10s` | `16.759 ms` | `1.738 ms` | `+3.969 ms` | `+2.231 ms` |
| `15s` | `24.572 ms` | `2.495 ms` | n/a | n/a |
| `30s` | `51.629 ms` | `5.001 ms` | `+19.979 ms` | `+14.977 ms` |

Generated artifacts:

- `outputs/external_bakeoff/hnsf_source_stft_timing_local.json`
- `outputs/external_bakeoff/hnsf_source_boundary_net.md`
- `outputs/external_bakeoff/nyquist_formula_candidate_probe.json`
- `outputs/nyquist_phase_contribution/summary.md`
- `scripts/external_bakeoff/summarize_hnsf_source_boundary.py`

## Decision

- Do not spend more time on `atan2`, `atan_manual`, or `atan_swift` branch
  variants as direct strict replacements.
- Do not promote padded/Nyquist source-boundary packages unless a future change
  removes a Core ML prediction boundary or materially reduces the generator
  body cost.
- Do not promote the branch-free Nyquist formula
  `(pi / 2) * ((mag - real) / (mag + 1e-7))`. It is phase-equivalent modulo
  wrap, but raw-branch wrong: `3s/7s/10s` produced `2548/5323/7687` raw branch
  errors and natural-geometry waveform SNR only `16.21/15.41/15.06 dB`.
- Do not promote scalar, negated, or affine Nyquist calibration. Even
  oracle-fitted affine repair on padded geometry remains around `26-28 dB`
  SNR, far below the `48-50 dB` dumped-Nyquist repair.
- The exact deployable Nyquist formula is Swift Float real/imag dot products
  followed by `atan2`, not branch-only `+pi/-pi`. The branch-only version has
  zero `2*pi` branch errors but still fails waveform strictness because the
  generator is sensitive to the continuous residual phase offset near Nyquist.
- The remaining researchable path is graph/runtime restructuring around that
  solved contract: phase reparameterization, weight folding, a no-extra-boundary
  Nyquist formula inside the graph, or a cheaper strict source representation.
- Do not repeat exact-Nyquist tail-padding sweeps as a standalone speed path.
  The first strict `3s`/`7s` points save less than 1% of HAR frames versus full
  padded geometry.
- The no-side-input fused Core ML graph needs two explicit phase repairs:
  DC phase must be `real < 0 ? pi : 0`, and Nyquist phase must use exact Swift
  Float real/imag dot products followed by `atan2`. This makes the MLProgram
  strict, but it is still slower than the current HAR-post generator on local
  3s/7s warmed CPU+GPU, so do not promote it as a speed win.
- Combining the no-side-input phase repair with the HAR-post upsample rewrite
  does not change that decision. The fp16 combined graph is fast but quality
  failing; the fp32 combined graph is strict but slower than the production
  rewrite.

The first ingested external research report is summarized in
[Kokoro M1 HAR/STFT contract repair guide](../Guides/apple-silicon/Kokoro-M1-HAR-STFT-contract-repair-guide.md).

## Repro Commands

Standalone STFT branch check:

```bash
uv run --no-sync python scripts/probe_coreml_stft_semantics.py \
  outputs/generator_isolation/dumps/3s \
  --label 3s_atan_swift_fp32 \
  --precision fp32 \
  --phase-mode atan_swift \
  --compute-units cpu_only
```

Strict padded/Nyquist source-noise split:

```bash
uv run --no-sync python scripts/probe_har_source_noise_split.py \
  outputs/generator_isolation/dumps/3s \
  --label 3s_atan_manual_fp32_nyquist_padded \
  --report-name report_har_source_noise_nyquist_padded.json \
  --phase-mode atan_manual \
  --noise-precision fp32 \
  --body-precision fp16 \
  --tail-precision fp32 \
  --nyquist-input \
  --pad-har-to 28801 \
  --decoder-pre-package coreml/kokoro_decoder_pre_3s.mlpackage \
  --fused-package coreml/kokoro_decoder_har_post_3s.mlpackage \
  --decoder-pre-compute-units cpuAndNeuralEngine \
  --fused-compute-units cpuAndGPU \
  --noise-compute-units cpuAndGPU \
  --body-compute-units cpuAndGPU \
  --tail-compute-units cpuAndGPU \
  --warmup 2 \
  --iterations 7
```

Net HnSF source-boundary timing:

```bash
cd swift
swift run -c release kokoro-hnsf-bench \
  --warmup 5 \
  --iterations 30 \
  --output ../outputs/external_bakeoff/hnsf_source_stft_timing_local.json
cd ..
python3 scripts/external_bakeoff/summarize_hnsf_source_boundary.py
```
