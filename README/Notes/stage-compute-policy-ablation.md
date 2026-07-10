# Stage Compute Policy Ablation

Collected: 2026-06-06.

This note records the local per-stage compute-unit sweep added to
`kokoro-bench`. The goal was to see whether non-generator stages can recover
part of the lower-end Mac gap without changing the generator graph.

## Harness

`kokoro-bench` now accepts per-stage overrides:

```bash
--duration-compute-units cpuAndNeuralEngine
--f0n-compute-units cpuAndNeuralEngine
--decoder-pre-compute-units cpuAndGPU
--generator-compute-units cpuAndGPU
```

`scripts/external_bakeoff/run_config_f_reference.py` passes those flags through
to the persistent Swift batch subprocess and records them in provenance.

## Local Sweep

Machine: local M2 Studio. Inputs: `3s`, `7s`, `10s`. Warmed inference only:
`--preflight-runs 2 --iterations 5`.

| Policy | 3s total delta | 7s total delta | 10s total delta | Decision |
| --- | ---: | ---: | ---: | --- |
| baseline staged | `0.000 ms` | `0.000 ms` | `0.000 ms` | control |
| duration on CPU+ANE | `+2.158 ms` | `-0.809 ms` | `+2.830 ms` | reject: inconsistent/noise-sized |
| F0Ntrain on CPU+ANE | `+0.410 ms` | `+11.280 ms` | `+16.135 ms` | reject |
| decoder-pre on CPU+GPU | `+0.021 ms` | `+2.560 ms` | `+4.372 ms` | reject |
| duration+F0Ntrain on CPU+ANE | `-1.858 ms` | `+11.241 ms` | `+18.366 ms` | reject for 7s/10s; verify 3s |

The only positive-looking point was `3s` with duration and F0Ntrain forced to
CPU+ANE, so it received a longer same-machine confirmation:
`--preflight-runs 5 --iterations 20`.

| Policy | 3s total median | 3s delta | Duration | F0Ntrain | DecoderPre | Generator |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline staged | `48.705 ms` | `0.000 ms` | `9.527 ms` | `4.837 ms` | `5.180 ms` | `27.370 ms` |
| duration+F0Ntrain on CPU+ANE | `48.492 ms` | `-0.213 ms` | `9.389 ms` | `4.325 ms` | `3.869 ms` | `27.285 ms` |

The N=20 speed signal collapsed to `0.213 ms`, which is below the threshold for
a meaningful optimization.

## Quality Gate

The N=20 spotcheck WAVs do not hash-match. Comparing the CPU+ANE prefix policy
against the staged baseline WAV:

| Comparison | Correlation | SNR | Max abs |
| --- | ---: | ---: | ---: |
| raw | `0.691758` | `2.38 dB` | `1.06162` |
| lagged best | `0.710032` | `2.65 dB` | `1.10016` |
| lag+affine best | `0.710032` | `3.49 dB` | `0.93290` |

This is not strict waveform parity. Duration/F0Ntrain compute-unit changes can
materially change the final audio, so prefix compute-unit overrides are not a
valid paper-strict optimization without a separate quality recovery.

## Decision

- Keep the per-stage compute-unit override harness for diagnostics.
- Do not promote duration/F0Ntrain CPU+ANE as a production candidate.
- Do not run noisy lower-end timing for this path; the local quality gate
  already rejects it.
