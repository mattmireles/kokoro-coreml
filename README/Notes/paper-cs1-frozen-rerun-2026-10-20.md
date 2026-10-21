# Paper CS1 Frozen Rerun (2026-10-20)

Date: 2026-10-20 (PT)
Plan: [`013-paper-cs1-frozen-rerun-plan.md`](../Plans/013-paper-cs1-frozen-rerun-plan.md)
Source: `d32ef0d3` on every Mac. Models: HF `mattmireles/kokoro-coreml` at `dadebca4`.
Raw data: `outputs/frozen_rerun/<host>/` (gitignored; `summary.json` and
`summarize.py` sit beside them).

## Why

The Surgical Inference paper's Kokoro numbers come from builds made before PRs #6–#12
(April ledger, June bakeoff, July iPhone runs). This run re-measures the same
matrix on the same machines with one protocol, so the paper can replace its tables.

## Protocol as run

- Config F: `run_config_f_reference.py --compute-units staged --use-padded-duration-models --iterations 10 --preflight-runs 3`.
  Padded masked duration is the shipped default. Exact-duration packages are not
  published at `dadebca4`.
- **Generator: bucketed on all three Macs.** The flexible generator aborts the
  process on macOS 27.0.1 (see below), so `--models-dir coreml-bucketed` (every
  package except `kokoro_decoder_har_post_range`) was used everywhere. m2-studio
  also ran the flexible generator as a separate row. Owner decision, 2026-10-20.
- MLX 0.4.3 `862dfbe`, soniqo `0d09a2e`, laishere `484907d`: `--iterations 10`,
  no preflight (the adapters have none).
- A/D/E: `bakeoff_harness.py run --iterations 5 --order-seed 0`; irvine-m1 split
  into `a` and `d,e` for 16 GB.
- Botnet paused on m2-air and irvine-m1 (m2-studio was already paused). Restored
  afterwards: m2-air and irvine-m1 back to `adaptive`; m2-studio left `paused` as found.

## Quiet-host status

| Host | OS | Gate (1-min load ≤ 1.5) | Load during arms |
| --- | --- | --- | --- |
| m2-studio (M2 Ultra) | macOS 26.7.1 (25G241) | **failed every arm**; owner's other sessions held load | 6–13 |
| m2-air (M2) | macOS 27.0.1 (26A434) | failed every arm after a 20 min wait | 2.2–2.9 |
| irvine-m1 (M1) | macOS 27.0.1 (26A434) | failed Config F (2.29), passed MLX, soniqo, A, D/E | 1.4–2.3 |

The m2-studio rows are upper bounds. A second m2-studio pass (`m2-studio-quiet`)
waits up to 6 h per arm for the gate.

## Config F, bucketed generator: warm median ms (IQR), n=10

| Input | M2 Ultra (loaded) | M2 Air | M1 Mini |
| --- | ---: | ---: | ---: |
| 3s | 64.1 (3.0) | 123.4 (0.8) | 205.3 (1.9) |
| 7s | 116.4 (2.9) | 273.6 (3.3) | 410.3 (2.4) |
| 10s | 164.9 (8.9) | 397.2 (0.7) | 589.8 (2.4) |
| 15s | 207.6 (10.4) | 589.1 (7.2) | 834.9 (9.8) |
| 30s | 373.4 (17.1) | 1175.5 (9.4) | 1635.5 (10.2) |

M2 Ultra with the flexible generator (loaded): 67.7 / 121.7 / 156.8 / 192.4 / 349.1 ms.
It is 5% slower at 3 s and 7 s, and 5–7% faster at 10–30 s.

Against the June table, the Air is 15–17% faster at every bucket and the Mini is
12–17% faster. This comes from a new OS and a new pipeline together, so the two
effects cannot be separated.

## Ratios (competitor ÷ Config F bucketed)

| Competitor | M2 Ultra (loaded) | M2 Air | M1 Mini |
| --- | --- | --- | --- |
| MLX (7–30 s; 3 s errors) | 1.58–2.61× | 1.82–2.07× | 1.84–1.94× |
| laishere (chain only) | 1.26–1.40× | could not convert | could not convert |
| soniqo, 3 s only | 1.63× | 1.96× | 1.79× |
| A (Python HAR hybrid) | 1.97–2.38× | 1.60–2.88× | 1.26–1.37× |
| D (MPS), 3 s and 7 s only | 3.61–4.32× | 2.97–6.45× (n=2) | 2.69–4.42× |
| E (CPU) | 5.03–5.67× | 4.99–5.34× | 3.18–3.98× |

## Failures and caveats

- **Flexible generator on macOS 27.0.1:** an MPSGraph abort
  (`'mps.expand_dims' op invalid axis`, then `shape for TensorData is not static`)
  on M1 and M2. laishere's converter hits the same assertion and cannot finish on
  either 27.0.1 host. Fixed for our runtime by gating the flexible generator off
  on OS 27 (`523cac42`, branch `claude/eloquent-tereshkova-16568e`, not merged).
  iOS 27.0.1 runs it 6/6.
- MLX still fails every 3 s input with `[broadcast_shapes] (1,67200,1) vs (1,67500,9)`,
  as in June.
- soniqo still emits 5.0 s of audio for every input of 7 s and over (2.7 s for
  the 2.8 s input). Only its 3 s cell is comparable.
- D (MPS) now **raises** at 15 s and 30 s on all three Macs, with
  `Output channels > 65536 not supported at the MPS device` (torch 2.5.0). In
  April it ran out of memory on the Air only. On m2-air, 3 s and 7 s also raised
  on 3 of 5 iterations.
- The harness A/D/E arms have no 10 s input, as in April.
- Per-stage times are the last warm call per input (`raw_last_result`), not medians.

## Per-stage, last warm call (ms)

Generator = Core ML total − duration − F0Ntrain − DecoderPre.

| Host | Input | Duration | F0Ntrain | DecoderPre | hn-nsf | Generator | Wall |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M2 Ultra (loaded) | 3s | 16.6 | 7.3 | 3.7 | 3.7 | 8.1 | 63.9 |
| M2 Ultra (loaded) | 30s | 74.5 | 25.9 | 51.9 | 41.1 | 79.2 | 389.0 |
| M2 Air | 3s | 12.8 | 4.7 | 4.0 | 5.4 | 79.4 | 124.3 |
| M2 Air | 30s | 70.3 | 27.3 | 32.7 | 48.7 | 888.5 | 1168.4 |
| M1 Mini | 3s | 40.0 | 10.5 | 5.2 | 9.2 | 86.9 | 204.0 |
| M1 Mini | 30s | 174.3 | 51.0 | 45.7 | 48.3 | 1083.0 | 1639.1 |

The paper calls the pipeline "well-balanced". That holds on the M2 Ultra, but not
on the base M2 and M1 with the bucketed generator, where the GPU generator is
64–76% of wall time.

## Quiet M2 Ultra pass (2026-10-21, 01:34–02:11 PT)

`m2-studio-quiet` re-ran the m2-studio matrix overnight. Every arm passed the
1.5 load gate (1.20–1.46 at start). **The paper's M2 Ultra rows use this pass;
the loaded rows above are superseded.** soniqo failed in this pass (its CLI's
last stdout line was not JSON), so the paper keeps the loaded-session soniqo 3 s
ratio and labels it.

| Input | F bucketed | F flexible | MLX | laishere | A | D | E |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3s | 44.5 (0.6) | 44.7 | error | 87.5 | 123.7 | 192.6 | 258.7 |
| 7s | 80.0 (0.9) | 79.6 | 183.8 | 162.8 | 222.1 | 375.6 | 529.7 |
| 10s | 111.8 (1.2) | 109.9 | 242.7 | 211.1 | — | — | — |
| 15s | 147.6 (1.5) | 141.1 | 329.7 | 294.9 | 386.8 | err | 1009.7 |
| 30s | 286.5 (3.3) | 271.8 | 685.3 | 532.4 | 630.7 | err | 1921.3 |

- Ratios: MLX 2.2–2.4×, laishere 1.9–2.0×, A 2.2–2.8×, D 4.3–4.7×, E 5.8–6.8×.
- Against June (exact-duration packages): 1.13–1.32× faster at every bucket
  (30 s: 379.3 → 286.5 ms).
- Per-stage, last warm call: 3 s duration 11.4, F0Ntrain 5.5, DecoderPre 4.5,
  hn-nsf 4.3, generator 2.2, wall 45.5; 30 s 44.9 / 19.4 / 51.9 / 40.0 / 52.1 /
  288.7 ms.
- Across three Macs the paper now states MLX 1.8–2.4×, MPS 2.7–6.5×, CPU
  3.2–6.8×.

## Cleanup (2026-10-21)

Local bench state deleted after the runs: `coreml/`, `coreml-bucketed/`, `.venv`,
`~/kokoro-external-bakeoff` (MLX venv, laishere, soniqo, iOS DerivedData),
`ios-bench/Resources/`, and the three Kokoro HF caches the run downloaded. Raw
results stay in `outputs/frozen_rerun/` (38 MB). m2-air and irvine-m1 keep their
setups for the later flexible-generator rerun.

## iPhones (iOS 27.0.1, Release, 21:15–23:37 PT)

`ios-bench` was patched to run the shipped iOS default: staged policy, padded
duration, and the flexible generator (`--bucket-generator 1` falls back to the
fixed generators). Each launch ran one arm on one bucket: 3 preflight calls, 1
cold call, 10 warm calls. The phone cooled at least 10 minutes before every
launch, and arm order alternated by bucket. Raw data:
`outputs/frozen_rerun/iphone-*`; summarizer: `summarize_iphone.py`.

| Input | A17 Pro F | A17 Pro MLX | ratio | A14 F | A14 MLX | ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3s | 195 | 543 | 2.8× | 421 | 926 | 2.2× |
| 7s | 417 | 1198 | 2.9× | 917 | 2050 | 2.2× |
| 10s | 609 | 1634 | 2.7× | 1391 | 2809 | 2.0× |
| 15s | 881 | 2802 † | 3.2× | 1984 | 4062 | 2.0× |
| 30s | 1721 | 4508 | 2.6× | 4017 | OOM (signal 9) | — |

† Both MLX 15 s runs on the A17 Pro started at thermal state `fair` and ended
`serious`; the better one is shown, and the ratio is an upper bound.

- Against July (iOS 26.5, fixed generator), Config F is 1.8–2.2× faster on both
  phones (A17 Pro 30 s: 3742 → 1721 ms). The OS changed too.
- `.all` now runs on iOS 27.0.1 (3 s only: 200 ms on the A17 Pro, 412 ms on the
  A14). In July it failed with Core ML error −9.
- The iPhone 12 Pro hit `CoreDeviceError 10002 / POSIX 22` on every launch from
  20:30 to 21:00 PT, probably because Auto-Lock let the screen sleep. The driver
  retries launches now. One MLX 15 s attempt hung after 11 of 14 calls and was
  rerun.
