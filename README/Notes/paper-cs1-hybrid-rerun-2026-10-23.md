# Paper CS1 Rerun on the ANE Pipeline (2026-10-23)

Date: 2026-10-23 (PT)
Plan: [`014-paper-cs1-hybrid-frozen-rerun-plan.md`](../Plans/014-paper-cs1-hybrid-frozen-rerun-plan.md)
Source: `903f1fc1`. Packages: the frozen set in
`paper/data/frozen_rerun_014/manifest.json` (SHA-256 per package; uploaded
unchanged to HF `mattmireles/kokoro-coreml` at `e65729b9` by the paper session).
Data: `paper/data/frozen_rerun_014/` (per-call CSV, summary JSON, host state,
quality, placement, RAM, GPU share, harness scripts). Raw logs and WAVs:
`outputs/frozen_rerun_014/` (gitignored).

## Why

Plan 013 measured the October pipeline (bucketed GPU generator). Five changes
landed after it: split duration and F0Ntrain, ANE-layout DecoderPre, the folded
generator, the hybrid generator, and host-code work (parallel hn-NSF, vectorized
sine, parallel tails). This run re-measures the paper's matrix on them, with an
ablation ladder, placement proof and a quality gate.

## Result in one line

The hybrid generator is the shipped default under the owner's rule: faster than
folded at every bucket of 7 s and over on all five devices, with no ties, no
crashes and equal quality. At 30 s it is 8.0x faster than the October pipeline
on the A14 (480 vs 3,829 ms) and 3.1x on the M2 Air (377 vs 1,163 ms, loaded).

## Headline: shipped default (hybrid; folded below 7 s) vs MLX, warm median ms

| Device | Arm | 3s | 7s | 10s | 15s | 30s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| M2 Ultra | Core ML (hybrid default) | 73.6 | 53.7 | 70.6 | 88.9 | 158.9 |
| M2 Ultra | MLX | error | 189.1 | 272.5 | 340.4 | 769.8 |
| M2 Air | Core ML (hybrid default) | 73.0 | 71.9 | 240.1 | 272.9 | 377.4 |
| M2 Air | MLX | error | 1431.6 | 1817.5 | 2426.3 | 5186.7 |
| M1 Mini | Core ML (hybrid default) | 94.9 | 98.9 | 123.8 | 168.5 | 308.5 |
| M1 Mini | MLX | error | 800.5 | 1102.9 | 1542.3 | 3014.6 |
| A17 Pro | Core ML (hybrid default) | 61.8 | 103.2 | 155.9 | 223.8 | 318.5 |
| A17 Pro | MLX | 551.0 | 1226.8 | 1717.1 | 2319.3 | 4797.3 |
| A14 | Core ML (hybrid default) | 98.2 | 144.8 | 185.7 | 260.9 | 479.9 |
| A14 | MLX | 911.1 | 2050.3 | 2814.4 | 3950.8 | OOM-killed |

## Every cell: warm median [min-max], cold, ms

### M2 Ultra (`m2-studio`)

| Arm | 3s | 7s | 10s | 15s | 30s |
| --- | ---: | ---: | ---: | ---: | ---: |
| folded | 73.2 [71.7-73.9], cold 74.1 | 171.1 [170.6-172.2], cold 171.2 | 255.5 [254.8-256.9], cold 256.5 | 352.8 [351.5-372.4], cold 352.9 | 706.4 [704.3-711.6], cold 707.0 |
| hybrid | 73.6 [72.0-77.7], cold 73.6 | 53.7 [51.5-63.2], cold 51.7 | 70.6 [68.3-102.8], cold 69.5 | 88.9 [85.5-90.7], cold 89.4 | 158.9 [154.9-164.8], cold 159.4 |
| hybrid-e5 | 72.7 [71.7-73.5], cold 73.7 | 87.9 [84.6-96.4], cold 89.8 | 93.5 [89.0-96.7], cold 93.2 | 125.3 [116.1-199.5], cold 120.4 | 213.1 [204.0-216.8], cold 225.4 |
| mlx | error | 189.1 [183.4-197.9], cold 233.3 | 272.5 [245.7-341.8], cold 28627.7 | 340.4 [328.0-1145.7], cold 640.2 | 769.8 [695.5-2346.6], cold 2458.1 |

### M2 Air (`m2-air`)

| Arm | 3s | 7s | 10s | 15s | 30s |
| --- | ---: | ---: | ---: | ---: | ---: |
| folded | 72.2 [70.6-74.9], cold 73.5 | 186.1 [184.5-200.9], cold 185.7 | 286.5 [282.6-292.3], cold 291.9 | 392.9 [389.5-411.4], cold 409.1 | 821.3 [814.5-829.4], cold 817.1 |
| hybrid | 73.0 [70.7-82.7], cold 71.6 | 71.9 [69.2-77.4], cold 80.3 | 240.1 [144.0-258.7], cold 244.0 | 272.9 [264.2-283.6], cold 265.2 | 377.4 [342.0-393.1], cold 382.5 |
| hybrid-e5 | 82.6 [76.9-86.2], cold 80.6 | 154.4 [137.3-175.3], cold 149.2 | 95.0 [92.7-104.0], cold 97.0 | 131.0 [125.5-138.1], cold 127.1 | 229.7 [225.9-245.3], cold 241.5 |
| mlx | error | 1431.6 [1363.9-1527.1], cold 1308.0 | 1817.5 [1708.5-1960.7], cold 13080.9 | 2426.3 [2321.0-2530.6], cold 2543.6 | 5186.7 [4559.5-5509.9], cold 5428.2 |

### M1 Mini (`irvine-m1`)

| Arm | 3s | 7s | 10s | 15s | 30s |
| --- | ---: | ---: | ---: | ---: | ---: |
| folded | 93.9 [91.1-101.3], cold 93.3 | 243.0 [239.0-252.3], cold 239.6 | 362.3 [354.1-380.0], cold 373.9 | 522.5 [496.2-538.0], cold 505.7 | 1091.7 [1045.3-1099.7], cold 1146.2 |
| hybrid | 94.9 [91.0-115.4], cold 92.9 | 98.9 [89.4-104.4], cold 94.1 | 123.8 [118.6-136.8], cold 132.9 | 168.5 [166.0-171.3], cold 178.9 | 308.5 [293.0-316.6], cold 304.8 |
| hybrid-e5 | 93.7 [90.7-99.3], cold 98.1 | 93.0 [90.2-104.3], cold 104.0 | 124.1 [120.1-135.2], cold 132.3 | 172.5 [169.2-181.5], cold 181.7 | 298.2 [286.5-305.0], cold 327.8 |
| mlx | error | 800.5 [790.3-812.8], cold 784.1 | 1102.9 [1090.8-1116.4], cold 9252.4 | 1542.3 [1506.1-1566.3], cold 1557.3 | 3014.6 [2955.2-3051.9], cold 2996.4 |

### A17 Pro (`iphone-15-pro-max`)

| Arm | 3s | 7s | 10s | 15s | 30s |
| --- | ---: | ---: | ---: | ---: | ---: |
| folded | 62.9 [60.9-111.8], cold 63.4 | 152.6 [147.7-188.0], cold 145.8 | 228.6 [225.1-255.1], cold 231.3 | 373.8 [371.3-377.5], cold 377.7 | 780.9 [768.7-797.0], cold 777.2 |
| hybrid | 61.8 [58.5-63.9], cold 62.6 | 103.2 [92.8-111.1], cold 96.9 | 155.9 [148.7-168.0], cold 148.4 | 223.8 [200.5-242.6], cold 202.3 | 318.5 [302.5-324.2], cold 312.0 |
| hybrid-e5 | 58.7 [57.9-59.6], cold 58.7 | 125.6 [102.3-151.8], cold 127.1 | 188.6 [177.9-198.4], cold 189.2 | 245.3 [202.9-256.0], cold 244.6 | 335.0 [319.1-345.9], cold 335.0 |
| mlx | 551.0 [529.0-562.2], cold 521.2 | 1226.8 [1165.6-1286.0], cold 1220.7 | 1717.1 [1677.0-1788.1], cold 1732.6 | 2319.3 [2295.9-2367.5], cold 2290.6 | 4797.3 [4427.1-8127.7], cold 4521.1 |

### A14 (`iphone-12-pro`)

| Arm | 3s | 7s | 10s | 15s | 30s |
| --- | ---: | ---: | ---: | ---: | ---: |
| folded | 98.5 [95.7-99.2], cold 98.6 | 274.4 [271.9-276.1], cold 274.8 | 453.6 [451.0-455.3], cold 452.8 | 711.5 [704.4-740.5], cold 708.7 | 1555.7 [1552.7-1599.5], cold 1597.8 |
| hybrid | 98.2 [97.7-98.9], cold 97.9 | 144.8 [142.7-149.5], cold 145.2 | 185.7 [179.9-195.2], cold 183.6 | 260.9 [255.8-269.2], cold 259.5 | 479.9 [469.4-499.9], cold 485.1 |
| hybrid-e5 | 97.7 [96.2-99.4], cold 99.4 | 142.1 [139.8-144.7], cold 145.5 | 187.8 [183.2-191.0], cold 189.1 | 264.3 [256.9-273.2], cold 255.5 | 454.7 [440.3-466.9], cold 456.1 |
| mlx | 911.1 [892.3-921.2], cold 907.5 | 2050.3 [2039.8-2054.1], cold 2049.3 | 2814.4 [2789.1-2831.8], cold 2811.9 | 3950.8 [3911.5-3980.0], cold 3924.9 | OOM-killed |

## Folded vs hybrid (ties: 5% / overlapping ranges, or below the 3 s same-pipeline gap)

| Device | 3 s gap | 7 s | 10 s | 15 s | 30 s | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| M2 Ultra | 0.5% | -68.6% | -72.4% | -74.8% | -77.5% | hybrid faster, no ties |
| M2 Air | 1.1% | -61.4% | -16.2% | -30.5% | -54.0% | hybrid faster, no ties |
| M1 Mini | 1.1% | -59.3% | -65.8% | -67.8% | -71.7% | hybrid faster, no ties |
| A17 Pro | 1.7% | -32.4% | -31.8% | -40.1% | -59.2% | hybrid faster, no ties |
| A14 | 0.3% | -47.2% | -59.1% | -63.3% | -69.2% | hybrid faster, no ties |

## Ablation, warm median ms

| Device | Input | R0 October | R1 + split | R2 + ANE DecoderPre | R3 + folded | R4 hybrid |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| M2 Air | 3s | 122.5 | 115.3 | 115.5 | 72.2 | 73.0 |
| M2 Air | 30s | 1162.8 | 1118.6 | 1090.9 | 821.3 | 377.4 |
| A14 | 3s | 410.1 | 395.2 | 394.1 | 98.5 | 98.2 |
| A14 | 30s | 3829.2 | 3715.2 | 3678.9 | 1555.7 | 479.9 |

## GPU share (hybrid, generator only, instrumented launches)

`KOKORO_HYBRID_TIMELINE=1`, one launch per phone per input, median of 10 warm
calls. Busy times are host timestamps around ANE calls and GPU command buffers,
so they are occupancy, not hardware counters.

| Phone, input | Span | ANE busy | GPU busy | Both |
| --- | ---: | ---: | ---: | ---: |
| iphone-12-pro 10s | 149.5 | 108.0 (72%) | 107.0 (72%) | 70.5 |
| iphone-12-pro 30s | 375.0 | 291.5 (78%) | 267.5 (71%) | 192.0 |
| iphone-15-pro-max 10s | 101.5 | 76.0 (75%) | 62.0 (61%) | 40.0 |
| iphone-15-pro-max 30s | 240.0 | 185.5 (77%) | 171.0 (71%) | 122.5 |

## Peak RAM at 30 s (MB)

Macs: maximum resident set size, one fresh process per cell. Phones: process
`phys_footprint` peak over the launch.

| Mac | folded | hybrid | hybrid-e5 | MLX |
| --- | ---: | ---: | ---: | ---: |
| M2 Ultra | 415 | 434 | 367 | 815 |
| M2 Air | 393 | 376 | 360 | 806 |
| M1 Mini | 404 | 375 | 362 | 801 |

| Phone | folded | hybrid | hybrid-e5 |
| --- | ---: | ---: | ---: |
| A17 Pro | 779.6 | 1337.8 | 1057.2 |
| A14 | 284.3 | 830.6 | 643.0 |

## Placement proof

- M2 Air, `MLComputePlan` (`scripts/inspect_coreml_compute_plan.m`, extended to
  one plan per function of a multifunction package): folded 30 s, both the
  native-sin and `poly` variants, 53 of 53 convs on the ANE; hybrid r10, 53 of
  53 functions with their conv on the ANE.
- A14, on device: folded `poly` 30 s (the A14's variant), 1,927 of 1,928 ops
  on the ANE (`scripts/dump_device_compute_plan.py --device-name Webcam`);
  hybrid r10 through `tools/ane-lab --function`, 83 ANE ops across the 53
  functions and none on CPU or GPU (same 83 as the M2 Air).
- e5rt compiles with device mask 4 (ANE only); a non-ANE op would fail compile.

## Quality

Gate set before results: SNR >= 40 dB per clip against the fp32 PyTorch
generator on the same inputs (`quality_014.py`), and equal predicted frames.

- Folded 40.8-44.1 dB, hybrid and hybrid-e5 40.8-45.8 dB over the five buckets
  plus three hard inputs (numbers, heavy punctuation, "Hi."). Worst clip:
  "Hi." at 40.83 dB (folded generator in every arm, below 7 s).
- Split vs unsplit duration: identical predicted frames on all eight clips.
- Listening: the owner listened to five clips (30 s, numbers, punctuation,
  "Hi.", folded 10 s) and passed all five. The Gemini judge was not used: its
  Google project returned HTTP 403 (billing), and the owner chose human
  listening. Not a blind or rated study.

## Protocol as run, and deviations

- Config F: `run_config_f_reference.py --compute-units staged
  --use-padded-duration-models --iterations 10 --preflight-runs 3` (Macs);
  `ios-bench --mode ladder --policy runtime` (phones, 3 warm-ups + 1 cold + 10).
- Hybrid serves buckets >= 7 s (`HybridGenerator.minBucketSec`); at 3 s the
  folded and hybrid arms run the same pipeline, so the 3 s gap is the device's
  launch-to-launch noise floor.
- **Mac load gate failed on every arm** (background apps; Botnet paused with no
  active jobs): M2 Ultra load 4-7, M2 Air 3-5, M1 2-3. All Mac rows are upper
  bounds. The M2 Air was the noisiest: its hybrid 30 s (377 ms) is far above a
  quiet 223 ms measured on 2026-10-22, and its MLX ran ~2x slower than in plan
  013, so the M2 Air MLX ratio is not comparable with October's. Gate wait cut
  from 20 to 2 minutes for the remaining arms at 12:20 PT (owner: finish today).
- **Phone cooldown cut from 10 to 3 minutes** after the first 3 s cells
  (owner). Check on the A14: folded 3 s 92.0 ms (10 min) vs 98.5 ms (3 min),
  inside the 92.0-98.2 spread of two 10-minute same-pipeline cells. The A17 has
  no clean 10-minute reference: the owner was using it during its early cells;
  those and two warm starts are in `iphone-15-pro-max/superseded/`. The A17
  ran warm ("fair") at 3 minutes several times; every kept cell started and
  ended at nominal, and its last six used 6 minutes.
- iPhone app rebuilt at 11:30 PT to restore the MLX arm's model files
  (`ios-bench/Resources/mlx` had been emptied on 2026-10-21); Core ML code and
  packages unchanged. Hashes for both builds are in the manifest.
- MLX: every 3 s Mac input fails with the plan 013 broadcast error; the A14 MLX
  30 s cell is OOM-killed (twice), as in plan 013.

## Botnet

Paused for the runs, restored to the prior override: m2-studio `paused` (as
found), m2-air `paused` (as found), irvine-m1 `dedicated`.

## Not measured

- The SDK's own provider paths (`KokoroSDKModelProvider` hybrid, split, fold):
  the matrix used `kokoro-bench`'s and `ios-bench`'s providers. Smoke-test the
  release bundle before publishing.
- Multifunction packages: excluded everywhere; with a multifunction duration
  the split duration is skipped (`KokoroPipeline.swift:640`).
