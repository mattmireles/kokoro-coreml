# Hugging Face Release of the Re-exported Model Set (2026-09-24)

Date: 2026-09-24
Source commit: `40044e1f` (packages), `69bf1177` (SDK pins)

## Why

The outside-contributor stack (#5, #7–#13) merged to `main` changed every
exported graph, but Hugging Face still served packages from before #6. Those
packages had no `mask` inputs, carried the doubled `har` geometry (#9) and had
no flexible generator (#12). This note records the re-export, the gates it
passed, and the publish. Decisions and mechanisms live in
[`debug-notes.md`](debug-notes.md) (issues "Bucket padding cost every utterance
the price of its bucket" and "Bucket padding contaminated every time-axis
statistic…"); the earlier single-family re-upload procedure is
[`hf-fixed-package-reupload-2026-07-14.md`](hf-fixed-package-reupload-2026-07-14.md).

## Export

Environment: repo `.venv` (coremltools 8.3.0, torch 2.6.0; the paper pins 2.5.0,
and coremltools warns but nothing broke). Weights: HF `hexgrad/Kokoro-82M`
snapshot `f3ff3571`, because the local `checkpoints/` symlinks were unreadable.

```bash
KOKORO_DURATION_EXPORT_SIZES=32,64,128,256,320,384,512 .venv/bin/python export_duration.py
.venv/bin/python export_f0ntrain.py --t-frames 120 280 400 600 1200 --output-dir coreml
.venv/bin/python export_decoder_pre.py --buckets 3 7 10 15 30 --output-dir coreml
.venv/bin/python -m export_synth.main --mode decoder-har --buckets 3s,7s,10s,15s,30s -o coreml
.venv/bin/python -m export_synth.main --mode decoder-har --time-axis range --buckets 30s -o coreml
.venv/bin/python scripts/build_multifunction_packages.py --models-dir coreml
```

`export_synth/main.py` and `export_decoder_pre.py` exit 0 even when conversion
fails. Check each log for the `Saved` line.

## Gates (all passed)

Parity against the same export wrapper in PyTorch fp32, on real tensors from
the frozen inputs, with the same mask, over the valid region:

| stage | result |
| --- | --- |
| duration `pred_dur` | 1,448 of 1,449 tokens exact (one frame of rounding at 10 s) |
| f0ntrain F0 / N | 66.6–71.0 dB |
| decoder-pre x_pre (on the ANE) | 54.5–58.7 dB |
| fixed generators | corr ≥ 0.999988, 46.1–46.8 dB |
| flexible generator | 44.4–45.1 dB at exact length; 42.5–45.2 dB with the 0.5 s granule |

`scripts/check_coreml_generator_from_dump.py` reports 6–8 dB because its
PyTorch reference omits the mask. PyTorch without the mask against PyTorch with
it reproduces those numbers with no Core ML involved. Fixing the checker is an
open follow-up.

Tests, zero skips:
- pytest: 147 passed.
- `swift test --package-path swift`: 60 tests, 0 failures.
- `swift test --package-path swift-tts` with `KOKORO_RUN_MISAKI_RUNTIME_TESTS=1`: 74 tests, 0 failures.

Getting to zero skips exposed two stale tests and one silent break:
- `IdentityAdaIN` had rejected the mask argument since #6, so the `--mode full`
  and `--mode decoder` exports failed silently. Fixed, with a regression test.
- The multifunction test pointed at a contributor's home directory. Fixed.

Bench (staged units, 2% duration check, 3 warmups then 10 runs, M2 Ultra under
background load of ~11.7, so these are upper bounds): 50 of 50 runs `status=ok`.

| input | warm median | realtime |
| --- | ---: | ---: |
| 3s | 56.7 ms | 50x |
| 7s | 93.9 ms | 72x |
| 10s | 128.2 ms | 74x |
| 15s | 162.0 ms | 87x |
| 30s | 306.9 ms | 90x |

Compute plans (preferred device, production units, M2 Ultra):

| package | placement |
| --- | --- |
| decoder-pre 3, 15, 30 s (`cpuAndNeuralEngine`) | 250 of 250 ops on the ANE |
| duration t128 | 73.5% GPU, 26.5% CPU (all 10 LSTM ops on the CPU) |
| f0ntrain t600 | 99.3% GPU; its 2 LSTM ops on the CPU |
| flexible generator | 100% GPU |

These are compute-plan placements, not an Instruments trace.

## Publish

Order: binaries, then provenance manifests, then bundles, then metadata.

1. Packages: HF commit `dadebca4`, via `upload_folder` from `coreml/`
   restricted to the 23-package publish set:
   - duration t32–t512;
   - f0ntrain ×5;
   - decoder-pre ×5;
   - fixed generators ×5;
   - `kokoro_decoder_har_post_range`.

   Not published: multifunction, legacy, `exact_t*` and
   `kokoro_duration.mlpackage`. The generator `weight.bin` is byte-identical to
   the July upload (`e4ada8b2…`); the changes are in `model.mlmodel`.
2. `scripts/download_models.py --revision dadebca4` for the starter and full
   profiles. All 69 local publish-set files hashed identically before and after
   the download.
3. `scripts/build_sdk_bundle.mjs` for starter and full, then
   `validate_sdk_bundle.mjs`: exit 0 for both. The builder refuses a dirty tree,
   so the COORD ledgers were stashed around it. The starter bundle now ships
   `kokoro_decoder_har_post_range` instead of per-bucket generators.
4. `scripts/prepare_hf_sdk_metadata.py --upload`: HF `5bb7d75d`,
   `sdk_commit 40044e1f`. The starter `HostedManifest.json` SHA-256 is
   `2f61bf6fce9f7fafa58a5f08f5aada0035a81741552692eda36bd0c331be8b6c`.
5. Model card corrected and re-uploaded (`2bd8f920`): pins, `mask` inputs,
   `har` shape, flexible generator, measured placement.

End-to-end checks with `kokoro-sdk-smoke`:
- From the local starter bundle: speech-level output, about −27 dBFS RMS.
- From the published manifest in downloaded-resource mode: 3.45 s of audio,
  exit 0.
- A cache directory under `/private/tmp` trips the SDK's `pathEscape` guard. The
  same run with a cache under `~/Library/Caches` passes.

## Downstream

- Botnet: `kokoro-coreml-runtime` synced to `69bf1177` (botnet `f28103c0`,
  pushed; its Swift suite has 0 failures). Botnet's artifact list does not
  include the flexible generator, so the Mac workers keep the bucketed
  generators. Installing the artifacts on the fleet is a separate step.

## iPhone 12 Pro measurement (2026-09-24)

Physical iPhone 12 Pro (iPhone13,3, 4 GB), iOS 27.0, Release build. Thermal
state was nominal on every call, with 7 launches and a 10-minute cooldown
between them; warm runs went in the order A, B, B, A. The harness was a
minimal scratch app that calls the SDK's synthesis executor with the SDK
compute policy. One binary holds both generators, and the arms differ only in
which one loads:

- A: `kokoro_decoder_har_post_15s` (bucketed, today's starter bundle before
  this release).
- B: `kokoro_decoder_har_post_range` (flexible).

Warm end-to-end medians, 10 calls per arm:

| audio | A | B | speedup | generator alone, A → B |
| --- | ---: | ---: | ---: | --- |
| 2.15 s | 1,799 ms | 418 ms | 4.31x | 1,645 → 288 ms |
| 6.78 s | 1,823 ms | 966 ms | 1.89x | 1,652 → 799 ms |
| 13.9 s | 1,850 ms | 1,825 ms | 1.01x (tie, ±6%) | 1,659 → 1,636 ms |

- A costs about 1.8 s at every length because it always computes the full
  15 s bucket. B scales with the real length.
- First-call stall (B only):
  - After a fresh install: +620–760 ms at each new short length, and +330 ms
    at 13.9 s.
  - After a relaunch: +290–400 ms.
  - It is paid once per length per process, and B's first short call still
    beats A's warm call.
- Cold start is a wash: 11.1 s (A) against 10.5–10.7 s (B) after install, and
  2.2 s against 2.2–2.5 s on relaunch.
- Peak memory: 986–1,075 MB (A) against 609–634 MB (B).
- Audio: finite, with equal RMS and sample counts across the arms.
- Caveats:
  - The two arms' waveforms correlate at 0.995; nobody has listened.
  - The 13.9 s input needed the t256 duration model, while the real SDK
    would split it at 128 tokens.
  - The harness skips G2P, chunking and on-device compile.
  - One phone, one session.
