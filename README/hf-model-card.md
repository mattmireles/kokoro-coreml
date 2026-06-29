---
license: apache-2.0
library_name: coreml
pipeline_tag: text-to-speech
language:
  - en
base_model: hexgrad/Kokoro-82M
tags:
  - text-to-speech
  - coreml
  - kokoro
  - apple-silicon
  - ane
  - neural-engine
  - on-device
---

# Kokoro 82M TTS -- Surgically Optimized for Apple Silicon

**30 seconds of speech in 379 ms on a Mac Studio. 2x faster than MLX on the same hardware. Running on the Apple Neural Engine.**

[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) compiled to Core ML and cut into five models, each running on the processor that's best at its job. On-device, offline, no API keys, no cents-per-character. This repo is the pre-converted `.mlpackage` files; you load them with a Swift `MLModel(contentsOf:)` call.

> **Source, exporters, Swift runtime:** [github.com/mattmireles/kokoro-coreml](https://github.com/mattmireles/kokoro-coreml)

## The numbers

Median warm wall time for one full synthesize call -- tokenize in, 24 kHz PCM out. Measured June 2026, counterbalanced harness in the GitHub repo.

| Audio | M2 Studio (64 GB) | M2 Air (24 GB) | M1 Mini (16 GB) |
|---|---:|---:|---:|
| 3s  | **51 ms** | 148 ms | 234 ms |
| 10s | **126 ms** | 466 ms | 686 ms |
| 30s | **379 ms** | 1,405 ms | 1,959 ms |

That's 12-79x realtime across the lineup. The M2 Studio synthesizes 30 seconds of audio in 379 ms, but the M1 Mini is the number that matters -- the cheapest Apple Silicon Mac you can buy turns text into speech 14x faster than you can listen to it.

## vs MLX

Same machines, same utterances, same voice (`af_heart`), same timing boundary, median of warm calls. Comparator: [Blaizzy/mlx-audio](https://github.com/Blaizzy/mlx-audio) 0.4.3 at commit `862dfbe`, running `mlx-community/Kokoro-82M-bf16`.

| Audio | M2 Studio | M2 Air | M1 Mini |
|---|---|---|---|
| 3s  | 51 ms vs *error* | 148 ms vs *error* | 234 ms vs *error* |
| 7s  | 96 vs 224 ms -- **2.3x** | 331 vs 686 ms -- **2.1x** | 493 vs 824 ms -- **1.7x** |
| 10s | 126 vs 289 ms -- **2.3x** | 466 vs 836 ms -- **1.8x** | 686 vs 1,124 ms -- **1.6x** |
| 30s | 379 vs 763 ms -- **2.0x** | 1,405 vs 2,600 ms -- **1.9x** | 1,959 vs 3,078 ms -- **1.6x** |

**Faster on every bucket, on every machine.** The gap is widest on the newest silicon -- the Neural Engine keeps scaling while a GPU-bound port doesn't. (The pinned MLX version fails 3-second clips with a broadcast-shape error; no time to report.)

This is not a knock on MLX -- it's a fine framework. It's the surgery. A monolithic port runs wherever the scheduler drops it. A dissected pipeline runs each stage where it belongs.

## On iPhone

Same `.mlpackage` files, deployed to phones (June 2026, iOS 26.5, median of 5 warm calls). Comparator: [mlalma/kokoro-ios](https://github.com/mlalma/kokoro-ios) 1.0.8, the MLX Swift port of Kokoro -- the Python mlx-audio above doesn't run on iOS. Its timing includes its Misaki G2P pass; ours starts from token IDs.

| Audio | iPhone 15 Pro Max | iPhone 12 Pro |
|---|---|---|
| 3s  | 702 vs 919 ms -- **1.3x** | 1,383 vs 1,624 ms -- **1.2x** |
| 7s  | 1,492 vs 1,875 ms -- **1.3x** | 2,966 vs 2,405 ms -- 0.8x |
| 15s | 3,272 vs 3,805 ms -- **1.2x** | 6,250 vs 5,022 ms -- 0.8x |
| 30s | 6,374 vs 7,792 ms -- **1.2x** | 12,301 ms vs *OOM* |

Faster on every bucket on the A17 Pro (4-4.5x realtime). On the 4 GB iPhone 12 Pro it's split: MLX takes the middle buckets, but the memory watchdog kills it on 30-second clips -- this pipeline synthesizes them in 12.3 s. One disclosure: the iPhone ANE compiler (A14 **and** A17 Pro) rejects the full-ANE plan that every M-series Mac runs (`ANECCompile() FAILED`), so iPhone rows use the staged policy -- decoder-pre on the ANE, the other stages on CPU+GPU.

Cold start takes a few seconds (Core ML compiles on first load); everything after is steady-state. Benchmarks drift with macOS and hardware -- rerun them on your target machine with the harness in the GitHub repo before you ship a claim of your own.

## Why surgery?

Apple Silicon isn't one processor. It's three -- **CPU, GPU, and the Neural Engine (ANE)** -- each built for different work. The ANE devours fixed-shape convolutions at a fraction of the GPU's power draw. But it has rules: no dynamic shapes, no data-dependent control flow. Shove a whole TTS model through Core ML and the scheduler quietly dumps you on the CPU.

So we cut the pipeline at the joints:

```
                  ┌────────────────────────────────┐
"Hello world" ──▶ │  DURATION  (kokoro_duration_t*) │ ◀── CPU/GPU
                  │  BERT + LSTMs                   │     branching, variable lengths
                  └──────────────┬─────────────────┘
                                 ▼
                  ┌────────────────────────────────┐
                  │  ALIGNMENT  (Swift)             │ ◀── CPU
                  │  Matrix from durations, ~50 LoC │     small, data-dependent logic
                  └──────────────┬─────────────────┘
                                 ▼
                  ┌────────────────────────────────┐
                  │  F0 / NOISE  (kokoro_f0ntrain)  │ ◀── ANE
                  │  Pitch + aperiodicity contours  │     fixed-shape dense math
                  └──────────────┬─────────────────┘
                                 ▼
                  ┌────────────────────────────────┐
                  │  DECODER PRE (kokoro_decoder_pre)│ ◀── ANE
                  │  Text features → decoder state  │     fixed-shape convolutions
                  └──────────────┬─────────────────┘
                                 ▼
                  ┌────────────────────────────────┐
                  │  HARMONIC SOURCE  (Swift/vDSP)  │ ◀── CPU
                  │  hn-NSF sine + noise excitation │     cheap DSP, exact phase
                  └──────────────┬─────────────────┘
                                 ▼
                  ┌────────────────────────────────┐
                  │  GENERATOR (kokoro_decoder_     │ ◀── ANE
                  │  har_post) convs + iSTFT        │     dense parallel tensor math
                  └──────────────┬─────────────────┘
                                 ▼
                            24 kHz Audio
```

Four models on the ANE, one DSP stage in Swift with double-precision phase accumulation. The generator has zero `nn.Linear` ops -- all 48 replaced with `Conv1d(kernel_size=1)` so the MIL graph stays on the ANE path.

**Redesign the inference pipeline, not the model.** That's where the 2x over MLX comes from -- not by fighting the GPU, but by routing around it.

## What's in the download

Five fixed-duration buckets: **3s, 7s, 10s, 15s, 30s**. Pick the smallest bucket that fits your predicted utterance. That's the whole strategy.

| File | What it does | Runs on |
|---|---|---|
| `kokoro_duration_t{32,64,128,256,320,384,512}.mlpackage` | Phoneme durations + text/style encodings, one per padded token length | CPU/GPU |
| `kokoro_duration.mlpackage` | Legacy single duration model (fallback) | CPU/GPU |
| `kokoro_f0ntrain_t{120,280,400,600,1200}.mlpackage` | Pitch + noise prediction, one per bucket's frame count | ANE |
| `kokoro_decoder_pre_{3,7,10,15,30}s.mlpackage` | Text features → decoder hidden state | ANE |
| `kokoro_decoder_har_post_{3,7,10,15,30}s.mlpackage` | Generator: harmonic-excited convolutions + iSTFT → waveform | ANE |

The alignment matrix and the hn-NSF harmonic source are not models -- they're a few hundred lines of Swift/vDSP in the GitHub repo's `KokoroPipeline`.

## Usage (Swift SDK)

This section is the Swift SDK contract for the matching Git release commit. The
repo publishes SDK bundle manifests and checksums at the top level for the
starter profile and under `sdk/starter/` and `sdk/full/` for profile-specific
metadata. If you are using an older HF snapshot, use the low-level
`KokoroPipeline` snippets from that snapshot instead.

```swift
import KokoroTTS

let resources = KokoroResourceProvider.directory(bundleURL)
let tts = try await KokoroTTS.load(resources: resources)
try await tts.prewarm(text: "Hello world.", voice: .afHeart)

let audio = try await tts.synthesize("Hello world.", voice: .afHeart)
let samples = audio.samples        // 24 kHz mono Float PCM
let buffer = try audio.makePCMBuffer()
```

The SDK is in the GitHub repo's `swift-tts` package. It owns raw-text
preparation, Misaki phonemization, Botnet-compatible chunking, model loading,
and AVFoundation PCM buffer creation. Current SDK contract: iOS 18.0+,
macOS 15.0+, sample rate `24000`, starter voice `af_heart`, duration token
sizes `32,64,128,256,320,384,512`, full buckets `3,7,10,15,30`, max caller
chunk tokens `450`, voice embedding dimension `256`.

Build a starter bundle:

```bash
python3 scripts/download_models.py \
  --repo-id mattmireles/kokoro-coreml \
  --revision <hf-revision> \
  --sdk-profile starter \
  --manifest-out /tmp/kokoro-download-manifest.json

node scripts/build_sdk_bundle.mjs \
  --profile starter \
  --compile-models 1 \
  --output /tmp/kokoro-sdk-starter \
  --repo-id mattmireles/kokoro-coreml \
  --revision <hf-revision> \
  --download-manifest /tmp/kokoro-download-manifest.json

node scripts/validate_sdk_bundle.mjs /tmp/kokoro-sdk-starter
```

Downloaded-resource apps can hydrate the top-level starter
`HostedManifest.json` with `KokoroDownloadedModelStore`. Production apps should
serve manifests over HTTPS and pin the expected HF revision or
`sdk/SDKReleaseManifest.json` checksum. Bundled-resource apps can use
`KokoroResourceProvider.directory`, `.appBundle`, or `.packageBundle`.

Previous snippets that used `KokoroPipeline` directly are now low-level
examples. Keep them for benchmarking and graph work; use `KokoroTTS` for app
integration.

## Tensor shapes (3s bucket)

```
kokoro_duration_t128:
  in   input_ids       [1, 128]        int32     phoneme token IDs (padded)
  in   attention_mask  [1, 128]        float16
  in   ref_s           [1, 256]        float16   voice embedding
  in   speed           [1]             float16
  out  pred_dur        [1, 128]                  per-token frame counts
  out  t_en, d, s, ref_s_out                     encodings for downstream stages

kokoro_f0ntrain_t120:
  in   en   [1, 640, 120]   out  F0_pred [1, 240], N_pred [1, 240]

kokoro_decoder_pre_3s:
  in   asr [1, 512, 120]  f0 [1, 1, 240]  n_input [1, 1, 240]  ref_s [1, 256]
  out  x_pre [1, 512, 240]

kokoro_decoder_har_post_3s:
  in   x_pre [1, 512, 240]  ref_s [1, 256]  har [1, 22, 28801]
  out  waveform [1, 1, 72000]   -- 3s @ 24 kHz
```

Everything is static and float16. No dynamic ops. No `RangeDim`. No `non_zero` kernels.

## Requirements

- **iOS 18.0+ / macOS 15.0+** for the drop-in raw-text `KokoroTTS` SDK
- **Apple Silicon** (M1+) or **A15+** for Neural Engine acceleration
- Runs on older chips too, just slower

## License

Apache 2.0, inherited from [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M). Ship it. Sell it. Fork it.

## Credits

- **[@hexgrad](https://huggingface.co/hexgrad)** -- Kokoro-82M weights, training, and the Apache release
- **[@yl4579](https://huggingface.co/yl4579)** -- StyleTTS 2 architecture
- **Apple's coremltools team** -- for maintaining the PyTorch-to-Core ML path

---

*Kokoro (心) -- Japanese for "heart."*
