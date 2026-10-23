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

# Kokoro TTS: 9.8x faster than MLX on an M1 Mac

**It speaks 27 seconds of audio in under a third of a second, on a 2020 M1 Mac Mini or an iPhone 15 Pro Max.** MLX was measured on the same Mac, with the same text.

[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) compiled to Core ML and cut at its seams, so each stage runs on the engine that suits it. Even the vocoder runs on the Apple Neural Engine, which rejects it as exported. On-device, offline, no API keys, no cents per character.

> **Source, Swift SDK, paper:** [github.com/mattmireles/kokoro-coreml](https://github.com/mattmireles/kokoro-coreml)

## Try it

On an Apple Silicon Mac (macOS 15+, Xcode 26+):

```bash
git clone https://github.com/mattmireles/kokoro-coreml && cd kokoro-coreml
swift run -c release kokoro-say "Hello from the Neural Engine."
afplay kokoro.wav
```

The CLI downloads the pinned starter bundle from this repo (about 210 MB, SHA-256 checked) and prints the time per stage and which path ran each one.

In an app (iOS 18+ / macOS 15+):

```swift
// .package(url: "https://github.com/mattmireles/kokoro-coreml", revision: "<latest main commit>")
// .product(name: "KokoroTTS", package: "kokoro-coreml")
import KokoroTTS

let tts = try await KokoroTTS.load(resources: .huggingFace())
let audio = try await tts.synthesize("Hello world.", voice: .afHeart)   // 24 kHz mono PCM
```

## The numbers

Warm median end-to-end time, token IDs in, 24 kHz audio out. Measured October 23, 2026 against MLX on the same device in the same session:

| Audio | M2 Ultra | M1 Mini | iPhone 15 Pro Max | iPhone 12 Pro |
| --- | ---: | ---: | ---: | ---: |
| 3s | 74 ms | 95 ms | 62 ms | 98 ms |
| 7s | 54 ms | 99 ms | 103 ms | 145 ms |
| 10s | 71 ms | 124 ms | 156 ms | 186 ms |
| 15s | 89 ms | 169 ms | 224 ms | 261 ms |
| 30s (27.4 s input) | **159 ms** | **309 ms** | **319 ms** | **480 ms** |
| **vs MLX** | 3.5-4.8x | 8.1-9.8x | 8.9-15.1x\* | 9.3-15.2x; MLX runs out of memory at 30 s |

At least 57x real time on the 27.4 s input on every device.
- **Comparators:** [mlx-audio](https://github.com/Blaizzy/mlx-audio) 0.4.3 on Macs (it fails every 3 s input) and [kokoro-ios](https://github.com/mlalma/kokoro-ios) 1.0.8 on iPhones.
- **Mac rows are upper bounds.** The hosts were busy, with background load 2-7.
- **\* A17 ratios at 10 s and above are upper bounds.** MLX ran hot on that phone.
- **Phone numbers are a sustained workload.**

Full protocol and raw data: [paper-cs1-hybrid-rerun-2026-10-23.md](https://github.com/mattmireles/kokoro-coreml/blob/main/README/Notes/paper-cs1-hybrid-rerun-2026-10-23.md).

## How it works

```
text -> Misaki G2P -> phoneme IDs
  DURATION      BERT front on the ANE, LSTM back on the CPU
  ALIGNMENT     Swift
  F0 + NOISE    CPU front, ANE back
  DECODER-PRE   ANE   || harmonic source in Swift/vDSP, concurrently
  GENERATOR     >=7 s: convolutions on the ANE, normalization + activations on the GPU (hybrid)
                <7 s or no Metal: folded, entirely on the ANE
  iSTFT + trim  Swift  -> 24 kHz audio
```

- **Folding.** The ANE rejects the vocoder because one time axis is up to 720,000 samples against a 16,384 limit. Folding that axis into overlapping rows, with five exact op rewrites, puts 1,927 of 1,928 ops on the ANE. The math is unchanged: 68 dB against the unfolded model.
- **Hybrid.** Splitting each vocoder stage between the ANE and the GPU, which share memory, cuts its time by 16-78% against the all-ANE form at every input of 7 s and longer (up to 4.4x faster).

Audio matches the fp32 PyTorch reference at 40.8-45.8 dB SNR per clip.

## What's in the repo

The SDK bundles list exact files with SHA-256 in `HostedManifest.json` (starter) and `sdk/<profile>/KokoroRuntimeManifest.json`. Pin a commit; see "Pinning" below.

| Path | What it is | Runs on |
| --- | --- | --- |
| `coreml/kokoro_duration_ane_front_t128` + `kokoro_duration_back_t128` | Duration model, split | ANE + CPU |
| `coreml/kokoro_f0ntrain_front_t{120,280,400,600,1200}` + `kokoro_f0ntrain_ane_back_t*` | Pitch and noise, split, one pair per bucket | CPU + ANE |
| `coreml/kokoro_decoder_pre_{3,7,10,15,30}s` | Decoder features, ANE layout | ANE |
| `coreml/kokoro_generator_fold_poly_{b}s`, `kokoro_generator_fold_{b}s` | Folded generator: polynomial Snake (every chip) and native `sin` (A17-class ANE) | ANE |
| `coreml/hybrid/` | Hybrid generator: `plan.json`, `adain.bin`, `hybrid_generator_r{2,3,4,6,10,11}` (53 Core ML functions each) | ANE + GPU (Metal) |
| `voices/` (starter voice), `kokoro.js/voices/` (all 28), `runtime/`, `g2p/` | Voice embeddings, vocab and hn-NSF weights, English G2P assets | -- |
| `coreml-{dur-ane,f0n-split,dpre-ane,fold,hybrid}/` | The paper's frozen package set (revision `e65729b9`), kept for reproduction | -- |
| other `coreml/` packages | Previous pipeline (GPU generator, whole duration/F0, and the old multifunction duration/F0/DecoderPre packages); not used by schema-2 bundles | -- |

Five fixed buckets: 3, 7, 10, 15 and 30 s. The runtime pads each utterance to the smallest bucket that fits and masks the padding.

## Pinning

Pin the commit that published the manifests: the one that last changed `HostedManifest.json` in the Files history. Every file a manifest lists is byte-identical there. The Swift SDK's `KokoroHostedModels` carries the current pin and the manifest's SHA-256. Bundles use manifest schema 2; older SDK commits reject them with `unsupportedManifestSchema`, so use the latest GitHub `main` SDK commit.

## Requirements

- iOS 18.0+ / macOS 15.0+, Apple Silicon. Intel Macs are untested and have no Neural Engine.
- watchOS is not supported: the MisakiSwift/MLX phonemizer does not build for it.

## License

Apache 2.0, inherited from [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M). The `g2p/` assets are Apache 2.0 from [FluidInference/kokoro-82m-coreml](https://huggingface.co/FluidInference/kokoro-82m-coreml) (commit `c032f6d05d2a58179b817201a443fbbcef06f962`).

## Credits

- **[@hexgrad](https://huggingface.co/hexgrad)** -- Kokoro-82M weights, training, and the Apache release
- **[@yl4579](https://huggingface.co/yl4579)** -- StyleTTS 2 architecture
- **[FluidInference](https://huggingface.co/FluidInference)** -- the English G2P lexicon and BART G2P Core ML models in `g2p/`
- **Apple's coremltools team** -- for maintaining the PyTorch-to-Core ML path

---

*Kokoro (心) -- Japanese for "heart."*
