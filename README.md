# Kokoro TTS: Surgically Optimized for Apple Silicon
**15.1x faster than MLX on iPhone 15 Pro Max. 9.8x faster than MLX on M1 Mac Mini. 4.8x faster than MLX on m2 Ultra.** MLX was measured on the same Mac, with the same text, same 27 sec audio duration.

[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) running on the Apple Neural Engine through Core ML. No Python, no server, no API key. The model is cut at its seams, and each stage runs on the part of the chip that suits it. That includes the vocoder, which the Neural Engine rejects as exported; a reshape that leaves its math unchanged gets it in.

> **Pre-converted models:** [huggingface.co/mattmireles/kokoro-coreml](https://huggingface.co/mattmireles/kokoro-coreml) · **Paper:** [`paper/main.pdf`](paper/main.pdf)

## Performance

Warm median end-to-end time, token IDs in, 24 kHz audio out. Measured October 23, 2026 on the shipped pipeline, against MLX on the same device in the same session:

| Audio | M2 Ultra | M1 Mini | iPhone 15 Pro Max | iPhone 12 Pro |
| --- | ---: | ---: | ---: | ---: |
| 3s | 74 ms | 95 ms | 62 ms | 98 ms |
| 7s | 54 ms | 99 ms | 103 ms | 145 ms |
| 10s | 71 ms | 124 ms | 156 ms | 186 ms |
| 15s | 89 ms | 169 ms | 224 ms | 261 ms |
| 30s (27.4 s input) | **159 ms** | **309 ms** | **319 ms** | **480 ms** |
| **vs MLX** | 3.5-4.8x | 8.1-9.8x | 8.9-15.1x\* | 9.3-15.2x; MLX runs out of memory at 30 s |

Every device runs at least 57x faster than real time on the 27.4 s input. MLX is [mlx-audio](https://github.com/Blaizzy/mlx-audio) 0.4.3 on Macs and [kokoro-ios](https://github.com/mlalma/kokoro-ios) 1.0.8 on iPhones; the pinned mlx-audio fails every 3 s input.

Caveats:
- **Mac rows are upper bounds.** No Mac reached the quiet-host gate; the background load was 2-7.
- **\* A17 ratios at 10 s and above are upper bounds.** MLX ran hot on that phone.
- **iPhone numbers are a sustained workload.** A single request after the phone has been idle is slower.
- **Short clips on the M2 Ultra regressed.** 3 s clips use the all-ANE generator, which is slower there than the old GPU generator: 74 ms here against 64 ms in October's run under similar load (45 ms in its quiet pass).

Full protocol, every cell with its min-max range, the ablation and raw data: [paper-cs1-hybrid-rerun-2026-10-23.md](README/Notes/paper-cs1-hybrid-rerun-2026-10-23.md).

### Memory

Peak memory for one synthesis, in MB, same run and same inputs as the latency table:

| Audio | M2 Ultra | M1 Mini | iPhone 15 Pro Max | iPhone 12 Pro |
| --- | ---: | ---: | ---: | ---: |
| 3s | 144 | 134 | 255 | 61 |
| 7s | 235 | 177 | 551 | 282 |
| 10s | 263 | 199 | 655 | 349 |
| 15s | 275 | 247 | 869 | 500 |
| 30s (27.4 s input) | **434** | **375** | **1,338** | **831** |
| **MLX** | 786-815 | 771-801 | not measured | runs out of memory at 30 s |

On a Mac this pipeline uses about half of MLX's memory at 30 s (434 vs 815 MB on the M2 Ultra, 375 vs 801 MB on the M1 Mini), and far less on short clips.

- **What's counted:** on Macs, the peak resident memory of a fresh process doing one synthesis; on iPhones, the app's peak memory footprint over the whole run.
- **Where the memory goes on iPhone:** the hybrid generator, used at 7 s and longer, buys its speed with memory. If memory matters more than speed, set `KOKORO_HYBRID_GENERATOR=0` to use the all-ANE generator everywhere. At 30 s it peaks at 780 MB on the iPhone 15 Pro Max and 284 MB on the iPhone 12 Pro, but takes 2.5-3.2x longer.
- **The MLX 3 s figures are missing:** mlx-audio fails every 3 s input.

### Thermal (iPhone)

iOS thermal state at the end of each run of 14 back-to-back syntheses of the same clip. Every run started cool (`nominal`):

| Audio | iPhone 15 Pro Max | MLX | iPhone 12 Pro | MLX |
| --- | --- | --- | --- | --- |
| 3s | nominal | nominal | nominal | nominal |
| 7s | nominal | nominal | nominal | nominal |
| 10s | nominal | fair | nominal | nominal |
| 15s | nominal | fair | nominal | nominal |
| 30s | nominal | **serious** | nominal | killed (out of memory) |

This pipeline never warmed either phone. MLX did, because it works 10-15x longer for the same speech: by the end of its 30 s run on the iPhone 15 Pro Max, iOS was throttling it (`serious`).

- **The four states:** `nominal`, `fair`, `serious` and `critical` are iOS's own reading ([`ProcessInfo.thermalState`](https://developer.apple.com/documentation/foundation/processinfo/thermalstate)), recorded at each call. They are not temperatures.
- **The old pipeline heated phones too:** before the vocoder moved to the Neural Engine, the iPhone 12 Pro ended its 30 s run at `serious`.

## How it works

Apple Silicon has three engines: CPU, GPU and the Neural Engine (ANE). The ANE runs fixed-shape convolutions fast and cheaply, but it accepts shapes, not models. Kokoro's vocoder fails ANE compilation because one axis is 720,000 samples long against a 16,384 limit. So the pipeline is cut at its seams, and each stage is reshaped until the right engine accepts it:

```
  text ──▶ Misaki G2P (Swift) ──▶ phoneme IDs
       │
       ▼
  DURATION        BERT front on the ANE, LSTM back on the CPU
       │
       ▼
  ALIGNMENT       Swift: expand tokens to frames
       │
       ▼
  F0 + NOISE      CPU front, ANE back
       │
       ▼
  DECODER-PRE     ANE                   ∥  HARMONIC SOURCE   Swift/vDSP, concurrently
       │
       ▼
  GENERATOR       ≥7 s: 53 convolutions on the ANE + normalization and activations on the GPU,
                  sharing IOSurface memory, both engines busy at once
                  <7 s, or if Metal is unavailable: folded, entirely on the ANE
       │
       ▼
  iSTFT + trim (Swift) ──▶ 24 kHz audio
```

- **Folding.** The generator's time axis is folded into rows of frames with 8-frame overlaps, and five rejected ops are rewritten into exact equivalents. Global normalization statistics stay exact. 1,927 of its 1,928 ops then run on the ANE, and it matches the unfolded model to 68 dB.
- **Hybrid.** Inside the generator, the ANE is slow at elementwise work and the GPU at convolutions. So the hybrid gives each engine the half it is good at. At every input of 7 s and longer, on all five devices measured, it takes 16-78% less time than the all-ANE version (up to 4.4x faster).

Audio matches the fp32 PyTorch reference at 40.8-45.8 dB SNR per clip.

**Redesign the inference pipeline, not the model.** The paper has the full story: [`paper/main.pdf`](paper/main.pdf).

## Reproduce the benchmark

The paper's numbers come from a frozen build: source commit `903f1fc1`, with its models at HF revision `e65729b9`. Plan, harness and per-call data:
- [README/Plans/014-paper-cs1-hybrid-frozen-rerun-plan.md](README/Plans/014-paper-cs1-hybrid-frozen-rerun-plan.md)
- [`paper/data/frozen_rerun_014/`](paper/data/frozen_rerun_014/)

The older PyTorch, Python and MLX bakeoff harness is in `scripts/bakeoff_harness.py`; see [bakeoff-results.md](README/Notes/bakeoff-results.md).

## Quickstart

On an Apple Silicon Mac with macOS 15+ and Xcode 26+ (Swift 6.2):

```bash
git clone https://github.com/mattmireles/kokoro-coreml && cd kokoro-coreml
swift run -c release kokoro-say "Hello from the Neural Engine."
afplay kokoro.wav
```

The first run builds the CLI (about a minute) and downloads the pinned ~210 MB starter model bundle from Hugging Face into `~/Library/Caches/kokoro-coreml`. Each download is checked against a SHA-256 before use. `kokoro-say` prints the time per stage and which path ran each one.

- **Options:** `--out speech.wav`, `--runs 3` (the first run includes one-time Core ML compilation). The starter bundle has one voice (`af_heart`) and one 15 s bucket, so every input is timed at that bucket; build the `full` bundle (README/SDK.md) and pass `--bundle` for the other voices and buckets.
- **Help:** `swift run -c release kokoro-say --help`.
- **If the build stops compiling Metal shaders** on a fresh Xcode, run `xcodebuild -downloadComponent MetalToolchain` once.

## Use it in your app

```swift
// Package.swift
.package(url: "https://github.com/mattmireles/kokoro-coreml", revision: "<commit>")
// target dependency
.product(name: "KokoroTTS", package: "kokoro-coreml")
```

```swift
import KokoroTTS

let tts = try await K#okoroTTS.load(resources: .huggingFace())   // download + verify once, then cached
let audio = try await tts.synthesize("Hello world.", voice: .afHeart)
let buffer = try audio.makePCMBuffer()                           // 24 kHz mono, ready for AVAudioEngine
```

iOS 18+ / macOS 15+. Text preparation, phonemization, chunking, model loading and fallbacks are handled for you. The full SDK guide, including bundled-resource mode and playback, is in [README/SDK.md](README/SDK.md). Pin the latest `main` commit.


## Python fallback

Don't need Core ML? The original Kokoro Python library works standalone (`pip install kokoro soundfile`):

```python
from kokoro import KPipeline
import soundfile as sf

pipeline = KPipeline(lang_code='a')
for i, (gs, ps, audio) in enumerate(pipeline('Hello world!', voice='af_heart')):
    sf.write(f'{i}.wav', audio, 24000)
```

## For contributors

<details>
<summary>Repository layout, exporting models, in-repo memory</summary>

```
kokoro-coreml/
  Package.swift              # Root SwiftPM manifest: KokoroTTS, KokoroPipeline, KokoroG2P, kokoro-say
  swift-tts/                 # KokoroTTS SDK (raw text -> speech) and kokoro-say
  swift/                     # KokoroPipeline runtime, kokoro-bench, tests
  scripts/                   # Exporters, HF release tooling (sdk_profiles.json), benchmarks
  kokoro/                    # Python TTS library (PyTorch reference)
  paper/                     # Surgical Inference paper, figures, frozen benchmark data
  README/                    # Wiki, Notes, Plans, Guides
```

- **Exports.** You don't need them to run anything: every shipped package is on Hugging Face. To re-export, use Python 3.12 and `uv sync`, then `uv run huggingface-cli download hexgrad/Kokoro-82M --local-dir kokoro-82m` for the weights. The exporters take `--weights kokoro-82m/kokoro-v1_0.pth --config kokoro-82m/config.json --out <dir>`: `scripts/export_duration_ane.py` and `scripts/export_f0ntrain_split.py` (split duration and F0Ntrain), `scripts/export_decoder_pre_ane.py` and `scripts/export_hybrid_generator.py`. The folded generator came from the lab probe `scripts/probe_generator_ane_fold.py`; see [ane-folded-generator.md](README/Notes/ane-folded-generator.md).
- **Tests.** `uv run --with pytest python -m pytest -q`, plus `swift test` in `swift/` and `swift-tts/`.
- **Releases.** The release section of [README/SDK.md](README/SDK.md).
- **In-repo memory.** For current belief before changing runtime, export or benchmark behavior, start at [README/Wiki/README.md](README/Wiki/README.md). Check it with `node scripts/memory-health.mjs --strict`.

</details>

---

Built on [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) by [hexgrad](https://huggingface.co/hexgrad), [StyleTTS 2](https://github.com/yl4579/StyleTTS2) by [@yl4579](https://huggingface.co/yl4579), [MisakiSwift](https://github.com/mattmireles/MisakiSwift), and Apple's coremltools.

Run it on your Mac. [Tell us how fast it is.](https://discord.gg/QuGxSWBfQy)
