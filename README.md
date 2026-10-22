# Kokoro 82M TTS - Surgically Optimized for Apple Silicon

**30 seconds of speech in 1.6 s on an M1 Mac Mini, and in 287 ms on an M2 Ultra. 1.8-2.4x faster than MLX on the same Macs.**

[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) running natively on Apple Silicon via Core ML: four model families, one Swift pipeline, zero Python at inference time. Each stage runs on the part of the chip that suits it: the Neural Engine, the GPU, or the CPU.

> **Pre-converted `.mlpackage` files:** [huggingface.co/mattmireles/kokoro-coreml](https://huggingface.co/mattmireles/kokoro-coreml)

## Why split the model?

Apple Silicon isn't one processor. It's three — **CPU, GPU, and the Neural Engine (ANE)** — each built for different work. The ANE runs fixed-shape dense math at a fraction of the GPU's power draw. The GPU handles large convolutions the ANE rejects. The CPU handles small, data-dependent logic.

Each has rules. The ANE takes no dynamic shapes and no data-dependent control flow, and it rejects Kokoro's vocoder outright: the waveform tensors are too long for it. Most Core ML ports push the whole model through one policy and let the scheduler guess. It guesses badly.

This repo cuts the Kokoro pipeline at those seams:

```
  "Hello world"
       │
       ▼
  DURATION MODEL (Core ML)          ◀── GPU, LSTMs on CPU
       │
       ▼
  ALIGNMENT (Swift)                 ◀── CPU: small, data-dependent
       │
       ▼
  F0 + NOISE (Core ML)              ◀── GPU
       │
       ▼
  DECODER-PRE (Core ML)             ◀── Neural Engine
    ∥ HARMONIC SOURCE (Swift DSP)   ◀── CPU, runs at the same time
       │
       ▼
  GENERATOR / VOCODER (Core ML)     ◀── GPU
       │
       ▼
  24 kHz audio
```

**Redesign the inference pipeline, not the model.** A whole-model Core ML export doesn't even convert correctly. Split at the right seams, each piece runs where it runs best.

## Why not just use PyTorch MPS?

Even if you stay on the GPU, PyTorch MPS has three problems with this model:

1. **It doesn't finish long clips.** On every Mac we tested, PyTorch MPS fails 15 s and 30 s inputs with `Output channels > 65536 not supported at the MPS device`.
2. **Unsupported ops fall back to the CPU.** `aten::angle` has no MPS kernel, so the vocoder round-trips data to the CPU.
3. **Python adds overhead.** Five `forward()` calls through eager-mode dispatch pay interpreter costs this pipeline never pays.

Swift + Core ML removes all three: native orchestration, no op fallbacks, and every input length completes.

## Performance

An M1 Mac Mini with 16 GB of RAM — the cheapest Apple Silicon Mac you can buy — synthesizes 30 seconds of speech in 1.64 seconds. That's 17x realtime.

Warm median end-to-end time, October 2026 (current `main`, published models, fixed-bucket generator on Macs):

| Audio | M1 Mini (16 GB) | M2 Air (24 GB) | M2 Ultra (64 GB) | iPhone 15 Pro Max | iPhone 12 Pro |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3s | 205 ms | 123 ms | **45 ms** | 195 ms | 421 ms |
| 7s | 410 ms | 274 ms | **80 ms** | 417 ms | 917 ms |
| 10s | 590 ms | 397 ms | **112 ms** | 609 ms | 1,391 ms |
| 15s | 835 ms | 589 ms | **148 ms** | 881 ms | 1,984 ms |
| 30s | 1,636 ms | 1,176 ms | **287 ms** | 1,721 ms | 4,017 ms |

Every device runs faster than realtime: 7x on the 2020 iPhone 12 Pro, 17x on the M1 Mini, up to 96x on the M2 Ultra. Macs ran macOS 26.7.1 (M2 Ultra) and 27.0.1 (M2 Air, M1 Mini); iPhones ran iOS 27.0.1 with the flexible-length generator.

### vs alternatives

| Baseline | Speedup |
| --- | --- |
| MLX (mlx-audio 0.4.3) | **1.8-2.4x** faster |
| MLX Swift on iPhone (kokoro-ios 1.0.8) | **2.0-3.2x** faster; MLX runs out of memory on 30 s clips on the 4 GB iPhone 12 Pro |
| PyTorch MPS (GPU) | **2.7-6.5x** faster where MPS finishes (it fails 15 s and 30 s clips) |
| PyTorch CPU | **3.2-6.8x** faster |
| Python + Core ML hybrid | **1.3-2.9x** faster |

Same machines, same frozen inputs, warm medians. Protocol, raw-data paths and caveats: [paper-cs1-frozen-rerun-2026-10-20.md](README/Notes/paper-cs1-frozen-rerun-2026-10-20.md).

> **macOS 27 note:** on macOS 27.0.1, Apple's MPSGraph can crash on the flexible-length generator, so the runtime uses the fixed-bucket generators on OS 27 (`KOKORO_FLEXIBLE_GENERATOR=1` overrides). Details: [debug-notes.md](README/Notes/debug-notes.md) "Flexible generator aborts on OS 27".

## Architecture

Four Core ML model families chained with native Swift DSP. Text goes in, 24 kHz audio comes out. Zero Python at inference time.

## In-Repo Memory

For current belief before changing runtime, export, or bakeoff behavior, start
at [README/Wiki/README.md](README/Wiki/README.md). The wiki routes back to
canonical sources; it does not replace code, scripts, notes, or measured
outputs. Check it with:

```bash
node scripts/memory-health.mjs --write-coverage
node scripts/memory-health.mjs --strict
```

```
Text → Phonemes → Duration (CoreML) → Alignment (Swift: expand token vectors to frames)
  → F0Ntrain (CoreML) → DecoderPre (CoreML) ∥ hn-nsf (Swift) → GeneratorFromHar (CoreML) → Trim → Audio
```

Decoder-pre runs on the ANE; duration, F0 and the generator run on the GPU (the ANE compiler rejects the generator's waveform-length tensors). One DSP stage (harmonic source) runs in Swift with double-precision phase accumulation, concurrently with decoder-pre. The generator has zero `nn.Linear` ops — all replaced with `Conv1d(kernel_size=1)` (48 → 0 linear ops in the MIL graph).

<details>
<summary>Full pipeline diagram and model inventory</summary>

```
Text --> Phonemes (tokenizer)
    |
    v
Duration CoreML [32/64/128/256/320/384/512 tokens] --> pred_dur, d, t_en, s, ref_s
    |
    v
Alignment (Swift) --> expand d and t_en to frames by pred_dur (en, asr)
    |
    v
F0Ntrain CoreML --> F0_pred, N_pred (pitch + noise contours)
    |
    v
Pad to bucket geometry (Swift)
    |
    +--> DecoderPre CoreML --> x_pre (decoder features)
    |
    +--> hn-nsf (Swift/Accelerate, Double-precision phase) --> har (harmonics)
    |
    v
GeneratorFromHar CoreML --> waveform (24 kHz)
    |
    v
Trim (Swift) --> final audio
```

| Model | Sizes | Input | Output | Role |
| --- | --- | --- | --- | --- |
| `kokoro_duration_t{N}` | T=32, 64, 128, 256, 320, 384, 512 | input_ids, attention_mask, ref_s, speed | pred_dur, d, t_en, s, ref_s_out | BERT + prosody prediction |
| `kokoro_f0ntrain_t{N}` | T=120, 280, 400, 600, 1200 | en, s, mask | F0_pred, N_pred | Pitch/noise from aligned features |
| `kokoro_decoder_pre_{N}s` | 3s, 7s, 10s, 15s, 30s | asr, f0, n_input, ref_s, mask | x_pre | Decoder stack (Conv + AdaIN) |
| `kokoro_decoder_har_post_{N}s` | 3s, 7s, 10s, 15s, 30s | x_pre, ref_s, har, mask | waveform | Generator, fixed shape per bucket (GPU) |
| `kokoro_decoder_har_post_range` | 0.5 s granule up to 30 s | x_pre, ref_s, har, mask | waveform | Generator, one flexible-length program (GPU; skipped on OS 27) |

Duration, F0Ntrain and decoder-pre stay bucketed: the ANE rejects flexible shapes, an LSTM over a symbolic length runs about 5x slower on the CPU, and `EnumeratedShapes` costs about 140 ms per shape change. The caller pads to the nearest bucket and passes a `mask` so padding never reaches a statistic. Only the GPU generator ships as a flexible (`RangeDim`) program.
Newly exported padded-stage packages require the binary prefix `mask`; the runtime also recognizes legacy published packages that predate that input.

</details>

## Hear it yourself

```bash
bash scripts/setup_bakeoff.sh                     # deps, models, exports, Swift build (~10 min)
uv run python scripts/bakeoff_listen.py            # render WAVs through the Swift+CoreML pipeline
afplay outputs/bakeoff/listen/config_f_3s.wav      # listen
```

Three commands. WAVs land in `outputs/bakeoff/listen/`. Use `--skip-download` if models are already local.

### Run the benchmark

```bash
BAKEOFF_SKIP_SMOKE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 \
uv run python scripts/bakeoff_harness.py run \
  --configs a,d,e,f --iterations 5 --order-seed 0
```

<details>
<summary>Manual setup (step-by-step)</summary>

```bash
uv sync
uv run python scripts/download_models.py --coreml
uv run python export_duration.py
uv run python export_f0ntrain.py --t-frames 120 400 560 1200 2400
uv run python export_decoder_pre.py --buckets 3 7 10 15 30
uv run python -m export_synth.main --buckets "3s,7s,10s,15s,30s" --mode decoder-har --output_dir coreml
cd swift && swift build -c release --product kokoro-bench && cd ..
uv run python scripts/bakeoff_harness.py prepare-inputs
uv run python scripts/prepare_swift_bench_inputs.py
```

</details>

## Swift Package

The `swift/` directory contains a Swift Package (`KokoroPipeline`) with:

- **`KokoroPipeline.swift`** -- 9-stage orchestrator with per-stage timing
- **`HarmonicSource.swift`** -- hn-nsf in Swift/Accelerate (Double-precision phase accumulator)
- **`AlignmentBuilder.swift`** -- one-hot alignment matrix from phoneme durations
- **`MLMultiArrayHelpers.swift`** -- matrix multiply (cblas_sgemm), zero-padding, stride-safe MLMultiArray ops
- **`MultifunctionPackages.swift`** -- loads the bucketed stages from one multifunction package each (macOS 15 / iOS 18)
- **`CompiledModelCache.swift`** -- optional on-disk cache of compiled models, so a relaunch skips the compile
- **`BucketSelector.swift`** -- smallest bucket >= ceil(audio_seconds)

```swift
import KokoroPipeline

let pipeline = try KokoroPipeline(
    modelsDirectory: coremlURL,
    buckets: [3, 7, 10, 15, 30],
    linearWeights: hnsfWeights,
    linearBias: hnsfBias,
    compiledModelCache: cacheURL  // optional; nil recompiles on every launch
)

let result = try pipeline.synthesize(
    inputIds: tokenIds,
    attentionMask: mask,
    refS: voiceEmbedding,
    speed: 1.0
)

// result.audio: [Float] at 24 kHz
// result.timings: per-stage breakdown
// result.timings.total: end-to-end wall time
```

`init` only finds the model packages. Each model compiles and opens the first time it's used. Call `prepareForBucket(bucketSec:tFrames:)` to do that ahead of time. Point `compiledModelCache` at a directory whose name comes from the model set's digest, so that updated models never reuse stale compiled bundles.

## Bakeoff configs

<details>
<summary>Config details (A through F)</summary>

| Config | What it measures | Pipeline | Status |
| --- | --- | --- | --- |
| **F** | Swift + CoreML (winner) | 5 CoreML models + Swift hn-nsf DSP | **Production** |
| **A** | Python HAR-post hybrid | PyTorch prefix + CoreML GeneratorFromHar | Baseline |
| **D** | PyTorch MPS (GPU with CPU fallback) | Full PyTorch on MPS device | Comparison |
| **E** | PyTorch CPU | Full PyTorch on CPU | Comparison |
| B/C | ANE participation (diagnostic) | Decoder-only CoreML under .all / .cpuAndGPU | Diagnostic |

The bakeoff harness uses a persistent batch subprocess for Config F — models compile once at startup and stay cached across all iterations. See [bakeoff-results.md](README/Notes/bakeoff-results.md) for the full cross-machine matrix.

</details>

## Python fallback

Don't need CoreML? The original Kokoro Python library works standalone:

```python
from kokoro import KPipeline
import soundfile as sf

pipeline = KPipeline(lang_code='a')
for i, (gs, ps, audio) in enumerate(pipeline('Hello world!', voice='af_heart')):
    sf.write(f'{i}.wav', audio, 24000)
```

For MPS GPU acceleration: `PYTORCH_ENABLE_MPS_FALLBACK=1 python your_script.py`

## Deep dive

<details>
<summary>Repository structure</summary>

```
kokoro-coreml/
  coreml/                          # CoreML .mlpackage files (downloaded from HF)
  swift/                           # Swift Package (KokoroPipeline)
    Sources/KokoroPipeline/        # Pipeline library (production)
    Sources/KokoroBenchmark/       # Benchmark CLI with batch mode + model cache
  kokoro/                          # Python TTS library (PyTorch)
  export_duration.py               # Duration model export
  export_f0ntrain.py               # F0Ntrain export
  export_decoder_pre.py            # DecoderPre export
  export_synth/                    # GeneratorFromHar export
  scripts/
    setup_bakeoff.sh               # One-command setup
    bakeoff_harness.py             # Benchmark harness
    bakeoff_listen.py              # Render WAVs from benchmark inputs
    download_models.py             # HF Hub downloader
```

</details>

<details>
<summary>Documentation index</summary>

- [bakeoff-results.md](README/Notes/bakeoff-results.md) -- v5 cross-machine comparison (M2 Ultra, M2 Air, M1 Mini)
- [performance-notes.md](README/Notes/performance-notes.md) -- full bakeoff history, stage breakdowns
- [002-ane-optimization-plan.md](README/Plans/002-ane-optimization-plan.md) -- Linear→Conv1d swap (48→0 MIL linear ops)
- [004-swift-prefix-rewrite-plan.md](README/Plans/004-swift-prefix-rewrite-plan.md) -- Swift pipeline architecture
- [debug-notes.md](README/Notes/debug-notes.md) -- active issues and resolved investigations
- [learnings.md](README/learnings.md) -- conversion challenges and solutions
- [CoreML-Compute-Unit-Scheduling-guide.md](README/Guides/apple-silicon/CoreML-Compute-Unit-Scheduling-guide.md)

</details>

---

Built on [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) by [hexgrad](https://huggingface.co/hexgrad), [StyleTTS 2](https://github.com/yl4579/StyleTTS2) by [@yl4579](https://huggingface.co/yl4579), and Apple's coremltools.

Run it on your Mac. [Tell us how fast it is.](https://discord.gg/QuGxSWBfQy)
