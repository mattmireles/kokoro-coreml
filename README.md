# Kokoro 82M TTS - Surgically Optimized for Apple Silicon

**15 seconds of speech in 691ms on an M1 Mac Mini. 2.8x faster than Metal. Using Apple Neural Engine.**

[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) running natively on the Apple Neural Engine via CoreML. Five compiled models, one Swift pipeline, zero Python at inference time. Every Mac with Apple Silicon is a TTS server.

> **Pre-converted `.mlpackage` files:** [huggingface.co/mattmireles/kokoro-coreml](https://huggingface.co/mattmireles/kokoro-coreml)

## Why the Neural Engine?

Apple Silicon isn't one processor. It's three — **CPU, GPU, and the Neural Engine (ANE)** — each built for different work. The ANE devours fixed-shape convolutions and dense matrix math at a fraction of the power draw of the GPU. It's the same silicon that runs Face ID, Live Text, and on-device Siri.

But it has rules. No dynamic shapes. No data-dependent control flow. No `RangeDim` with multiple variable inputs. Most Core ML ports shove the whole model through and hope the runtime scheduler figures it out. It doesn't — you end up on CPU wondering why your "Neural Engine model" runs at 1x realtime.

This repo dissects the Kokoro TTS pipeline and makes deliberate cuts:

```
                    ┌──────────────────────────────┐
  "Hello world" ──▶ │  DURATION MODEL               │
                    │  BERT + LSTMs                  │ ◀── CPU/GPU
                    │  Sequential, variable-length   │     Best at: branching,
                    └───────────┬──────────────────┘     variable sequences
                                │
                                ▼
                    ┌──────────────────────────────┐
                    │  ALIGNMENT  (Swift / CPU)      │
                    │  Build matrix from durations   │ ◀── CPU
                    │  ~50 lines of code             │     Best at: small, complex
                    └───────────┬──────────────────┘     data-dependent logic
                                │
                                ▼
                    ┌──────────────────────────────┐
                    │  DECODER + VOCODER             │
                    │  Dense convolutions + iSTFT    │ ◀── Neural Engine (ANE)
                    │  Fixed shapes, pure math       │     Best at: dense parallel
                    └───────────┬──────────────────┘     tensor operations
                                │
                                ▼
                           24 kHz Audio
```

**Redesign the inference pipeline, not the model.** Give the ANE clean static tensors. Let the CPU handle the messy parts. That's where the 2.8x over Metal comes from — not by fighting the GPU, but by routing around it.

## Why not just use PyTorch MPS?

Even if you stay on the GPU, PyTorch MPS has two problems specific to this model:

1. **`aten::angle` doesn't exist on MPS.** The vocoder hits unsupported ops, forcing per-op CPU fallbacks. Every fallback is a round-trip data transfer that kills throughput.
2. **Python is the bottleneck.** Five `model.forward()` calls through the GIL and eager-mode dispatch. The interpreter overhead alone costs more than the M1 Mini's total inference time.

Swift+CoreML eliminates both: models run on the ANE with no fallback, orchestration is native Swift with no Python.

## Performance

An M1 Mac Mini with 16 GB of RAM — the cheapest Apple Silicon Mac you can buy — synthesizes 30 seconds of speech in 1.2 seconds. That's 22x realtime.

| Audio | M1 Mini (16 GB) | M2 Air (24 GB) | M2 Ultra (64 GB) |
| --- | --- | --- | --- |
| 3s | 157 ms | 200 ms | **59 ms** |
| 7s | 511 ms | 326 ms | **136 ms** |
| 15s | 691 ms | 783 ms | **278 ms** |
| 30s | **1,229 ms** | 1,829 ms | **422 ms** |

13-70x realtime across the lineup. The M2 Ultra finishes 30s of audio in 422 ms (70x RT), but the M1 Mini is the number that matters — it proves the pipeline ships on hardware people already own.

### vs alternatives

| Baseline | Speedup |
| --- | --- |
| PyTorch MPS (GPU) | **1.8-3.4x** faster |
| PyTorch CPU | **3.5-7.3x** faster |
| Python+CoreML hybrid | **1.1-2.0x** faster |

Counterbalanced ordering, 5 iterations, warm median. Full data: [bakeoff-results.md](README/Notes/bakeoff-results.md).

## Architecture

Five CoreML models chained with native Swift DSP. Text goes in, 24 kHz audio comes out. Zero Python at inference time.

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
Text → Phonemes → Duration (CoreML) → Alignment (Swift) → Matrix ops (Accelerate)
  → F0Ntrain (CoreML) → DecoderPre (CoreML) + hn-nsf (Swift) → GeneratorFromHar (CoreML) → Audio
```

Four models run on the ANE. One DSP stage (harmonic source) runs in Swift with double-precision phase accumulation. The generator has zero `nn.Linear` ops — all replaced with `Conv1d(kernel_size=1)` to keep it on the ANE path (48 → 0 linear ops in the MIL graph).

<details>
<summary>Full pipeline diagram and model inventory</summary>

```
Text --> Phonemes (tokenizer)
    |
    v
Duration CoreML [32/64/128/256/320/384/512 tokens] --> pred_dur, d, t_en, s, ref_s
    |
    v
Alignment (Swift) --> one-hot matrix from pred_dur
    |
    v
Matrix ops (Accelerate) --> en = d x alignment, asr = t_en x alignment
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
| `kokoro_f0ntrain_t{N}` | T=120, 400, 560, 1200, 2400 | en, s | F0_pred, N_pred | Pitch/noise from aligned features |
| `kokoro_decoder_pre_{N}s` | 3s, 7s, 10s, 15s, 30s | asr, f0, n_input, ref_s | x_pre | Decoder stack (Conv + AdaIN) |
| `kokoro_decoder_har_post_{N}s` | 3s, 7s, 10s, 15s, 30s | x_pre, ref_s, har | waveform | Generator (0 linear ops, all Conv1d for ANE) |

The Duration model uses per-size exports because E5RT (Apple Neural Engine runtime) cannot handle `RangeDim` or `EnumeratedShapes` with multiple variable inputs. The caller pads tokens to the nearest enumeration.

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
- **`WaveformPostProcess.swift`** -- punctuation-span fade-to-silence after generator trim
- **`BucketSelector.swift`** -- smallest bucket >= ceil(audio_seconds)

```swift
import KokoroPipeline

let pipeline = try KokoroPipeline(
    modelsDirectory: coremlURL,
    buckets: [3, 7, 10, 15, 30],
    linearWeights: hnsfWeights,
    linearBias: hnsfBias
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
- [ane-optimization-v1.md](README/Plans/ane-optimization-v1.md) -- Linear→Conv1d swap (48→0 MIL linear ops)
- [swift-prefix-rewrite-v1.md](README/Plans/swift-prefix-rewrite-v1.md) -- Swift pipeline architecture
- [debug-notes.md](README/Notes/debug-notes.md) -- active issues and resolved investigations
- [learnings.md](README/learnings.md) -- conversion challenges and solutions
- [CoreML-Compute-Unit-Scheduling-guide.md](README/Guides/apple-silicon/CoreML-Compute-Unit-Scheduling-guide.md)

</details>

---

Built on [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) by [hexgrad](https://huggingface.co/hexgrad), [StyleTTS 2](https://github.com/yl4579/StyleTTS2) by [@yl4579](https://huggingface.co/yl4579), and Apple's coremltools.

Run it on your Mac. [Tell us how fast it is.](https://discord.gg/QuGxSWBfQy)
