# Kokoro TTS Bakeoff: ANE Surgery vs GPU Baselines

## Objective

Empirically measure whether ANE-optimized Kokoro-82M outperforms GPU-based inference, and by how much. Verify that naive CoreML conversion silently falls back to CPU (not ANE). Produce publication-ready benchmark data.

---

## Hardware & Environment

### Two test machines — run ALL configs on BOTH machines.

```
Machine 1:      Mac Studio, M2 Ultra, 64GB unified memory
                - 24-core CPU (16P + 8E)
                - 76-core GPU
                - 32-core Neural Engine
                - 800 GB/s memory bandwidth

Machine 2:      Mac Mini, M1, 16GB unified memory
                - 8-core CPU (4P + 4E)
                - 8-core GPU  
                - 16-core Neural Engine
                - 68.25 GB/s memory bandwidth
```

Record on EACH machine:
```
macOS:          (record exact version: `sw_vers`)
Xcode:          (record exact version: `xcodebuild -version`)
coremltools:    (record exact version: `pip show coremltools`)
Python:         3.11+ (for PyTorch baselines)
Swift:          (record exact version: `swift --version`)
Chip:           (record: `sysctl -n machdep.cpu.brand_string`)
Memory:         (record: `sysctl -n hw.memsize`)
```

**Why two machines matter for the paper:**
- M2 Ultra is a high-end workstation chip. Fast at everything. A speedup here could be dismissed as "the machine is just fast."
- M1 Mini is the base consumer chip. 16GB RAM, 8 GPU cores, dramatically less memory bandwidth. If ANE surgery delivers a proportionally LARGER win on M1 — because the GPU is weaker but the ANE is still capable — that's the strongest possible argument for the paper's thesis. It means ANE surgery democratizes performance across the Apple Silicon lineup.
- Conversely, if the M1 Mini can't fit certain configs in 16GB, that's also data: ANE surgery may enable workloads that don't fit in the GPU's working set on constrained devices.

**Before any benchmarks (on EACH machine):**
- Close all apps except Terminal
- Plug in power (no battery, relevant for Mac Mini power management)
- Disable Spotlight indexing: `sudo mdutil -i off /`
- Wait 2 minutes after boot for background tasks to settle
- Record thermal state: `sudo powermetrics --samplers smc -n 1`
- Record available memory: `vm_stat | head -5`

---

## Configurations to Test

### Config A: ANE Surgical (existing kokoro-coreml)

- **Source:** github.com/mattmireles/kokoro-coreml
- **What it is:** Duration model on CPU/GPU, alignment in Swift, decoder on ANE with bucketed static shapes, float16
- **Expected behavior:** Decoder executes on ANE

### Config B: CoreML Naive (compute_units=ALL)

- **What to build:** Export the FULL Kokoro pipeline as a single `.mlpackage` with `compute_units=coremltools.ComputeUnit.ALL`
- **Purpose:** Show what happens when you don't do surgery. Hypothesis: CoreML scheduler will fail to place decoder on ANE due to dynamic shapes / unsupported ops, and will silently fall back to CPU or GPU
- **This is the critical baseline.** If this is fast, the paper is weak. If this is slow, the paper is strong.

### Config C: CoreML GPU-only (compute_units=CPU_AND_GPU)

- **What to build:** Export the FULL Kokoro pipeline as a single `.mlpackage` with `compute_units=coremltools.ComputeUnit.CPU_AND_GPU`
- **Purpose:** Best-case GPU performance via CoreML, ANE explicitly excluded. Isolates GPU vs ANE comparison.

### Config D: PyTorch MPS (Metal Performance Shaders)

- **What to build:** Run the original Kokoro-82M PyTorch model with `device="mps"` (Apple's Metal backend for PyTorch)
- **Source:** github.com/hexgrad/Kokoro or huggingface.co/hexgrad/Kokoro-82M
- **Purpose:** What a developer gets if they just run the PyTorch model on Mac with GPU acceleration. No CoreML, no conversion, no surgery. This is the "path of least resistance" baseline.

### Config E: PyTorch CPU-only

- **What to build:** Same PyTorch model with `device="cpu"`
- **Purpose:** Floor baseline. Worst case. Also useful for confirming that Config B's naive CoreML is falling back to CPU (if their times are similar, that's evidence of CPU fallback).

---

## Test Inputs

Use the SAME text inputs across all configurations. Do not vary text between configs.

### Sequence Lengths

Generate 7 test inputs spanning short to long utterances:

```python
TEST_INPUTS = {
    "tiny":    "Hello.",                                          # ~1s output
    "short":   "The quick brown fox jumps over the lazy dog.",    # ~3s output  
    "medium":  """Machine learning models are becoming increasingly 
                capable of generating human-like speech from text 
                input, with applications spanning accessibility, 
                education, and virtual assistants.""",             # ~8s output
    "long":    # 3-4 sentences, ~15s output
               """The development of neural text-to-speech systems 
                has accelerated dramatically in recent years. Modern 
                architectures combine transformer-based sequence 
                modeling with convolutional vocoders to produce 
                natural-sounding audio. These systems can now clone 
                voices from just a few seconds of reference audio, 
                opening new possibilities for personalized speech 
                synthesis.""",
    "longer":  # 5-6 sentences, ~25s output  
               """(Compose a paragraph of approximately 80 words 
                that produces ~25 seconds of audio. Record the 
                exact text used.)""",
    "longest": # Target ~45s output
               """(Compose a paragraph of approximately 150 words
                that produces ~45 seconds of audio. Record the
                exact text used.)""",
    "max":     # Target ~60s output  
               """(Compose a paragraph of approximately 200 words
                that produces ~60 seconds of audio. Record the
                exact text used.)"""
}
```

**Important:** After the first run, record the ACTUAL output audio duration for each input. The relationship between text length and audio duration is what we're characterizing.

### Voice Preset

Use the same Kokoro voice preset for all tests. Pick one and lock it:
- Recommended: `af_heart` (or whichever preset was used in the existing benchmarks)
- Record which preset was used

---

## Measurement Protocol

### Warmup

For each configuration, before measuring:
1. Run the model once to trigger compilation / shader caching / model loading
2. Wait 5 seconds
3. Run 3 additional warmup iterations (discard results)
4. Wait 2 seconds

### Measurement

For each (configuration, input) pair:
1. Run **20 iterations**
2. For each iteration, record:
   - `wall_time_seconds`: total time from text input to audio output (end-to-end)
   - `audio_duration_seconds`: length of generated audio in seconds
3. Calculate:
   - `rtf = wall_time_seconds / audio_duration_seconds` (real-time factor; lower is better)
   - `speed_vs_realtime = audio_duration_seconds / wall_time_seconds`
4. Report: mean, median, std dev, min, max across 20 iterations

### For Config A (ANE Surgical) ONLY — additional stage-level timing:

Record per-stage breakdown for each iteration:
- `t_duration_model`: time for duration prediction (CPU/GPU)
- `t_alignment`: time for alignment matrix construction (Swift/CPU)
- `t_ane_predict`: time for decoder ANE prediction
- `t_istft`: time for iSTFT post-processing
- `t_orchestration`: total wall time minus sum of above (this is overhead)
- `bucket_used`: which decoder bucket was selected (3s, 5s, 10s, 15s, 30s)
- `padding_ratio`: (bucket_size - actual_size) / bucket_size

---

## Implementation Guide

### Building Config B (CoreML Naive)

```python
"""
Export full Kokoro as single CoreML model with compute_units=ALL.
The key is to NOT split the model. Export the entire pipeline 
as one traced/scripted model.
"""
import torch
import coremltools as ct

# 1. Load the full Kokoro PyTorch model
#    (Refer to hexgrad/Kokoro-82M for model loading code)

# 2. Trace or script the full forward pass
#    Input: token_ids (phoneme tokens), style embedding
#    Output: waveform

# 3. Convert to CoreML
model_ct = ct.convert(
    traced_model,
    inputs=[
        ct.TensorType(name="tokens", shape=tokens_shape),
        # ... other inputs as needed
    ],
    compute_units=ct.ComputeUnit.ALL,  # Let scheduler decide
    minimum_deployment_target=ct.target.macOS13,
    compute_precision=ct.precision.FLOAT16,
)
model_ct.save("kokoro_naive_all.mlpackage")

# IMPORTANT: Record any warnings during conversion.
# CoreML will warn about ops that can't run on ANE.
# These warnings ARE data for the paper.
```

**Expected issues during conversion:**
- LSTM layers may trigger warnings about ANE incompatibility
- Dynamic shapes in the alignment stage will likely fail
- If conversion fails entirely, that itself is a finding — document the error

**If full-model conversion fails:** Try converting just the decoder portion as a single model with dynamic shapes (no bucketing). This isolates whether the problem is the full pipeline or specifically the decoder's shape dynamism.

### Building Config C (CoreML GPU-only)

Same as Config B but with:
```python
compute_units=ct.ComputeUnit.CPU_AND_GPU  # Explicitly exclude ANE
```

### Building Config D (PyTorch MPS)

```python
import torch

device = torch.device("mps")  # Metal Performance Shaders

# Load model and move to MPS
model = load_kokoro_model()
model = model.to(device)
model.eval()

# Run inference
with torch.no_grad():
    # Ensure all input tensors are on MPS device
    tokens = tokens.to(device)
    style = style.to(device)
    
    start = time.perf_counter()
    waveform = model(tokens, style)
    # IMPORTANT: synchronize before stopping timer
    torch.mps.synchronize()
    end = time.perf_counter()
    
    wall_time = end - start
```

**Critical:** You MUST call `torch.mps.synchronize()` before stopping the timer. MPS dispatch is asynchronous — without sync, you're measuring dispatch time, not compute time.

### Building Config E (PyTorch CPU)

Same as Config D but with:
```python
device = torch.device("cpu")
# No sync needed — CPU execution is synchronous
```

---

## Instruments Profiling (Config A and B only)

This is separate from the benchmark runs. Do ONE profiling run per config.
**Run Instruments on M2 Ultra** (it will have Xcode installed). If M1 Mini also has Xcode, run Instruments there too — comparing the compute unit assignment across chip generations is valuable.

### Setup

1. Open Instruments (from Xcode → Open Developer Tool → Instruments)
2. Choose the **Core ML** template
3. Target: your benchmark Swift app (Config A) or Python process (Config B)
4. Record a single inference pass

### What to Capture

For EACH configuration, record:

1. **Compute Unit Assignment:** For each neural network operation, which compute unit did it actually execute on? (ANE, GPU, CPU)
   - Screenshot the Core ML Instruments trace showing compute unit colors
   - Count: how many ops on ANE vs GPU vs CPU
   
2. **For Config A (Surgical):** Confirm decoder ops are green (ANE). Duration model ops should be blue (GPU) or orange (CPU).

3. **For Config B (Naive):** Document WHERE ops land. Hypothesis: most/all ops land on CPU or GPU, NOT on ANE. If ANY decoder ops land on ANE, that weakens the paper's thesis — document honestly.

4. **Neural Engine utilization:** Is ANE at 0% during Config B? If so, screenshot that. It's the smoking gun.

### Export

- Export the Instruments trace as `.trace` file (for reproducibility)
- Take screenshots of:
  - The compute unit assignment view (colored bars)
  - The timeline view showing GPU/CPU/ANE utilization
  - Any warnings or fallback indicators

---

## Output Format

### Raw Data

Save all results as a single JSON file PER MACHINE:

Filenames: `results_m2_ultra.json` and `results_m1_mini.json`

```json
{
  "metadata": {
    "machine": "Mac Studio M2 Ultra 64GB",
    "chip": "Apple M2 Ultra",
    "gpu_cores": 76,
    "ane_cores": 32,
    "memory_gb": 64,
    "memory_bandwidth_gbs": 800,
    "macos_version": "...",
    "xcode_version": "...",
    "coremltools_version": "...",
    "pytorch_version": "...",
    "date": "2026-04-XX",
    "kokoro_version": "...",
    "voice_preset": "af_heart",
    "warmup_iterations": 3,
    "measurement_iterations": 20
  },
  "results": {
    "config_a_ane_surgical": {
      "tiny": {
        "audio_duration_s": 1.2,
        "bucket_used": "3s",
        "iterations": [
          {
            "wall_time_s": 0.45,
            "rtf": 0.375,
            "t_duration_model": 0.08,
            "t_alignment": 0.001,
            "t_ane_predict": 0.12,
            "t_istft": 0.01,
            "t_orchestration": 0.239,
            "padding_ratio": 0.60
          }
          // ... 19 more iterations
        ],
        "summary": {
          "wall_time_mean": null,
          "wall_time_median": null,
          "wall_time_std": null,
          "rtf_mean": null,
          "speed_vs_realtime_mean": null
        }
      }
      // ... other sequence lengths
    },
    "config_b_coreml_naive_all": { /* same structure minus stage breakdown */ },
    "config_c_coreml_gpu_only": { /* same structure */ },
    "config_d_pytorch_mps": { /* same structure */ },
    "config_e_pytorch_cpu": { /* same structure */ }
  },
  "conversion_logs": {
    "config_b_warnings": ["list of CoreML conversion warnings"],
    "config_b_errors": ["list of any conversion errors"],
    "config_c_warnings": ["..."]
  },
  "instruments": {
    "config_a_compute_units": {
      "ane_ops": null,
      "gpu_ops": null,
      "cpu_ops": null,
      "screenshot_paths": []
    },
    "config_b_compute_units": {
      "ane_ops": null,
      "gpu_ops": null,
      "cpu_ops": null,
      "screenshot_paths": []
    }
  }
}
```

### Summary Tables (generate from raw data)

**One table per machine:**

```
=== M2 Ultra (64GB, 76 GPU cores, 32 ANE cores) ===
| Config     | Input   | Audio (s) | Wall Time (s) | RTF    | Speed  |
|------------|---------|-----------|---------------|--------|--------|
| A: ANE     | tiny    |           |               |        |        |
| A: ANE     | short   |           |               |        |        |
| ...        | ...     |           |               |        |        |

=== M1 Mini (16GB, 8 GPU cores, 16 ANE cores) ===
| Config     | Input   | Audio (s) | Wall Time (s) | RTF    | Speed  |
|------------|---------|-----------|---------------|--------|--------|
| A: ANE     | tiny    |           |               |        |        |
| ...        | ...     |           |               |        |        |
```

**Cross-machine comparison table:**

```
| Config     | Input   | M2 Ultra RTF | M1 Mini RTF | Ratio (M2/M1) |
|------------|---------|-------------|-------------|----------------|
| A: ANE     | medium  |             |             |                |
| D: MPS     | medium  |             |             |                |
```

This cross-machine table reveals whether ANE surgery degrades more
gracefully than GPU across chip tiers.

---

## Decision Gates

After collecting results from BOTH machines, evaluate:

### Gate 1: Does naive CoreML fall back to CPU?

- Compare Config B (naive ALL) wall times to Config E (PyTorch CPU) **on each machine**
- If they are within 20% of each other on both machines → evidence of CPU fallback → paper thesis supported
- Check Instruments: are ANE ops at 0% for Config B? (Run Instruments on M2 Ultra — it has Xcode)
- If Config B is actually fast and using ANE → paper thesis is wrong, stop here

### Gate 2: Does ANE surgery provide meaningful speedup?

- Compare Config A (ANE surgical) to Config D (PyTorch MPS, the "developer default") **on each machine**
- Calculate speedup ratio for each machine separately:
  - If Config A is >= 2x faster on BOTH machines → strong paper
  - If Config A is >= 2x faster on M1 but only 1.3x on M2 Ultra → interesting story: ANE surgery matters MORE on constrained hardware
  - If Config A is within 1.3x of Config D on BOTH machines → weak paper, speedup doesn't justify surgery complexity

### Gate 3: Does the advantage hold across sequence lengths?

- Plot RTF vs audio duration for all configs, **one chart per machine**
- Does the ANE advantage grow, shrink, or stay constant with longer outputs?
- If it grows → good story (ANE scales better)
- If it shrinks → the orchestration overhead is amortized but ANE's relative advantage diminishes
- If it's constant → clean, simple story

### Gate 4: What percentage of time is orchestration overhead?

- From Config A stage breakdowns: t_orchestration / wall_time across all inputs
- Compare orchestration overhead between M2 Ultra and M1 Mini — is it the same absolute time, or proportional to chip speed?
- If overhead is > 50% for short inputs but < 20% for long inputs → interesting crossover story
- If overhead is > 40% everywhere → the surgery is leaving performance on the table, future work is reducing this

### Gate 5 (NEW): Does ANE advantage scale differently across chip tiers?

This is the cross-machine analysis that makes two machines more than twice as useful as one.

- Calculate for each sequence length: `ANE_speedup_M1 / ANE_speedup_M2`
- If ratio > 1.0 → ANE surgery is MORE valuable on weaker chips → strong argument for democratizing performance
- If ratio ≈ 1.0 → consistent advantage regardless of chip → clean generalizability story
- If ratio < 1.0 → ANE surgery is only worth it on high-end chips → weaker story

Key hypothesis: M1 Mini has 8 GPU cores but 16 ANE cores. M2 Ultra has 76 GPU cores but 32 ANE cores. The GPU/ANE ratio is dramatically different:
- M1: 8 GPU / 16 ANE = 0.5 (ANE has 2x the cores of GPU)
- M2 Ultra: 76 GPU / 32 ANE = 2.375 (GPU has 2.4x the cores of ANE)

If the GPU is relatively weak (M1), routing work to ANE should provide a bigger relative win. This is the most interesting prediction the paper can make, and if the data confirms it, it's a genuine contribution.

---

## Stretch Goals (if time permits)

### S1: Power Consumption

```bash
# Record power during benchmark run (on EACH machine)
sudo powermetrics --samplers smc,gpu,ane -i 100 -n 300 > power_log.txt
```

Compare watts consumed during Config A (ANE) vs Config D (MPS) on each machine. If ANE uses significantly less power for similar or better throughput, that's a second contribution: efficiency, not just speed. This matters especially on M1 Mini which is a low-power device.

### S2: Audio Quality Verification

Run the same text through Config A and Config D on M2 Ultra. Compare outputs:
- Mel spectrogram distance between PyTorch and ANE outputs
- Informal listening test: can you tell the difference?
- If CoreML float16 conversion introduces artifacts, document them

### S3: Memory Pressure Test (M1 Mini only)

The M1 Mini has only 16GB. During Config D (PyTorch MPS), monitor memory:
```bash
# In a separate terminal during benchmark
vm_stat 1 | head -60 > memory_pressure_log.txt
```

If PyTorch MPS causes memory pressure / swap on 16GB but the surgical ANE approach doesn't (because smaller models are loaded individually), that's a powerful finding for resource-constrained devices.

---

## Estimated Timeline

| Task | Time |
|------|------|
| Set up environment on BOTH machines, clone repos | 2 hours |
| Build Config D and E (PyTorch baselines) on both machines | 3-4 hours |
| Build Config B and C (CoreML naive exports) | 3-4 hours (conversion may fight you) |
| Write benchmark harness (reusable across machines) | 2-3 hours |
| Run all benchmarks on M2 Ultra (5 configs × 7 inputs × 20 iters) | 2-3 hours |
| Run all benchmarks on M1 Mini (same matrix) | 3-4 hours (slower machine) |
| Instruments profiling on M2 Ultra (2 configs) | 1-2 hours |
| Compile results into JSON + summary tables + cross-machine analysis | 2 hours |
| **Total** | **~20-24 hours of work across both machines** |

**Run M2 Ultra and M1 Mini benchmarks in parallel if possible** — the benchmark harness should be the same script, just pointed at different hardware.

**Most likely blocker:** Config B CoreML conversion may fail or require debugging. If the full pipeline can't be exported as a single model, document the failure mode — that itself is evidence for the paper. Also: the M1 Mini may struggle with PyTorch MPS for certain configs due to 16GB RAM. If it runs out of memory, DOCUMENT THAT — it strengthens the case for lightweight ANE-optimized models on constrained hardware.
