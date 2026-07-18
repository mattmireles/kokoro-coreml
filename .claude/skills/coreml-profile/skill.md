---
name: coreml-profile
description: Profile CoreML model execution to determine which compute units (ANE, GPU, CPU) are actually being used, detect silent fallback, and identify performance bottlenecks. Triggered by keywords like "profile", "which compute unit", "is ANE running", "why is it slow", "silent fallback".
---

# CoreML Profile

## Parent skill

Entry point for all Core ML work: [coreml](../coreml/SKILL.md). Use this skill
when the routed intent is **runtime placement** (ANE/GPU/CPU), not numerical parity.

## Repo gate

Read [references/repo-profiles.md](../references/repo-profiles.md). Requires
exported `.mlpackage` files and profiling tooling for the active profile.
**kokoro-coreml:** `coreml/` + `swift/.build/release/kokoro-bench`. **crossfade:**
use when Core ML artifacts exist; otherwise profile via `docs/benchmark.md` and
Instruments on the C++/MLX path first.

## Purpose

Answer the question: **"Where is my model actually running?"** Detect silent
fallback, measure per-model compute unit utilization, and identify performance
bottlenecks across ANE/GPU/CPU on Apple Silicon.

## Use When

- The user asks which compute units a CoreML model is using
- Performance is unexpectedly slow and the user suspects silent fallback
- The user wants to compare `.all` vs `.cpuAndGPU` vs `.cpuAndNeuralEngine`
- After exporting a new model and wanting to verify ANE utilization
- The user says "profile", "is ANE running", "which compute unit", "silent fallback"

## Do Not Use When

- The user wants to validate numerical correctness (use `coreml-validate`)
- The user wants to run the full bakeoff benchmark (use `bakeoff`)
- The user wants to export/convert a model (that's a different workflow)

## Prerequisites

- Xcode must be installed and selected: `sudo xcode-select -s /Applications/Xcode.app/Contents/Developer`
- For `powermetrics`: requires `sudo` (will prompt user)
- For `xctrace`: no special permissions needed
- Models must be exported (`.mlpackage` at any in-repo path; kokoro: usually `coreml/`)

## Reference Material

Before profiling, read these guides for context:

**crossfade (in-repo):**

- `README/Guides/Stateful-KV-caches-CoreML-guide.md` — Stateful KV, ANE residency,
  `powermetrics`, p50/p99 decode-loop measurement
- `CLAUDE.md` Part 5 — Validate/Profile/Iterate checklist

**kokoro-coreml profile or sibling `../kokoro-coreml/`:**

- `README/Guides/apple-silicon/CoreML-Compute-Unit-Scheduling-guide.md` — silent
  fallback, graph partitioning, verification
- `README/Notes/debug-notes.md` — institutional memory of past issues

Full index: [coreml/reference.md](../coreml/reference.md).

## Key Concepts

### Silent Fallback
`.all` is a **request**, not enforcement. CoreML may silently route ops to
CPU/GPU even when ANE is requested. The only proof is runtime telemetry.

### Graph Partitioning Overhead
When CoreML splits a model across ANE + CPU/GPU, each transition costs
~0.1-0.5ms of context-switching overhead. A model with many small ANE segments
interleaved with CPU fallback ops can be **slower** than pure CPU/GPU.

### ANE Memory Layout Rule
The ANE pads the last axis to a multiple of 64 bytes. If the last dimension is
small (e.g., channels=22), this wastes memory and bandwidth. Optimal layout:
`(Batch, Channels, 1, SequenceLength)` where SequenceLength is large.

## Procedure

### 1. Identify what to profile

Ask the user (or infer from context):
- Which model(s)? (e.g., `kokoro_decoder_har_post_30s.mlpackage`)
- Which compute unit configs to compare? (default: `.all` vs `.cpuAndGPU`)
- What input size? (bucket size matters for ANE behavior)

### 2. Quick power check (Level 0 — 30 seconds)

The fastest signal. Run inference in one terminal, powermetrics in another.

```bash
# Terminal 1: Start powermetrics (requires sudo)
sudo powermetrics -i 1000 --samplers ane -n 10
```

```bash
# Terminal 2: Run inference (Python example)
uv run python3 -c "
import coremltools as ct
import numpy as np

model = ct.models.MLModel('coreml/MODEL_NAME.mlpackage',
                           compute_units=ct.ComputeUnit.ALL)
# Build dummy input matching model spec
spec = model.get_spec()
inputs = {}
for inp in spec.description.input:
    shape = list(inp.type.multiArrayType.shape)
    inputs[inp.name] = np.random.randn(*shape).astype(np.float32)

# Run 20 iterations to sustain ANE activity
for i in range(20):
    model.predict(inputs)
print('Done')
"
```

**Interpretation:**
- `ANE Power: 0 mW` during inference → **silent fallback confirmed**
- `ANE Power: >0 mW` → ANE is doing *something* (but may still be partial)

### 3. Compute unit comparison (Level 1 — 2 minutes)

Compare wall time across compute unit configs to identify the fastest path.

```python
uv run python3 -c "
import coremltools as ct
import numpy as np
import time

MODEL = 'coreml/MODEL_NAME.mlpackage'

# Build dummy input
spec = ct.models.MLModel(MODEL).get_spec()
inputs = {}
for inp in spec.description.input:
    shape = list(inp.type.multiArrayType.shape)
    inputs[inp.name] = np.random.randn(*shape).astype(np.float32)

for cu_name, cu in [
    ('ALL', ct.ComputeUnit.ALL),
    ('CPU_AND_GPU', ct.ComputeUnit.CPU_AND_GPU),
    ('CPU_AND_NEURAL_ENGINE', ct.ComputeUnit.CPU_AND_NEURAL_ENGINE),
    ('CPU_ONLY', ct.ComputeUnit.CPU_ONLY),
]:
    model = ct.models.MLModel(MODEL, compute_units=cu)
    # Warmup
    for _ in range(3):
        model.predict(inputs)
    # Timed
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        model.predict(inputs)
        times.append((time.perf_counter() - t0) * 1000)
    median = sorted(times)[5]
    print(f'{cu_name:30s}  median={median:.1f}ms  min={min(times):.1f}ms  max={max(times):.1f}ms')
"
```

**Interpretation:**
- If `.cpuAndGPU` is faster than `.all` → ANE is hurting (graph partitioning overhead)
- If `.all` ≈ `.cpuOnly` → ANE/GPU not being used at all
- If `.cpuAndNeuralEngine` is very slow → ANE can't handle this model efficiently

### 4. xctrace profiling (Level 2 — 5 minutes)

For detailed per-op compute unit attribution, use Instruments via CLI.

```bash
# Record a 10-second trace of a running process
xctrace record --template "Core ML" \
  --output /tmp/coreml_profile.trace \
  --time-limit 10s \
  --launch -- uv run python3 YOUR_SCRIPT.py
```

Then open the trace in Instruments:
```bash
open /tmp/coreml_profile.trace
```

Look for:
- **Neural Engine track**: sustained activity = ANE is hot
- **Gaps in ANE track**: context switches back to CPU/GPU
- **Thread names**: `H11ANEServicesThread` (ANE), `Espresso::MPSEngine` (GPU),
  `Espresso::BNNSEngine` (CPU)

### 5. Swift binary profiling

For Config F (Swift pipeline), profile the benchmark binary directly:

```bash
# Profile the Swift benchmark on a specific input
xctrace record --template "Core ML" \
  --output /tmp/swift_profile.trace \
  --time-limit 30s \
  --launch -- swift/.build/release/kokoro-bench \
    --models-dir coreml \
    --inputs-dir outputs/swift_bench_inputs \
    --hnsf-weights outputs/swift_bench_inputs/hnsf_weights.json \
    --input-key 7s \
    --compute-units all
```

### 6. coremlcompiler analysis (Level 3 — static, no runtime needed)

Check what the compiler thinks about compute unit assignment:

```bash
xcrun coremlcompiler compile coreml/MODEL_NAME.mlpackage /tmp/compiled_model/
xcrun coremlcompiler generate coreml/MODEL_NAME.mlpackage /tmp/compiled_model/ --language Swift
```

This shows how CoreML plans to partition the graph at compile time. Inspect
the generated code or compiled model metadata for compute unit annotations.

## Output Template

```text
## CoreML Profile: [model name]

### Machine
- Chip: [e.g., M2 MacBook Air]
- RAM: [e.g., 24 GB]
- macOS: [version]

### Power Check
- ANE Power during inference: [X mW / 0 mW]
- Verdict: [ANE active / silent fallback]

### Compute Unit Comparison (median of 10, after 3 warmup)
| Config                  | Median (ms) | Min    | Max    |
| ----------------------- | ----------- | ------ | ------ |
| ALL                     |             |        |        |
| CPU_AND_GPU             |             |        |        |
| CPU_AND_NEURAL_ENGINE   |             |        |        |
| CPU_ONLY                |             |        |        |

### Fastest Config: [X] ([Y]x faster than ALL)

### Recommendation
- [Use .cpuAndGPU for this model on this hardware / ANE is working well / etc.]
```

## Anti-patterns

- **Assuming `.all` means ANE** — it doesn't. Always verify.
- **Profiling without warmup** — first prediction includes ANE plan compilation.
  Always warm up 2-3 iterations before timing.
- **Profiling on battery** — thermal throttling skews results. Plug in.
- **Comparing across compute units without controlling for model compilation** —
  each compute unit config triggers a different compilation. Warm each separately.
- **Single-iteration timing** — use median of 5-10 iterations minimum.

## Canonical References

- Master router: [coreml/SKILL.md](../coreml/SKILL.md)
- crossfade: `README/Guides/Stateful-KV-caches-CoreML-guide.md`, `CLAUDE.md` Part 5
- kokoro sibling: `../kokoro-coreml/README/Guides/apple-silicon/CoreML-Compute-Unit-Scheduling-guide.md`,
  `../kokoro-coreml/README/Notes/debug-notes.md`
- kokoro bakeoff: `scripts/bakeoff_harness.py`, `swift/Sources/KokoroBenchmark/main.swift`
