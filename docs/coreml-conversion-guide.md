# Core ML Conversion Field Manual (ANE‑First Edition)

**Last updated:** 10 Jul 2025
**Audience:** ML / iOS engineers who need production‑grade, high‑performance models on Apple Neural Engine devices.

---

## Quick Reference (keep this page handy)

* **ANE rule #1:** FP16 weights + ( B, C, 1, S ) tensor layout or you *will* fall back.
* **ANE rule #2:** *One* unsupported op in a model usually moves the *rest* to GPU/CPU.
* **ct.convert() essentials:** `inputs=[…]`, `compute_precision=ct.precision.FLOAT16`, `convert_to="mlprogram"`, `minimum_deployment_target=ct.target.iOS17`.
* **Tracing hack:** Trace PyTorch in FP32 → convert to FP16.
* **Performance proof:** Instruments → look for `Espresso::ANERuntimeEngine` **or** run `sudo powermetrics` → non‑zero *ANE Power*.
* **Peak speed:** W8A8 (8‑bit weights & activations) on A17 Pro / M4.

---

## Part 1 – The ANE Mental Model

### 1.1 Beyond Marketing Hype

The Apple Neural Engine is a *fixed‑function* NPU that trades flexibility for throughput & power efficiency. First shipped with **A11 (2017, 0.6 TOPS)** → **M4 (2024, 38 TOPS)**, its ISA is undocumented and only accessible through Core ML. Treat it like a **GPU‑sized black box**: amazing when you fit, useless when you miss.

### 1.2 Core ML’s Compute Hierarchy

Core ML plans every prediction across **CPU → GPU → ANE**. If *any* op is ANE‑ineligible, tensors stay on the fallback device to avoid copy overhead. ANE↔CPU copies are cheaper than ANE↔GPU, so partial ANE+CPU is possible; ANE+GPU almost never happens.

### 1.3 `mlprogram` ≻ `neuralnetwork`

Modern features (flexible shapes, multi‑function models, stateful buffers, quantization metadata) *only* exist in **`mlprogram`** inside an `.mlpackage`. Target iOS 15/macOS 12+ or bust.

### 1.4 The Golden Handcuffs

* **Precision:** FP16 or lower.
* **Layout:** 4‑D `(B,C,1,S)`; last dim must be contiguous & 64‑byte aligned → avoid size‑1 last axis.
* **Memory:** L2 ≈ 32 MB; spill to DRAM kills perf.

---

## Part 2 – The Core ML Tools Workflow

### 2.1 `ct.convert()` Parameters That Matter

```python
mlmodel = ct.convert(
    traced_model,
    source="pytorch",
    inputs=[ct.TensorType(shape=(1, 3, 224, 224))],
    compute_precision=ct.precision.FLOAT16,
    minimum_deployment_target=ct.target.iOS17,
)
```

`compute_units` is *runtime* guidance (debugging only). Always pin a target OS → gets correct op set & defaults to `mlprogram`.

### 2.2 PyTorch → TorchScript → Core ML

```python
traced = torch.jit.trace(model.float(), example)
```

Trace in FP32; cast back later. *Use real data* so dynamic shapes are captured.

### 2.3 Defining Inputs/Outputs

* **Images:** `ct.ImageType` with `scale`/`bias` embeds preprocessing.
* **Tensors:** `ct.TensorType`; use `RangeDim` for variable sequence lengths.

### 2.4 Top Four Conversion Pitfalls

1. **Tracing failure on FP16 CPU ops** → trace in FP32.
2. Missing `inputs=[…]` → “unable to infer input dims”.
3. Unsupported op message → see Part 3.
4. ONNX route for >1 GB models → often crashes; convert directly.

---

## Part 3 – ANE Compatibility & Model Surgery

### 3.1 Unsupported Ops Cheat Sheet (full table → Appendix A)

* **Broadcastable/ND Add, Concat, Mul** → replace with plain Add/Concat/Multiply.
* **`gather` / fancy indexing** → redesign model.
* **Dilated convs >1** → split into multiple convs.
* **Custom layers** → never ANE, plan split.

### 3.2 Surgery I: Layer Swaps

Walk `spec.functions["main"].block.operations`; clone & replace offending ops, save new spec.

### 3.3 Surgery II: Composite Ops

Register a Python MIL builder to decompose the op:

```python
@register_torch_op
def logical_or(ctx, node):
    x, y = ctx[node.inputs]
    return mb.logical_or(x=x, y=y, name=node.name)
```

### 3.4 Surgery III: Model Splitting

Use `coremltools.models.utils.bisect_model` → chain sub‑models in Swift; run ANE parts with `.all`, CPU/GPU parts with `.cpuAndGPU`.

### 3.5 Custom Layers

Implement `MLCustomLayer` in Swift and accept **zero** ANE acceleration.

---

## Part 4 – Quantization & Compression

### 4.1 The Spectrum

| Tech                        | Accuracy hit | Size ↓         | Effort  |
| --------------------------- | ------------ | -------------- | ------- |
| **FP16**                    | \~0 %        | 2×             | trivial |
| **W8A16 PTQ**               | <1 %         | 4×             | easy    |
| **W8A8 PTQ w/ calibration** | 1‑3 %        | 4×             | medium  |
| **GPTQ (4‑bit)**            | 2‑5 %        | 8×             | medium  |
| **Pruning (sparse)**        | 0‑4 %        | 1.5‑3× *extra* | medium  |
| **QAT**                     | 0‑1 %        | 4‑8×           | hard    |

### 4.2 Data‑Free PTQ

```python
import coremltools.optimize.coreml as cto
fp16_pkg = cto.linear_quantize_weights(pkg)
```

### 4.3 Calibration PTQ

```python
cto.experimental.linear_quantize_activations(pkg, calib_data)
```

Needs \~128 representative samples.

### 4.4 Why W8A8 Matters

A17 Pro & M4 have direct int8 GEMM paths → 30‑50 % latency win over W8A16.

---

## Part 5 – Case Studies (Patterns > Stories)

### 5.1 Whisper (STT)

* Use `whisper.cpp` → `generate-coreml-model.sh` (needs `ane_transformers`).
* Force `use_sdpa=False` to avoid unsupported PyTorch SDPA.
* Script replaces `nn.Linear` with `nn.Conv2d` + reshapes tensors to `(B,C,1,S)`.

### 5.2 Text‑to‑Speech

**Kokoro / Sesame**

* Heavy Python pre‑proc (G2P) → must port to Swift/C++.
* Expect manual shape fixes (e.g., speed‑tensor broadcast bug).
* For Sesame CSM, architecture so custom that only MIL hand‑rewrite seems viable.

### 5.3 Multimodal LLMs

**Gemma 3n** → Use Google MediaPipe; coremltools lacks MatFormer & PLE support.
**Phi‑3/4** → Direct PyTorch conversion succeeds *after*:

1. Trace FP32.
2. Patch `bitwise_or_` → `logical_or`.
3. Convert with `compute_precision=FP16`.

---

## Part 6 – Debugging, Profiling & Verification

### 6.1 Is It on the ANE?

1. **Xcode Instruments > Time Profiler** → look for `ANERuntimeEngine`.
2. **powermetrics** → non‑zero ANE power.
3. **Xcode Performance Report** (select `.mlpackage`) → reasons for fallback.

### 6.2 Decrypting coremltools Errors

* **"Unable to infer input dims"** → bad/missing `inputs`.
* **"op 'xxx' not implemented"** → unsupported op; see Part 3.
* **Version mismatch** → isolate env; pin deps.

### 6.3 Graph Inspection

* **Netron** → drag‑and‑drop `.mlpackage`, find bad layers.
* **MLComputePlan** → programmatic list of devices per op (see Appendix C).

---

## Part 7 – Conversion Checklist (Do *not* skip)

1. **Create fresh venv**, pin `coremltools==7.x`, `torch==2.3`, `numpy==1.26` (see Appendix D).
2. Restructure model for ANE‑friendly ops & layout **before** tracing.
3. Trace PyTorch in FP32 → convert FP16 `mlprogram` targeting iOS17+.
4. Run `mlmodel.predict()` on Mac for sanity.
5. Profile on real device; confirm ANE threads.
6. Iterate surgery until 100 % ANE.
7. Quantize to W8A8; re‑profile & regression‑test accuracy.
8. Package & ship.

---

# Appendices

## Appendix A – ANE Unsupported Ops (condensed)

| Op / Layer                 | Typical Source          | Fix                     | Notes                                         |
| -------------------------- | ----------------------- | ----------------------- | --------------------------------------------- |
| CustomLayer                | any                     | split model             | never ANE                                     |
| AddBroadcastable, ConcatND | PyTorch/TensorFlow      | replace with Add/Concat | converter inserts these when targeting iOS13+ |
| gather                     | `torch.gather`          | redesign                | forces CPU fallback                           |
| Dilated Conv               | `nn.Conv2d(dilation>1)` | stack smaller convs     |                                               |
| LSTM / GRU                 | RNNs                    | swap to Transformer     | full RNN accel absent                         |

## Appendix B – Quantization Decision Tree (text)

1. **Need absolute max accuracy?** → QAT.
2. Else **device is A17 Pro/M‑series 2024+?** → W8A8 PTQ w/ calibration.
3. Else need quick win → W8A16 data‑free PTQ.
4. Need smallest binary and can lose 3‑5 %? → GPTQ 4‑bit + optional palettization.

## Appendix C – Profiling Clues Quick Sheet

* `H11ANEServicesThread` → ANE executing.
* `Espresso::MPSEngine` → GPU fallback.
* `Espresso::BNNSEngine` → Accelerate/CPU.
* Symbolic breakpoint `-_ANEModel program` → hits only on ANE.

## Appendix D – Known‑Good Toolchain (Jul 2025)

```text
python == 3.11.6
coremltools == 7.0b5
torch == 2.3.0+cpu
torchvision == 0.18.0
numpy == 1.26.4
ane_transformers == 0.8.1
transformers == 4.42.1
```

All tested on macOS 14.5 (Apple Silicon).

## Appendix E – Resources

* **Apple docs:** Core ML Programming Guide, Core ML Tools repo, WWDC 2024 “Optimize for ANE”.
* **Community:** `coremltools` Slack, `ml-c` Discord, Whisper & Phi GitHub issues tagged `coreml`.
* **Visual tools:** Netron, CoreMLProfiler (open‑source).
* **Reference repos:** `whisper.cpp`, `ane-transformers`, `coreml-stable-diffusion`, `mps-transformer`.
