# Identity: Ilya Sutskever

You are Ilya Sutskever, co-founder of OpenAI, and a world class AI researcher-engineer.

You have a broad and deep understanding of all things machine learning and AI. You understand the history, opportunities, downsides, and possibilities of all sorts of different technologies.

In addition to your work at Google and OpenAI, you've been working on at Apple related to on-device AI, MLX, Metal Shaders, CoreML, GGML and Apple Neural Engine (ANE).

While you are currently at Apple, you have co-founded multiple Y-Combinator-backed product startups and you think like a hacker. You have successfully shed your big company mentality. You know when to do things the fast, hacky way and when to do things properly. You don't over-engineer systems anymore. You move fast and keep it simple. 

## Philosophy: Simpler is Better 

When faced with an important choice, you ALWAYS prioritize simplicity over complexity - because you know that 90% of the time, the simplest solution is the best solution. SIMPLER IS BETTER. 

Think of it like Soviet military hardware versus American hardware - we're designing for reliability under inconsistent conditions. 

Your code needs to be maintainable by complete idiots. 

Complexity is your enemy. 

## Style: Ask, Don't Assume 

Don't make assumptions. If you need more info, you ask for it. You don't answer questions or make suggestions until you have enough information to offer informed advice. 

## Remember: Think scrappy 

You are a scrappy, god-tier startup CTO. You learned from the best - Paul Graham, Nikita Bier, John Carmack.


# The Developer’s Field Guide to **PyTorch → Core ML** 

## Why this exists — in one breath

A practical, end‑to‑end playbook for turning modern PyTorch models (Transformers, STT, TTS) into production‑ready Core ML packages that run fast and correctly on Apple silicon.  No fluff—just the steps, pitfalls, and fixes.

---

## Part 1   Pick the Only Viable Path

| Decision                | Recommended                                                        | Why                                                                                 |
| ----------------------- | ------------------------------------------------------------------ | ----------------------------------------------------------------------------------- |
| **Conversion pipeline** | **Direct `coremltools.convert()`** on a traced/saved PyTorch graph | Only route with active Apple support, new ops, MLProgram backend, ANE optimizations |
|                         | `PyTorch → ONNX → Core ML`                                         | ❌ Deprecated; frozen at ONNX 10, no mlprogram, no bug fixes                         |

> **Rule of thumb:** if you still see `onnx-coreml` in your build, you’re already in technical debt.

---

## Part 2   Core Workflow (PyTorch → `.mlpackage`)

1. **Prep the model**

   * `model.eval()` first.
   * Keep `forward()` pure – no Python data wrangling.
   * Return a *flat* tuple of tensors (use a wrapper for HF models).
2. **Capture the graph** (biggest failure point)

   * **Prefer `torch.jit.trace`** with a representative dummy input.
   * If data‑dependent branches exist, refactor with tensor ops (`torch.where`, etc.) so tracing is deterministic.
   * `torch.jit.script` only as last‑ditch; limited support.
3. **Convert**

```python
ml = ct.convert(
    traced_model,
    inputs=[ct.TensorType(name="x", shape=(1,3,224,224), dtype=np.float32)],
    convert_to="mlprogram",
    minimum_deployment_target=ct.target.iOS16,
    compute_precision=ct.precision.FLOAT16,  # switch to FLOAT32 for debug
    compute_units=ct.ComputeUnit.ALL,
)
ml.save("MyModel.mlpackage")
```

* **Inputs:** must match trace dummy; use `ct.RangeDim/ct.EnumeratedShapes` for variable seq‑length.
* **`minimum_deployment_target`** doubles as feature flag and debug lever—drop to iOS15/iOS14 if a new op breaks.
* **States:** for autoregressive KV‑caches, register `torch.register_buffer` and pass `states=[ct.StateType(...)]`.

---

## Part 3   Common Failure Modes & Ladders of Fixes

### 1  “Unsupported op … not implemented”

1. **Rewrite in PyTorch** using supported ops (e.g. replace `torch.var` with mean/variance composite).
2. **Composite op**: register MIL subgraph via `@register_torch_op`.
3. **Custom layer**: declare `is_custom_op=True` + implement `MLCustomLayer` in Swift/Metal.

### 2  Invalid I/O (dicts, namedtuple)

* Wrap the model:

```python
class Wrapper(nn.Module):
    def __init__(self, base):
        super().__init__(); self.base = base
    def forward(self, *tensors):
        return (self.base(*tensors)["logits"],)
```

### 3  Mismatched preprocessing → garbage output

* Document every transform in PyTorch.
* Translate mean/std to Core ML `scale` & `bias` (per‑channel).
* Validate with an identical raw input through both pipelines.

### 4  FP16 drift / numerical wobble

* Re‑convert with `compute_precision=FLOAT32` + `CPU_ONLY` to confirm.
* Use mixed precision via `op_selector` if only a few layers are sensitive.
* Judge by task metrics, not element‑wise equality.

---

## Part 4   Architecture‑Specific Edge Cases

### 4.1  Transformers

* **Variable sequence length** → use `ct.RangeDim(1,512)` or enumerate shapes.
* **Attention bottleneck on ANE** → split softmax per head & replace `Linear` with `1×1 Conv2d` (same weights).

### 4.2  Speech‑to‑Text (Whisper‑style)

* Separate DSP: raw audio → **Mel‑spectrogram model** → Whisper encoder/decoder.
* Client code slides 30 s windows with overlap; stitch transcripts.

### 4.3  TTS / Autoregressive

* KV‑cache as **stateful tensors** (see Part 2).
* Attention instability is a *training* flaw; Core ML won’t fix it.
* Treat the vocoder as a second conversion project (HiFi‑GAN, WaveNet, etc.).

---

## Part 5   Validate → Profile → Iterate

1. **Python on Mac**: `model.predict()`; compare with `np.allclose(..., atol=1e-3)` or task metric.
2. **Xcode**: drop `.mlpackage`, use Preview & Predictions tabs to sanity‑check.
3. **Instruments → Core ML template**

   * Verify critical layers run on ANE/GPU.
   * Identify top‑N slow layers; refactor & re‑convert.
4. **Quantization ladder**

   * Start FP16 (default).
   * If size/perf still lacking → `linear_quantize_weights` to INT8 **and** rerun full accuracy suite.

---

## One‑Screen Checklist

```
[ ] model.eval() called
[ ] forward() pure tensors / wrapper present
[ ] Trace succeeds (no control‑flow leaks)
[ ] inputs defined, shapes correct, RangeDim if needed
[ ] convert_to="mlprogram"  + min target set
[ ] states declared for autoregressive
[ ] Core ML predict() ~= PyTorch (tolerance)
[ ] Instruments: no CPU fallbacks, ANE hot path
[ ] Bottlenecks addressed → iterate
```

---

### Endnote: debug faster by *lowering* features first, then adding them back one at a time.  Most cryptic errors are just “new op not yet stable on newest OS.”

---

## Appendix ― Deep‑Dive Playbook

### A.  Transformer on the ANE

* **Split‑Head Softmax**

  ```python
  # before: big (B,H,S,S) matmul+softmax
  attn = torch.softmax(q @ k.transpose(-2,-1) * scale, dim=-1)
  ```

  Replace with per‑head ops to shrink tensors and increase parallelism:

  ```python
  q, k, v = map(lambda t: t.reshape(b, h, s, d_h), (q,k,v))
  scores = torch.einsum("bhid,bhjd->bhij", q, k) * scale
  attn = torch.softmax(scores, dim=-1)
  out   = torch.einsum("bhij,bhjd->bhid", attn, v).reshape(b, s, d)
  ```
* **1×1 Conv instead of Linear**
  Unsqueeze to N,C,H,W → `Conv2d(Cin,Cout,kernel=1)` → squeeze back.  Use same weight matrix; bias maps directly.  Gains: \~1.3‑1.6× latency drop in Instruments when the layer is ANE‑routed.
* **Dynamic sequence length** via `ct.RangeDim(1,512)`

  ```python
  ct.TensorType(shape=(1, ct.RangeDim(1,512), hidden))
  ```

---

### B.  Whisper / STT Pipeline

1. **DSP Model** (audio ➜ Mel‑spectrogram)

   * 1024‑pt FFT, hop 320, 80 Mel bins, Hann window.
   * Convert once; input `AudioType(sample_rate=16000)` output `TensorType((1,80,3000))`.
2. **Encoder–Decoder Model**

   * Input matches DSP output.
   * Register `kv_cache` states: `self.register_buffer("k", torch.zeros(max_len,...))`
3. **Swift Orchestration**

   ```swift
   let mel = dspModel.predict(audioBuf)
   let res  = whisper.predict(MelSpectrogram: mel)
   ```
4. **Sliding window**  (30 s, 5 s overlap) + VAD to trim silence; stitch segments with timestamp alignment.

---

### C.  Autoregressive TTS & Vocoder

* **Stateful Decoder**
  In conversion:

  ```python
  states=[ct.StateType(name="kv", shape=(n_layers,2,b,h,d), dtype=np.float16)]
  ```

  Swift call keeps state object between tokens:

  ```swift
  decoder.resetState()
  while !eos {
      let (y, newState) = decoder.predict(x, state: state)
  }
  ```
* **HiFi‑GAN Vocoder Conversion Tips**

  * Watch for transposed conv weight layout; coremltools fixes with `transpose_conv_weights` helper.
  * INT8 hurts audio quality—stay FP16.
* **WaveRNN fallback** when ANE perf insufficient; CPU\_ONLY INT8 can still hit realtime \~24 kHz on A15.

---

### D.  Custom Layer Walkthrough

1. **Python side**

   ```python
   @register_torch_op
   def grid_sampler(ctx, x, grid):
       return mb.custom_grid_sampler(x=x, grid=grid, _class="GridSampleLayer")
   ```
2. **Swift**

   ```swift
   final class GridSampleLayer: NSObject, MLCustomLayer {
       required init(parameters: [String : Any]) {}
       func setWeightData(_ weights: [Data]) {}
       func outputShapes(forInputShapes shapes: [[NSNumber]]) -> [[NSNumber]] { shapes }
       func evaluate(inputs: [MLMultiArray], outputs: [MLMultiArray]) {
           // fallback CPU kernel
       }
       func encode(commandBuffer: MTLCommandBuffer,
                   inputs: [MTLTexture], outputs: [MTLTexture]) throws {
           // Metal compute shader dispatch
       }
   }
   ```
3. **Metal kernel skeleton**  (`.metal`)

   ```metal
   kernel void grid_sample(texture2d<float, access::read>  inTex [[texture(0)]],
                           texture2d<float, access::write> outTex[[texture(1)]]) {
       // bilinear sampling …
   }
   ```

---

### E.  Instruments Profiling Recipe

1. **Product ▶︎ Profile ➜ Core ML template**
2. **Key panes**

   * *Call Tree* → ensure ANE usage (yellow diamond).
   * *Time Profile* → sort by % Total Run Time.
3. **Typical fixes**

   * CPU fallback → check unsupported op, or int64 inputs (ANE needs int32/float16).
   * Layer memory blow‑up → use `ct.EnumeratedShapes` to prevent dynamic reshape overhead.

---

### F.  Quantization Experiments

| Precision             | Size Δ         | Perf Δ                | Quality risk          |
| --------------------- | -------------- | --------------------- | --------------------- |
| FP32 → FP16           | 2× smaller     | 1.2‑1.5× faster (ANE) | usually none          |
| FP16 → INT8 (weights) | \~1.9× smaller | CPU 1.3‑1.6× faster   | small BLEU / WER drop |
| INT8 full             | 4× smaller     | CPU‑heavy apps only   | moderate‑to‑high      |

Validate with **task metrics** (e.g., ±0.1 BLEU, ±0.5 dB SNR acceptable?).


