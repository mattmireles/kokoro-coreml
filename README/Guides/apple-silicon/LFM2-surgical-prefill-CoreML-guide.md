# LFM2.5 Surgical Prefill Export to Core ML

This guide ingests an external Max research report on full-sequence LFM2.5
prefill and surgical Core ML segment export. Treat the report as research
input, not canonical truth; the equations below were corrected against the
current model implementation and the repo's measured prior art.

Raw report:

- `/Users/mm/Documents/GitHub/llm-workflows/outputs/create-guide/lfm2-5-full-sequence-prefill-and-surgical-core-ml-segment-export/2026-07-20T23-58-47-025Z/raw-report.md`

## The smallest viable experiment

Do not start with a monolithic LFM2.5 prefill graph. Isolate one full
short-convolution block and one full-attention block from a real checkpoint,
then answer three questions:

1. Do finite prompt buckets survive conversion as enumerated shapes?
2. Does the short-convolution block actually map to the ANE on the target
   device?
3. Do every block output and state-handoff tensor remain within the frozen
   fp16-versus-fp32 error gate?

Only build the full interleaved pipeline if all three answers are yes.

## Short-convolution semantics

For input `X` with shape `[B, S, H]`, the LFM2.5 short-convolution operator is:

```text
B, C, x = split(in_proj(norm(X)), 3)
Bx      = B * x
z       = depthwise_causal_conv(Bx, kernel_size=3)
y       = out_proj(C * z)
```

The full block adds the operator residual, applies the FFN RMSNorm and SwiGLU
MLP, then adds the FFN residual.

For prefill, left-pad `Bx` by `kernel_size - 1` zeros before a no-padding
depthwise convolution. The state needed to seed the next decode token is the
last `kernel_size - 1` positions of **`Bx`**, not the ungated `x` branch. This
distinction is easy to miss and produces a numerically plausible but wrong
decode handoff.

Use static slicing for the state:

```python
state_out = bx[..., -2:]
```

In ANE layout, use `[B, H, 1, S]`; express dense projections as 1x1 Conv2d and
the temporal operator as depthwise Conv2d with kernel `(1, 3)`.

## Full-attention prefill semantics

For a 350M checkpoint with 16 query heads, 8 key/value heads, and head
dimension 64:

1. Apply the block RMSNorm to the residual input.
2. Compute Q, K, and V projections.
3. Apply the learned per-head Q/K RMSNorm before RoPE.
4. Apply precomputed cosine and sine tables at every prompt position.
5. Expand each K/V head across its two query heads.
6. Add a four-dimensional causal mask and run full-sequence attention.
7. Emit K and V in `[B, 8, S, 64]` form for decode-cache seeding.
8. Apply the output projection, residual, FFN norm, SwiGLU MLP, and final
   residual.

Use `-65504.0` as the finite fp16 causal-mask value. Do not hide cache reads,
writes, or data-dependent branches inside the prefill graph.

## Enumerated-shape contract

Use a finite bucket set such as `{128, 256, 512, 1024, 2048}`. Apple documents
that `EnumeratedShapes` can carry up to 128 variants and permits device-side
specialization. Starting with iOS 18, multiple inputs may use enumerated
shapes, but every flexible input must have the same number of variants and
only variants at the same index may be combined.

Therefore a GQA graph with `hidden_states`, `cosine`, `sine`, and
`attention_mask` inputs must enumerate all four in the same bucket order.
Using a range for one input or mismatching the order changes the experiment.

Canonical conversion shape contract:

```python
inputs = [
    ct.TensorType(
        name="hidden_states",
        shape=ct.EnumeratedShapes([(1, 1024, 1, s) for s in buckets]),
        dtype=np.float16,
    ),
    ct.TensorType(
        name="cosine",
        shape=ct.EnumeratedShapes([(1, s, 64) for s in buckets]),
        dtype=np.float16,
    ),
    ct.TensorType(
        name="sine",
        shape=ct.EnumeratedShapes([(1, s, 64) for s in buckets]),
        dtype=np.float16,
    ),
    ct.TensorType(
        name="attention_mask",
        shape=ct.EnumeratedShapes([(1, 1, s, s) for s in buckets]),
        dtype=np.float16,
    ),
]
```

Use iOS 18 or newer as the minimum deployment target for this multi-input
contract. Apple currently describes `torch.jit.trace` as the stable,
recommended Core ML capture path; `torch.export` support remains beta.

## Segment boundary and state handoff

Keep the short-convolution and GQA admission probes separate. Existing LFM2
decode work found that combining a KV `MLState` with convolution state could
produce ANE runtime status `0x1d`; explicit convolution state I/O avoided that
dual-state failure. This is prior-art evidence for isolation, not proof that
every combined prefill graph fails.

A complete interleaved pipeline still has to preserve the checkpoint's real
layer order. Do not batch all convolution layers before all attention layers.
Group only contiguous runs, hand the hidden tensor between segments, and emit
each layer's state at the point where that layer runs.

## Validation ladder

1. Compare an independent fp32 PyTorch block with the fp16 ANE-layout PyTorch
   candidate before conversion.
2. Convert with ML Program, fp16 precision, explicit flat tensor I/O, and the
   complete enumerated bucket set.
3. Compare every Core ML output with the independent fp32 reference on real
   prompts. Include convolution state and K/V outputs in the gate.
4. Build fixed-shape diagnostic twins for each bucket. A compute plan for an
   enumerated package may describe only its default specialization. Follow the
   [bucket-specific compute-plan guide](CoreML-enumerated-shape-compute-plan-specialization-guide.md)
   when the exact non-default bucket decides a gate.
5. Capture per-op `MLComputePlan` placement on every claim-bearing device.
   `.cpuAndNeuralEngine` is permission to use the ANE, not proof of residency.
6. Keep conversion, first-load compilation, warmed prediction, and sustained
   thermal measurements as separate rows.

## Environment discipline

Loading the official Transformers implementation and loading checkpoint
tensors directly are different dependencies. A direct config/safetensors/
tokenizer loader does not require the newest Transformers LFM2 model class.
Freeze the exact tested package versions and state which path the scripts use;
do not turn an incidental tokenizer compatibility problem into a model-format
requirement.

## Primary references

- [Apple: Flexible Input Shapes](https://apple.github.io/coremltools/docs-guides/source/flexible-inputs.html)
- [Apple: PyTorch Conversion Workflow](https://apple.github.io/coremltools/docs-guides/source/convert-pytorch-workflow.html)
- [Hugging Face Transformers: LFM2 implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/lfm2/modeling_lfm2.py)

## Related documentation

- [Core ML compute unit scheduling](CoreML-Compute-Unit-Scheduling-guide.md)
- [Enumerated-shape compute-plan specialization](CoreML-enumerated-shape-compute-plan-specialization-guide.md)
- [Core ML ANE compiler failure triage](CoreML-ANE-compiler-failure-triage-guide.md)
- [Splitting Core ML graphs](Splitting-CoreML-Graphs-guide.md)
- [Core ML ANE tensor layout](CoreML-ANE-tensor-layout-guide.md)
- [LFM2 Stage 0 report](../../Notes/lfm2-stage0-report.md)
