# LFM2.5 Surgical Prefill Core ML Research Brief

**Date:** 2026-07-20
**Status:** External research requested
**Target guide:**
`README/Guides/apple-silicon/LFM2-surgical-prefill-CoreML-guide.md`

## Context

We are executing
[`010-lfm2-surgical-prefill-plan.md`](../Plans/010-lfm2-surgical-prefill-plan.md)
against the real `LiquidAI/LFM2.5-350M` checkpoint. The frozen checkpoint
configuration has 16 layers with this order:

```text
conv, conv, full_attention, conv, conv, full_attention, conv, conv,
full_attention, conv, full_attention, conv, full_attention, conv,
full_attention, conv
```

The model uses hidden size 1024, 16 query heads, 8 key/value heads, head
dimension 64, and a double-gated short convolution with live cache length 3.
The experiment exports fixed/enumerated full-sequence prefill buckets at
128, 256, 512, 1024, and 2048 tokens. It compares homogeneous Core ML
compute-unit placement against conv segments permitted on ANE and GQA
segments permitted on GPU.

The current upstream CoreML-LLM implementation at commit
`5ef6b301d3a3d628e25c0605479f59dbf3a7d955` provides a monolithic one-token
decode wrapper. It uses a rank-4 Core ML state for K/V, explicit
`conv_state_in`/`conv_state_out` tensors for short-conv state, padded Conv2d
weights for ANE-friendly layout, and a direct `coremltools` MLProgram export.
It does not provide the full-sequence LFM2.5 prefill graph required here.

## Primary Research Goal

Produce an advanced developer field guide for implementing and debugging a
numerically faithful, fixed-shape LFM2.5 full-sequence prefill graph in
PyTorch and Core ML, including explicit segment boundaries between contiguous
short-conv and GQA runs. Focus on external architecture and framework
mechanics, not product decisions or a roadmap.

## Questions to Answer

- What are the exact full-sequence tensor equations and causal padding/state
  semantics for LFM2.5's double-gated LIV short-conv block?
- Which tensor represents the rolling decode state after a full prefill, and
  how should the last `k - 1` or `k` positions be extracted without changing
  semantics?
- What are the exact GQA, Q/K RMSNorm, RoPE, causal-mask, and K/V output
  semantics for full-sequence prefill?
- Which current official Liquid AI or Hugging Face implementation is the best
  ground truth, and what version-specific differences matter between
  Transformers 4.x and 5.x?
- Which PyTorch graph patterns for depthwise causal convolution, gating,
  RMSNorm, grouped-query attention, and state extraction are most likely to
  convert cleanly through current `coremltools` to MLProgram?
- What Core ML enumerated-shape limitations apply when sequence length changes
  across fixed buckets, especially when masks or K/V outputs also contain the
  sequence axis?
- For ANE scheduling, which layouts and operator re-expressions are supported
  evidence versus speculation? Clearly separate compiler admission from
  runtime placement.
- How should a block-level numerical parity harness isolate errors in conv,
  attention, residual, and MLP subpaths?
- What known upstream bugs or mismatches exist in LFM2/LFM2.5 slow decode,
  cache handling, tokenizer metadata, dtype/config parsing, or Core ML export?

## Source Priorities

Prefer primary sources:

- Liquid AI LFM2/LFM2.5 model code, technical report, model cards, and config.
- Hugging Face Transformers' current `modeling_lfm2.py` and `configuration_lfm2.py`.
- Apple Core ML and coremltools documentation, Apple ML Research, and WWDC.
- CoreML-LLM source and its LFM2 conversion findings.
- Reproducible GitHub issues only when official documentation is silent.

Avoid generic SEO tutorials and unsupported claims about ANE placement.

## Output Format

- Executive summary first.
- Exact equations and tensor-shape tables.
- Do-this/avoid-this tables.
- Minimal PyTorch wrappers for one short-conv block and one GQA prefill block.
- Core ML conversion patterns using enumerated shapes.
- Failure modes and a numbered debugging ladder.
- Primary-source references with stable links and version/commit identifiers.
- Mark speculation separately from evidence-backed behavior.
- Text only: no charts, images, diagrams, or generated visualizations.
