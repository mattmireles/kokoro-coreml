# LFM2.5 Surgical Prefill Stage 0 Report

**Date:** 2026-07-20
**Status:** Phase 0 inventory frozen; Phase 1 admission evidence pending
**Checkpoint:** `LiquidAI/LFM2.5-350M` at
`b9d6e4e2d75f440b12a2b4d731c808004ecbbd89`
**Prior art:** `john-rocky/CoreML-LLM` at
`5ef6b301d3a3d628e25c0605479f59dbf3a7d955`

## Summary

Use the real 350M checkpoint. Its 16-layer layout is explicit and easy to
isolate, so the 230M fallback is unnecessary. Reuse CoreML-LLM's weight-name
mapping, ANE-oriented Conv2d projections, RMSNorm re-expression, and its
proven short-conv state semantics. Do not reuse its monolithic exporter: it
traces a one-token decode graph with fixed context tensors and cannot answer
the full-sequence, enumerated-shape prefill question in plan 010.

The device matrix exists, but the mobile devices are not currently runnable
from Xcode. The iPad is physically connected over USB and visible to
`devicectl`, but it is unpaired. The iPhone is paired and Developer Mode is
enabled, but it is offline and running an OS 27.0 beta. Phase 1 may proceed on
the Mac while iPad pairing is resolved; no mobile dispatch claim is valid
until the device itself produces a compute plan.

## Upstream Converter Diff

CoreML-LLM's LFM2 path is a useful decode reference, not a surgical-prefill
implementation.

| Concern | CoreML-LLM behavior | Plan 010 decision |
| --- | --- | --- |
| Graph scope | One monolithic, one-token decode graph. | New standalone full-sequence conv and GQA prefill wrappers. |
| Weight loading | Direct safetensors loader; linear layers become 1x1 Conv2d; trained FFN width adjusts from 6656 to 4608. | Reuse the mapping and width adjustment. |
| Tensor layout | Hidden activations are permuted to `[B, C, 1, T]` for Conv2d projections. | Reuse inside wrappers; expose the smallest flat tensor contract needed by each block. |
| K/V cache | One rank-4 `MLState` with mask-based writes for decode. | No cache input for prefill blocks; emit per-layer K/V tensors for decode seeding. |
| Conv state | Explicit fp16 `conv_state_in`/`conv_state_out`; a second `MLState` caused ANE runtime status `0x1d`. | Keep explicit state I/O. For full prefill, emit the final live short-conv window. |
| Short convolution | Double gate `B*x`, depthwise causal conv, then `C*conv`; live cache length 3. | Reuse these semantics exactly. Do not inherit the upstream slow-decode bug. |
| Shapes | Token dimension is fixed at 1; context-length masks are fixed. | Enumerated prompt buckets `{128, 256, 512, 1024, 2048}` are mandatory. |
| Precision | Shipping LFM2 path is fp16. INT4 short-conv fails ANE admission upstream. | fp16 only; quantization remains out of scope. |
| Deployment | Upstream converter currently targets iOS 26. | Freeze the minimum target used by our export and record it beside every dispatch table. |
| Validation | Top-1 comparison against Hugging Face full-prefill logits; upstream documents a buggy HF slow-decode state update. | Block-output max-abs and cosine on 32 real prompts, using full-prefill semantics as ground truth. |

### Reuse versus rewrite

**Decision: surgical rewrite over copied primitives.** Copying the monolithic
wrapper would preserve the wrong graph boundary and hide the independent
variable. The smallest proper path is to implement two focused wrappers while
porting only these proven primitives:

- checkpoint/config parsing and safetensors weight mapping;
- the 6656-to-4608 FFN adjustment;
- Q/K RMSNorm before RoPE;
- Conv2d-based projections in ANE layout;
- explicit short-conv state I/O and correct `cat([state[..., 1:], Bx])`
  semantics.

This also isolates a critical upstream result: its fp16 LFM2 decode package is
reported as 97.8% ANE-resident overall, but all 10 depthwise short convolutions
are CPU-resident. The experiment must therefore measure the whole conv block,
not infer ANE admission from the model-wide percentage.

## Frozen 350M Architecture

Source: the real checkpoint's `config.json` at the checkpoint revision above.

| Property | Value |
| --- | --- |
| Hidden size | 1024 |
| Layers | 16 |
| Short-conv layers | 10 |
| Full-attention layers | 6 |
| Query heads | 16 |
| Key/value heads | 8 |
| Head dimension | 64 |
| Short-conv live cache length | 3 |
| Raw FFN dimension | 6656 |
| Effective trained FFN dimension | 4608 |
| Vocabulary | 65,536 |
| RoPE theta | 1,000,000 |
| Checkpoint dtype | bfloat16 |

| Layer | Type | Contiguous run |
| ---: | --- | --- |
| 0 | conv | conv 0-1 |
| 1 | conv | conv 0-1 |
| 2 | full attention | GQA 2 |
| 3 | conv | conv 3-4 |
| 4 | conv | conv 3-4 |
| 5 | full attention | GQA 5 |
| 6 | conv | conv 6-7 |
| 7 | conv | conv 6-7 |
| 8 | full attention | GQA 8 |
| 9 | conv | conv 9 |
| 10 | full attention | GQA 10 |
| 11 | conv | conv 11 |
| 12 | full attention | GQA 12 |
| 13 | conv | conv 13 |
| 14 | full attention | GQA 14 |
| 15 | conv | conv 15 |

The exact order is therefore:

```text
C C A C C A C C A C A C A C A C
```

The full Stage 1 decomposition will have 13 contiguous operator-class runs.
That is substantially more model boundaries than the conceptual two-class
diagram, so G1a is a serious gate rather than paperwork.

## Frozen Device and Toolchain Inventory

Inventory was captured on 2026-07-20 before export code was written.

| Device | Exact hardware | OS build | Xcode/toolchain | Availability and role |
| --- | --- | --- | --- | --- |
| Mac Studio | Mac14,14; M2 Ultra; 24 CPU cores; 64 GB | macOS 26.5.2 (25F84) | Xcode 26.6 (17F113); Python 3.11.15; PyTorch 2.6.0; coremltools 8.3.0 | Local and runnable; export, parity, provisional admission. |
| M1 Mini | Macmini9,1; M1; 8 CPU cores; 16 GB | macOS 15.7.7 (24G720) | Xcode 26.2 (17C52); coremltools 8.3.0 | Reachable over SSH at `irvine-m1.local`; Phase 4 rail attribution. |
| iPad Pro 11-inch M2 | iPad14,3 | iPadOS 26.5 (23F77) | Host Xcode 26.6 | Physically detected over USB; currently unpaired/offline to Xcode. Phase 1 mobile proxy and Phase 4 primary sustained-load device. |
| iPhone 15 Pro Max | iPhone16,2; 512 GB | iOS 27.0 beta (24A5380h) | Developer Mode enabled; host Xcode 26.6 | Paired but currently offline. Phase 5 only, after Phases 0-4. |

The local Python environment emits a compatibility warning: coremltools 8.3
was tested only through PyTorch 2.5, while this checkout has PyTorch 2.6. The
upstream CoreML-LLM main branch now pins a much newer stack
(`coremltools>=9.0`, PyTorch 2.11, Transformers 5.5). Phase 1 must freeze a
dedicated experiment environment rather than silently mutate the repo's Kokoro
environment.

## License Posture

- Model weights and derived model artifacts remain under the
  [LFM Open License v1.0](https://huggingface.co/LiquidAI/LFM2.5-350M/blob/main/LICENSE).
  Redistribution must include the license, retain applicable notices, and mark
  modified files. Commercial use is licensed only below the stated US $10M
  annual-revenue threshold unless Liquid AI grants separate terms.
- New experiment code is MIT-licensed. The public repo must make the split
  explicit: MIT covers our scripts and harness, not Liquid AI's weights.
- CoreML-LLM is MIT-licensed. Reused code, if any, must retain its copyright
  notice; the preferred approach is a clean, attributed implementation of the
  small primitives listed above.

## Phase 0 Decision

**GO to Phase 1 on the Mac Studio.** The 350M checkpoint is suitable, the
conversion reference is understood, and the hardware exists. iPad dispatch is
blocked on pairing, so Phase 1 cannot pass its provisional G0a gate until that
device is connected and produces its own dispatch table.

The external research brief is recorded in
[`lfm2-prefill-coreml-research-brief.md`](lfm2-prefill-coreml-research-brief.md).
The Stage 0 admission tables, numerics, and final G0 verdict will be appended
after the single-block experiment.

## Verification

```bash
curl -fsSL https://huggingface.co/LiquidAI/LFM2.5-350M/resolve/main/config.json
git -C /tmp/<clone>/CoreML-LLM rev-parse HEAD
xcrun devicectl list devices
ssh -n -o BatchMode=yes mattmireles@irvine-m1.local /usr/bin/sw_vers
```

Not testable by one command: mobile physical availability. Strongest evidence
is the same-session `devicectl` hardware/OS record above; dispatch readiness
still requires pairing and a device-produced compute plan.

