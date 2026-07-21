# LFM2.5 Surgical Prefill Stage 0 Report

**Date:** 2026-07-20
**Status:** Phase 0 and Phase 1 complete; Stage 0 admission gates pass
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

The Stage 0 wrappers now convert and pass the pre-registered numerical gate on
32 real prompts at every frozen bucket. The Mac Studio compute plans place all
28 conv-block operations on the Neural Engine and all 84 GQA-block operations
on the GPU. The iPad Pro M2 independently places all 27 costed conv-block
operations on ANE and all 84 GQA-block operations on GPU for the canonical
enumerated packages and every fixed-shape diagnostic twin. Stage 0 therefore
passes G0a, G0b, and G0c and may proceed to Phase 2.

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
| Short convolution | Double gate `B*x`, depthwise causal conv of kernel width `k=3`, then `C*conv`. | Reuse these semantics exactly. The prefill-to-decode handoff is the final `k-1=2` samples of `B*x`, not a three-token state and not ungated `x`. |
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
- explicit short-conv state I/O, a width-3 convolution kernel, and the correct
  two-sample `B*x` handoff required to seed decode.

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
| Short-conv kernel width `k` | 3 |
| Prefill-to-decode state width `k-1` | 2 |
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
| iPad Pro 11-inch M2 | iPad14,3 | iPadOS 26.5 (23F77) | Host Xcode 26.6 | Paired over wired USB; Developer Mode, DDI services, and tunnel verified. Explicit profile `7626bdf7-0b2c-498e-8067-ce3315083473` signed the model runner. Phase 1 mobile admission complete; Phase 4 primary sustained-load device. |
| iPhone 15 Pro Max | iPhone16,2; 512 GB | iOS 27.0 beta (24A5380h) | Developer Mode enabled; host Xcode 26.6 | Paired but currently offline. Phase 5 only, after Phases 0-4. |

The export and numerical environment is frozen in
`scripts/lfm2_surgical/requirements-stage0.txt`: coremltools 8.3.0, PyTorch
2.6.0, Transformers 4.48.3, NumPy 1.26.4, tokenizers 0.21.0, safetensors
0.5.2, and huggingface-hub 0.28.1. It loads `config.json`, safetensors, and
`tokenizer.json` directly rather than instantiating a Transformers LFM2 model.
Coremltools warns that 8.3 was tested only through PyTorch 2.5, so the warning
is part of the recorded environment rather than evidence of failure.

The independent official-oracle check has its own frozen
`scripts/lfm2_surgical/requirements-hf-reference.txt` with Transformers 5.5.0
and PyTorch 2.6.0. Keeping that path isolated avoids silently mutating the
Core ML export environment just to obtain the official LFM2.5 implementation.

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

## Stage 0 Single-Block Experiment

### Method and artifacts

Every artifact below was generated from `LiquidAI/LFM2.5-350M` revision
`b9d6e4e2d75f440b12a2b4d731c808004ecbbd89`. The downloaded
`model.safetensors` SHA-256 is
`1c9c77a4471a7f590f85240f74ed1fc26df7fbde88c3006724e2f93ca993ea4e`.
The graph-defining `config.json` SHA-256 is
`720b43d6ddc2ed25be23eed355aefcf342434a176dedad23dbe0a5e3ac24bbb8`,
and `tokenizer.json` is
`df1d8d5ec5d091b460562ffd545e4a5e91d17d4a0db7ebe733be34ed374377bd`.
The canonical extraction, export, official-reference, and numerical manifests
all carry the revision and all three digests; a changed config or tokenizer
now fails before measurement just like changed weights.

The experiment uses layer 0 as the representative double-gated LIV conv block
and layer 2 as the representative GQA block. GQA inputs are not synthetic
embeddings: the fp32 reference executes layers 0 and 1 first, then supplies the
actual layer-2 activation. Both wrappers preserve the checkpoint's trained
weights and emit flat tensors:

- conv: `hidden_states -> (hidden_out, conv_state_out)`, where
  `conv_state_out` is the final two `B*x` samples required by a width-3 causal
  convolution;
- GQA: `(hidden_states, cosine, sine, attention_mask) ->
  (hidden_out, key_out, value_out)`.

The conv wrapper implements the checkpoint's RMSNorm equation directly. It
does **not** use the LayerNorm re-expression. Inspection of the final MLProgram
shows that `ct.precision.FLOAT16` lowered every conv RMSNorm operation to fp16,
so the supported result is about the direct RMSNorm algebra, not retained fp32
accumulation. The LayerNorm diagnostic is retained only as failed evidence
because it crossed the numerical gate.

The canonical packages are `conv_block.mlpackage` and `gqa_block.mlpackage`
under ignored `outputs/lfm2_surgical/stage0/`. Each declares the five exact
enumerated shapes `{128, 256, 512, 1024, 2048}` and targets iOS 18. Fixed-shape
twins named `*_128.mlpackage` through `*_2048.mlpackage` exist only to obtain
an unambiguous per-bucket compute-plan proxy. They share the traced module,
weights, precision policy, and output contract, but they are diagnostic
artifacts: they do not prove which enumerated specialization the runtime
selects inside the canonical package and are not candidates for shipment.

The conv package is 36,725,164 bytes and the GQA package is 34,640,815 bytes.
The export manifest confirms five enumerated variants on every canonical
input. This passes G0b: the converter neither rejected enumerated shapes nor
silently replaced them with range dimensions.

### Independent official-reference check

The hand-written fp32 oracle was checked in an isolated Transformers 5.5.0
environment against the official `Lfm2Model` implementation. On a 128-token
probe, the embedding and outputs after conv layers 0 and 1 and GQA layer 2 all
had max-abs error exactly `0.0` against the official model. The registered gate
was `1e-5`; `hf_reference.json` records a pass with no failures.

This probe tokenizes the first registered natural-language prompt, then repeats
and truncates its token IDs to exactly 128 tokens. It proves that the
independent first-three-layer oracle matches the official implementation for
that input. It is not a claim about broad language-distribution coverage.

### Numerical validation

`check_numerics.py` compares Core ML against fp32 PyTorch on 32 real tokenizer
prompts. The validation compute policy is `.cpuAndGPU`: on this host,
`.cpuOnly` compilation of an fp16 MLProgram terminates inside Apple's runtime
with `SIGTRAP`, while `.cpuAndGPU` executes correctly and removes ANE execution
from the parity question. Every semantically live output is gated at max-abs
`<= 1e-2`; cosine similarity is reported independently.

Each of the 32 registered natural-language strings is tokenized independently,
then its token IDs are repeated and truncated to fill each exact bucket. The
suite therefore covers 32 distinct prompt prefixes and real trained embedding
activations, but it is not 32 organically authored 2,048-token documents. This
repeat/truncate construction is deterministic and identical on the fp32 and
Core ML paths.

| Bucket | Conv hidden max-abs | Conv state max-abs | GQA hidden max-abs | GQA key max-abs | GQA value max-abs | Lowest cosine |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 0.000259 | 0.007870 | 0.000359 | 0.006760 | 0.000081 | 0.99999962 |
| 256 | 0.000259 | 0.004650 | 0.000359 | 0.006760 | 0.000081 | 0.99999968 |
| 512 | 0.000259 | 0.006802 | 0.000359 | 0.006760 | 0.000081 | 0.99999972 |
| 1024 | 0.000259 | 0.007870 | 0.000359 | 0.006760 | 0.000081 | 0.99999975 |
| 2048 | 0.000259 | 0.007870 | 0.000359 | 0.006986 | 0.000081 | 0.99999976 |

**G0c passes.** The worst gated error is `0.007870` on the conv-state output;
the worst GQA key error is `0.006986` at bucket 2048. Both are below the frozen
`1e-2` threshold, and `numerics.json` records zero failures.

This pass required two narrow numerical fixes, both discovered against the
real downstream activations:

1. Replacing RMSNorm with LayerNorm algebra caused conv-state max-abs error
   `0.01322`. Restoring direct RMSNorm algebra reduced it below the gate without
   changing the conv dispatch plan. The compiled RMSNorm operations are fp16;
   LayerNorm is not used by the canonical package.
2. Embedding-only GQA inputs hid key-cache drift. Feeding the actual layer-2
   activation exposed a `0.01059` failure. The final precision selector keeps
   only three named module scopes in fp32: `operator_norm`,
   `attention_positioning` (Q/K RMSNorm plus RoPE), and `ffn_norm`. Each island
   has an explicit fp16 output boundary; attention scores, cache outputs,
   residual arithmetic, and the MLP remain fp16. This narrowed policy reduced
   the worst key error below the gate without changing the model equation.

The failed LayerNorm rewrite remains recorded in
`numerics_layernorm_rms.json`; negative diagnostics were not overwritten by
the passing run. That debugging-only file predates the stronger config and
tokenizer digest contract and is retained as historical evidence, not as a
canonical numerical manifest.

### Mac Studio dispatch

The compute plans below were captured from the final regenerated packages on
the M2 Ultra host. Fractions count the preferred device for every costed MIL
operation. They prove the toolchain and graph admit the intended placement on
this host; the independent mobile result follows in the next section.

| Bucket | Conv policy | Conv preferred device | GQA policy | GQA preferred device |
| ---: | --- | --- | --- | --- |
| Enumerated package | CPU+ANE | 28/28 ANE (100%) | CPU+GPU | 84/84 GPU (100%) |
| 128 diagnostic twin | CPU+ANE | 28/28 ANE (100%) | CPU+GPU | 84/84 GPU (100%) |
| 256 diagnostic twin | CPU+ANE | 28/28 ANE (100%) | CPU+GPU | 84/84 GPU (100%) |
| 512 diagnostic twin | CPU+ANE | 28/28 ANE (100%) | CPU+GPU | 84/84 GPU (100%) |
| 1024 diagnostic twin | CPU+ANE | 28/28 ANE (100%) | CPU+GPU | 84/84 GPU (100%) |
| 2048 diagnostic twin | CPU+ANE | 28/28 ANE (100%) | CPU+GPU | 84/84 GPU (100%) |

The per-op histograms are identical across the enumerated package and all five
fixed-shape diagnostic twins:

| Conv MIL op type | Count; preferred device |
| --- | ---: |
| `ios16.reduce_mean` | 2 ANE |
| `ios18.add` | 4 ANE |
| `ios18.conv` | 6 ANE |
| `ios18.mul` | 7 ANE |
| `ios18.pow` | 2 ANE |
| `ios18.rsqrt` | 2 ANE |
| `ios18.silu` | 1 ANE |
| `ios18.slice_by_index` | 2 ANE |
| `pad` | 1 ANE |
| `split` | 1 ANE |

| GQA MIL op type | Count; preferred device |
| --- | ---: |
| `ios16.reduce_mean` | 4 GPU |
| `ios18.add` | 9 GPU |
| `ios18.cast` | 8 GPU |
| `ios18.concat` | 2 GPU |
| `ios18.conv` | 7 GPU |
| `ios18.expand_dims` | 2 GPU |
| `ios18.matmul` | 2 GPU |
| `ios18.mul` | 16 GPU |
| `ios18.pow` | 4 GPU |
| `ios18.reshape` | 8 GPU |
| `ios18.rsqrt` | 4 GPU |
| `ios18.silu` | 1 GPU |
| `ios18.slice_by_index` | 4 GPU |
| `ios18.softmax` | 1 GPU |
| `ios18.transpose` | 10 GPU |
| `tile` | 2 GPU |

The durable JSON files also record each package-tree SHA-256, compute-unit
policy, coremltools version, total costed operations, device histogram, and
CPU-preferred operation list. Here every conv operation prefers ANE and every
GQA operation prefers GPU; neither family has a CPU-preferred operation.

`numerics.json` independently records the canonical conv and GQA package-tree
SHA-256 values, cryptographically binding the G0c results to the same exact
graphs used by the canonical dispatch plans.

### iPad Pro dispatch

The same packages were compiled and loaded through coremltools' remote model
runner on the physical iPad Pro M2. Every device JSON records iPad14,3,
iPadOS 26.5 build 23F77, wired transport, paired state, enabled Developer Mode,
available DDI services, connected tunnel, coremltools 8.3.0, and `where=device`.

| Bucket | Conv policy | Conv preferred device | GQA policy | GQA preferred device |
| ---: | --- | --- | --- | --- |
| Enumerated package | CPU+ANE | 27/27 ANE (100%) | CPU+GPU | 84/84 GPU (100%) |
| 128 diagnostic twin | CPU+ANE | 27/27 ANE (100%) | CPU+GPU | 84/84 GPU (100%) |
| 256 diagnostic twin | CPU+ANE | 27/27 ANE (100%) | CPU+GPU | 84/84 GPU (100%) |
| 512 diagnostic twin | CPU+ANE | 27/27 ANE (100%) | CPU+GPU | 84/84 GPU (100%) |
| 1024 diagnostic twin | CPU+ANE | 27/27 ANE (100%) | CPU+GPU | 84/84 GPU (100%) |
| 2048 diagnostic twin | CPU+ANE | 27/27 ANE (100%) | CPU+GPU | 84/84 GPU (100%) |

The iPad and Mac JSONs carry identical package-tree SHA-256 values for every
corresponding plan, proving that both devices inspected the same artifacts:

| Package | Conv SHA-256 | GQA SHA-256 |
| --- | --- | --- |
| Enumerated | `194e7608fb527baa7884338a1b6bceb0a4e13bbc9be0ce966c9adc25577c12f1` | `0f754224794bf5d1d1fc90b38712bb8cf95e066d6c8196616f9a3ca87aa63a25` |
| 128 twin | `060e508ac44d085e8f07b6752c48fa42d2b5d68687e2d2b0b5481ce045162875` | `83bb8a03099b2a7a145046147da95fc12fdaa695c8a2b60a7d1c5be2f97f6348` |
| 256 twin | `db44914d8c2dbeecdded798ac921a88d57b8b7b93d1b8c2302e7b19205829915` | `364754a5435372a23dbb82a2fa38b74bfa9cde926dc635a7b479c1bb80ff8c46` |
| 512 twin | `e6f3f05dd330eae2adfc5862492309fa344716ab484b6e072f2b598202ea6700` | `9053c38e51fb3aa27cc29b5f9a0b78776557515cb93be6749c4a044746a56b44` |
| 1024 twin | `c1f1635fd87cdd5cff7eaea2e37f73f1c8b72e650a4142505650d37a3abd62e0` | `5fd94e1c02e6e7e40a601edbeefdba728acdc32f2e0e214803b3cc2777dc079b` |
| 2048 twin | `6b15d8ac525ca72987d1a8d93ca2abce33cb5fb88fcaded3077cda22d828c82a` | `b68a22965c7c4f1359794cd41920e78c5c0439a3ba1ff67a2664dcd56735e22c` |

The Mac counts 28 costed conv operations while the iPad counts 27. Comparing
the durable per-op histograms shows exactly one difference: the Mac plan
contains one ANE-preferred `split` operation, while the iPad plan does not
return `split` as a costed operation. All shared operator-type counts match.
This evidence establishes the reported inventory difference; it does not by
itself establish whether the missing operation was fused, eliminated, or
merely omitted from device cost accounting.

The runner used explicit signing rather than automatic portal mutation:

- development team: `6ETYBAJKY8`;
- bundle identifier: `com.mattmireles.CoreMLModelRunner`;
- provisioning profile UUID: `7626bdf7-0b2c-498e-8067-ce3315083473`;
- profile name: `LFM2 Surgical ModelRunner iDesk 2026-07-20`;
- device UDID: `00008112-000E40821AE8A01E`.

`provisioning_registration.json` records that the profile was CMS-signed,
contains the measured device, carries two development certificates, and uses a
wildcard application identifier. Every iPad dispatch JSON independently
records `signing_mode=explicit_profile` and the same profile UUID.

### Phase 1 gate verdict

- **G0a:** pass. Mac Studio is 28/28 ANE and iPad Pro M2 is 27/27 ANE for the
  conv block at buckets 128, 256, and 512; both are 100%, above the 80% gate.
  The 1024 and 2048 diagnostic twins also report 100% ANE preference.
- **G0b:** pass. Exact enumerated shapes are present for all five buckets.
- **G0c:** pass. All five outputs are below `1e-2` on all 32 prompts and all
  five buckets.

**GO.** All Stage 0 kill gates pass. Proceed to Phase 2 while retaining the
fixed-shape packages as diagnostic proxies rather than shipping artifacts.

## Phase 0 Decision

**GO to Phase 1 on the Mac Studio.** The 350M checkpoint is suitable, the
conversion reference is understood, and the hardware exists. Phase 1 has since
validated this decision on both the Mac Studio and iPad Pro M2 and produced the
complete Stage 0 gate evidence above.

The external research brief is recorded in
[`lfm2-prefill-coreml-research-brief.md`](lfm2-prefill-coreml-research-brief.md).
The completed external report was ingested and corrected against current
primary sources in
[`LFM2-surgical-prefill-CoreML-guide.md`](../Guides/apple-silicon/LFM2-surgical-prefill-CoreML-guide.md).
This report now contains the checkpoint oracle, export, Mac admission, and
numerical evidence plus the physical iPad admission table and final GO verdict.

## Verification

```bash
# Independent source oracle: Transformers 5.5.0 from the isolated target.
PYTHONPATH="$PWD/outputs/lfm2_surgical/transformers5" \
  .venv/bin/python scripts/lfm2_surgical/check_hf_reference.py

# Real checkpoint extraction, canonical exports, and 32-prompt parity.
.venv/bin/python scripts/lfm2_surgical/extract_blocks.py
.venv/bin/python scripts/lfm2_surgical/export_blocks.py \
  --all-buckets --block both
.venv/bin/python scripts/lfm2_surgical/check_numerics.py --prompts 32

# Canonical host compute plans.
.venv/bin/python scripts/dump_device_compute_plan.py \
  --package outputs/lfm2_surgical/stage0/conv_block.mlpackage \
  --compute-units CPU_AND_NE \
  --out outputs/lfm2_surgical/stage0/dispatch_mac_conv_enumerated.json
.venv/bin/python scripts/dump_device_compute_plan.py \
  --package outputs/lfm2_surgical/stage0/gqa_block.mlpackage \
  --compute-units CPU_AND_GPU \
  --out outputs/lfm2_surgical/stage0/dispatch_mac_gqa_enumerated.json

# Per-bucket fixed-shape diagnostic proxies.
for bucket in 128 256 512 1024 2048; do
  .venv/bin/python scripts/lfm2_surgical/export_blocks.py \
    --all-buckets --block both --fixed-bucket "$bucket"
  .venv/bin/python scripts/dump_device_compute_plan.py \
    --package "outputs/lfm2_surgical/stage0/conv_block_${bucket}.mlpackage" \
    --compute-units CPU_AND_NE \
    --out "outputs/lfm2_surgical/stage0/dispatch_mac_conv_${bucket}.json"
  .venv/bin/python scripts/dump_device_compute_plan.py \
    --package "outputs/lfm2_surgical/stage0/gqa_block_${bucket}.mlpackage" \
    --compute-units CPU_AND_GPU \
    --out "outputs/lfm2_surgical/stage0/dispatch_mac_gqa_${bucket}.json"
done

# Live device prerequisites.
xcrun devicectl device info details --device iDesk

# Canonical iPad compute plans with the registered explicit profile.
.venv/bin/python scripts/dump_device_compute_plan.py \
  --package outputs/lfm2_surgical/stage0/conv_block.mlpackage \
  --compute-units CPU_AND_NE \
  --device-type ipad --device-name iDesk \
  --development-team 6ETYBAJKY8 \
  --bundle-identifier com.mattmireles.CoreMLModelRunner \
  --provisioning-profile-uuid 7626bdf7-0b2c-498e-8067-ce3315083473 \
  --out outputs/lfm2_surgical/stage0/dispatch_ipad_conv_enumerated.json
.venv/bin/python scripts/dump_device_compute_plan.py \
  --package outputs/lfm2_surgical/stage0/gqa_block.mlpackage \
  --compute-units CPU_AND_GPU \
  --device-type ipad --device-name iDesk \
  --development-team 6ETYBAJKY8 \
  --bundle-identifier com.mattmireles.CoreMLModelRunner \
  --provisioning-profile-uuid 7626bdf7-0b2c-498e-8067-ce3315083473 \
  --out outputs/lfm2_surgical/stage0/dispatch_ipad_gqa_enumerated.json

# Fixed-shape iPad diagnostic proxies.
for bucket in 128 256 512 1024 2048; do
  .venv/bin/python scripts/dump_device_compute_plan.py \
    --package "outputs/lfm2_surgical/stage0/conv_block_${bucket}.mlpackage" \
    --compute-units CPU_AND_NE \
    --device-type ipad --device-name iDesk \
    --development-team 6ETYBAJKY8 \
    --bundle-identifier com.mattmireles.CoreMLModelRunner \
    --provisioning-profile-uuid 7626bdf7-0b2c-498e-8067-ce3315083473 \
    --out "outputs/lfm2_surgical/stage0/dispatch_ipad_conv_${bucket}.json"
  .venv/bin/python scripts/dump_device_compute_plan.py \
    --package "outputs/lfm2_surgical/stage0/gqa_block_${bucket}.mlpackage" \
    --compute-units CPU_AND_GPU \
    --device-type ipad --device-name iDesk \
    --development-team 6ETYBAJKY8 \
    --bundle-identifier com.mattmireles.CoreMLModelRunner \
    --provisioning-profile-uuid 7626bdf7-0b2c-498e-8067-ce3315083473 \
    --out "outputs/lfm2_surgical/stage0/dispatch_ipad_gqa_${bucket}.json"
done

# Focused executable checks.
.venv/bin/python -m pytest -q tests/test_lfm2_surgical_tools.py
```

Not testable by one command: mobile physical availability. Strongest evidence
is the same-session `devicectl` record plus the 12 device-produced compute-plan
JSON files above, each bound to the explicit profile and exact package hash.
