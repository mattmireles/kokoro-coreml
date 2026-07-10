# ANE Graph Optimization Plan

**Date:** 2026-04-14  
**Status:** Implemented (MIL + waveform gates; harness timing deferred)  
**Plan audit:** 2026-04-14 — revised after multi-agent audits: bakeoff schema (`t_coreml_predict_s`), export mutation semantics, MIL depth, hook API, Pearson + secondary metric, `3s`/`10s` shipping set. Second audit pass: Conv1d/Conv2d MIL gate, hook round-trip test, op-count script requirement, bias docs, rollback procedure, reproducible validation inputs, fallback scope note.

> **Prerequisite (baseline numbers):** Prefer completing [Bakeoff Plan](kokoro-bakeoff-v2.md) so `scripts/bakeoff_harness.py` exists and Config A results (manifest + stage timings) are reproducible. Compare Phase 3 against the **bakeoff v2 schema** — e.g. `t_coreml_predict_s` for the Core ML predict stage in Config A (see `kokoro-bakeoff-v2.md` results JSON example), not the informal `t_ane_predict` name from older notes. **If the harness is not in-tree yet,** use [§ Benchmark fallback (pre-harness)](#benchmark-fallback-pre-harness) — do not block Phase 1–2 on the harness.

## Executive Summary

Cross-referencing the [CoreML Compute Unit Scheduling Guide](../Guides/apple-silicon/CoreML-Compute-Unit-Scheduling-guide.md) and the [vocoder scheduling field guide](../Guides/apple-silicon/Core%20ML-MLX-Scheduling-1D-ConvTranspose-ISTFTNet-vocoders-guide.md) against our production `GeneratorFromHar` ANE path supports one high-impact optimization: many `nn.Linear` layers inside the traced Core ML graph (from `AdaIN1d.fc` inside `AdaINResBlock1`) may be replaced with 1×1 convolution primitives so the stack aligns with ANE-favorable conv ops (Orion Constraint 17: 1×1 Conv executes ~3× faster than equivalent matmul on ANE matrix ALUs). Whether Core ML already lowers some `linear` MIL ops internally is **unknown** until Phase 0 and Phase 3.

**Conv1d vs Conv2d decision:** The scheduling guide prescribes `nn.Conv2d` with 4D layout `(B, C, 1, S)` for ANE alignment. This plan uses `nn.Conv1d(kernel_size=1)` because `AdaIN1d` operates on 3D tensors `(B, C, T)` — adding a spurious spatial dimension via Conv2d would require reshape ops that may themselves trigger ANE fallback. **Phase 0 must verify** that Conv1d(k=1) lowers to an equivalent MIL `conv` op as Conv2d(1,1) — if it does not, Phase 1 switches to Conv2d with the necessary unsqueezes. See Phase 0 task list.

**Export pipeline nuance (read before implementing):** `export_synth/convert.py` replaces `AdainResBlk1d.norm1/norm2` with `IdentityAdaIN` **once** on the shared `SynthesizerModel(kmodel)` **before** the per-bucket loop — for **every** `--mode` (`decoder`, `decoder-har`, `full`). That mutates the in-memory decoder tree even when the traced artifact is generator-only. Only the **JIT subgraph** matters for each package: **`decoder-har` traces `GeneratorFromHar(generator)`**, so **`AdainResBlk1d` never appears in that `.mlpackage`**; the Generator’s `AdaINResBlock1` / `AdaIN1d` layers are **not** swapped to identity. The **`decoder-har` subgraph does not include `AdainResBlk1d`** (Decoder encode/decode stacks); it starts at `x_pre` + `har` after PyTorch has already run the decoder front. This plan targets the real `AdaIN1d` modules on the HAR-post hot path. (Reload a fresh `kmodel` after export if you need unmodified `AdainResBlk1d` in the same Python process.)

**Fan-in:** `AdaIN1d` is shared by `AdaINResBlock1` (Generator) and `AdainResBlk1d` (Decoder). Any change must keep **PyTorch** `Decoder` / training checkpoints valid, not only the HAR-post Core ML package.

## Problem Statement

- **Symptom:** ANE decoder predict time on the order of ~0.25–0.31s (anecdotal / model-card class numbers; re-measure per baseline doc) may leave throughput on the table.
- **Root cause hypothesis:** Each `AdaINResBlock1` holds six `AdaIN1d` instances; each uses `nn.Linear(style_dim, num_features * 2)`. On ANE, dense linear stacks may be less favorable than explicit 1×1 convolutions (see guide + Apple transformer-on-ANE references).
- **Impact:** `num_upsamples * num_kernels` resblocks plus `num_upsamples` `noise_res` blocks each contribute multiple `AdaIN1d` forwards per inference — a large count of linear projections on the traced graph.

## Goals and Non-Goals

### Goals

- [x] Replace `nn.Linear` with `nn.Conv1d(kernel_size=1)` in `AdaIN1d` (`kokoro/istftnet.py`) for graph and ANE alignment.
- [x] Verify numerical parity: unit tests (`tests/test_adain1d_linear_vs_conv1d.py`) + package-level check in Phase 2.
- [x] Pretrained checkpoints load without retraining (`register_load_state_dict_pre_hook` reshapes `fc.weight` from 2D → 3D when needed).
- [x] Re-export **in-repo shipping** decoder HAR-post buckets (`3s`,`10s`); waveform gates vs `/tmp` baseline packages; MIL recount (Phase 3). Harness timing deferred (file not in tree).
- [x] Add a **checked-in** pytest for Linear-vs-Conv1d equivalence and hook round-trip; full `uv run pytest` green.

### Non-Goals

- Changing full-`Decoder` Core ML export as the primary deliverable (deprecated path for this iteration).
- Replacing `nn.Linear` in `SourceModuleHnNSF` (hn-nsf stays on CPU in the hybrid design).
- Modifying the DurationModel.
- Reworking **`AdainResBlk1d` / `IdentityAdaIN`** for **`decoder-har`** (those blocks are **not** in the traced `GeneratorFromHar` graph). **`IdentityAdaIN` is still applied in-process to `kmodel` for all modes** before trace — see nuance above; this plan does not change that policy.
- Full Generator architecture rework.
- Auditing the full Orion constraint catalog (20 constraints) beyond Constraints 1 (concat) and 17 (matmul vs conv). Other constraints (e.g., IOSurface 24 KB minimum, multi-input uniform allocation, GELU rejection) are not expected to interact with this change but are out of scope for this iteration.

### Already done in this repo (do not re-track as work)

- **Dead `torch.cat` in `AdaIN1d`:** Removed; `AdaIN1d.forward` uses `assert C == self.num_features` and documents why (no concat on the live path). Do not list “remove torch.cat” in Definition of Done.

## Scope and Constraints

- **Scope:** `AdaIN1d` in `kokoro/istftnet.py`, checkpoint hook, tests, re-export `coreml/kokoro_decoder_har_post_{3s,10s}.mlpackage`, benchmark log.
- **Shipping buckets:** Aligned with [kokoro-bakeoff-v2.md](kokoro-bakeoff-v2.md): checked-in HAR-post artifacts are **`3s` and `10s` only**. The exporter accepts any `--buckets` list — **3s/10s is policy**, not an enforced code gate. Optional additional buckets (e.g. `5s` for [Hugging Face distribution](https://huggingface.co/mattmireles/kokoro-coreml)) are **out of scope for bakeoff Config A** unless repo policy changes — if exported, document them separately from “shipping Config A.”
- **Guardrails:** Perceptual quality preserved. **Package-level:** Pearson **r > 0.99** on **comparable** runs (same bucket, same frozen inputs, same `waveform` length / crop rules). Use **real** `x_pre` / `har` / `ref_s` from PyTorch. **Secondary gates (hard):** After casting both waveforms to **float32** for the metric only — **SNR ≥ 40 dB** (signal = reference waveform) **and** **max absolute sample delta ≤ 1e-2**. **Near-silent reference** (RMS < **1e-4** on float32 waveform): skip Pearson; require max abs Δ only + optional listen. **Gate order:** export/smoke finiteness → Pearson (unless skipped) → SNR/delta → merge rule. **Unit:** `torch.allclose` on `AdaIN1d` outputs with tight atol/rtol as in new test.

## Ground Truth Contracts (Do Not Violate)

- **Weight compatibility:** `nn.Linear(in, out)` weight shape `(out, in)`. `nn.Conv1d(in, out, kernel_size=1)` weight shape `(out, in, 1)`. Transform: `conv.weight = linear.weight.unsqueeze(-1)`. Bias shape `(out,)` is identical for both `nn.Linear` and `nn.Conv1d` — no transformation needed; the load hook does **not** touch `fc.bias`.
- **Functional equivalence:** For `s` of shape `(B, style_dim)`, `nn.Linear(style_dim, out)(s)` yields `(B, out)`, then the current code `view`s to `(B, out, 1)` for chunking. With `nn.Conv1d(style_dim, out, 1)`, use `self.fc(s.unsqueeze(-1))` to obtain `(B, out, 1)` in one step; keep `torch.chunk(..., dim=1)` unchanged for `gamma` / `beta`.
- **No retraining:** Inference-time module swap + load hook only.

## Already Shipped (Do Not Re-Solve)

- **`GeneratorFromHar`:** `export_synth/wrappers.py` — class `GeneratorFromHar` (vocoder tail for `decoder-har`).
- **`AdaIN1d` normalization:** Manual mean/var (export-friendly).
- **CustomSTFT / vocoder STFT path:** As currently used in export.
- **Decoder HAR post (git):** `coreml/kokoro_decoder_har_post_3s.mlpackage`, `coreml/kokoro_decoder_har_post_10s.mlpackage`.

## Fresh Baseline (Current State)

- **`AdaIN1d.fc`:** `nn.Conv1d(..., kernel_size=1)` in `kokoro/istftnet.py` (`AdaIN1d.__init__`) + Linear checkpoint pre-hook.
- **Generator trace:** Six `AdaIN1d` calls per `AdaINResBlock1` × (resblocks + noise_res); MIL uses `conv` primitives for style projection (post–Phase 1 export).
- **ANE timing:** Bakeoff `t_coreml_predict_s` when harness exists; else `scripts/bench_decoder_har_post_predict.py` (fallback median ms on this machine).

## Solution Overview

| Phase | Purpose |
| ----- | ------- |
| 0 | MIL / op audit on existing `kokoro_decoder_har_post_3s.mlpackage` (and optionally `10s`) |
| 1 | `AdaIN1d`: Linear → `Conv1d(1)`, `register_load_state_dict_pre_hook`, pytest |
| 2 | Re-export HAR-post buckets (`3s,10s`); export gate + Pearson on real tensors |
| 3 | Benchmark vs baseline; re-audit MIL |

## Implementation Phases

### Phase 0: MIL and op audit

**Goal:** Enumerate MIL operation types in the **existing** traced HAR-post package — not just open the protobuf shell.

**Tasks:**

- [x] Load `coreml/kokoro_decoder_har_post_3s.mlpackage` with `coremltools` (pin version = repo environment; record `coremltools.__version__` in the log).
- [x] **Count ops (required script):** `scripts/count_mil_ops.py` walks `spec.mlProgram` / `block_specializations` / `operations`. *Do not* claim completion from `get_spec()` alone without descending to operations.
- [x] Baseline 3s histogram (2026-04-14, `uv run`, coremltools 8.3.0): **2207** ops; **`linear` 48**, **`conv` 51**, no `concat`; top types include `const`, `add`, `mul`, `tile`, `reduce_mean`.
- [x] **Conv1d vs Conv2d MIL equivalence (hard gate):** `uv run python scripts/count_mil_ops.py --probe-conv-lowering` — minimal Conv1d(k=1) and Conv2d(1×1) both yield MIL op set `{cast, const, conv}`; **proceed with Conv1d** for Phase 1.
- [x] **Memory / layout:** Generator AdaIN uses `num_features` 256 then 128; transient `(B, 2*C, 1)` matches pre/post change. Full-graph MIL does not expose AdaIN intermediates as package outputs (outputs are waveform); padding risk is compile-time internal.
- [ ] **Strongly recommended:** macOS 14+, `MLComputePlan` (or Instruments Core ML) per-op placement before claiming ANE impact in Phase 3 — **not run in this session** (`placement_evidence: unavailable` allowed per Phase 3 DoD).

**Verification:** `scripts/count_mil_ops.py` checked in; taxonomy locked to **MIL `op.type` strings** (e.g. `linear`, `conv`, `const`) as printed by the script. Phase 3 compares the same labels on the new 3s package.

---

### Phase 1: Replace Linear with Conv1d in AdaIN1d

**Goal:** Same semantics, Conv1d for export; checkpoints load.

**Tasks:**

- [x] In `AdaIN1d.__init__`, set `self.fc = nn.Conv1d(style_dim, num_features * 2, kernel_size=1)` (bias enabled to match Linear).
- [x] In `AdaIN1d.forward`, use `h = self.fc(s.unsqueeze(-1))` so `h` is `(B, 2 * num_features, 1)`; `gamma, beta = torch.chunk(h, 2, dim=1)`.
- [x] Register a **correct** load hook. PyTorch invokes `register_load_state_dict_pre_hook` with signature `(module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)`. Example:

```python
def _adain_fc_linear_weights_to_conv1d(
    module,
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):
    key = prefix + "fc.weight"
    tensor = state_dict.get(key)
    if tensor is not None and tensor.dim() == 2:
        state_dict[key] = tensor.unsqueeze(-1)

# In AdaIN1d.__init__, after super().__init__():
self.register_load_state_dict_pre_hook(_adain_fc_linear_weights_to_conv1d)
```

Verify the callback arity against PyTorch docs for the **pinned** export stack (`torch==2.5.0` in `requirements-export.txt`).

- [x] Do **not** reintroduce `torch.cat` padding; keep the existing channel assertion.
- [x] `tests/test_adain1d_linear_vs_conv1d.py`: Linear reference vs Conv1d `AdaIN1d`, hook3D round-trip, synthetic 2D checkpoint load.
- [x] `tests/test_adain1d_decoder_smoke.py`: tiny `AdainResBlk1d` forward finite.
- [x] `uv run pytest` —24 passed (2026-04-15).
- [x] `tests/test_export_wrappers_shapes.py::test_synthesizer_model_forward_runs_and_returns_1d_audio` passes.

**Verification:** New tests pass; hook round-trip covers both 2D→3D upgrade and 3D→3D no-op; full `tests/` green in venv.

---

### Phase 2: Re-export decoder HAR post buckets

**Goal:** Rebuild **3s** and **10s** packages (in-repo shipping set).

**Tasks:**

- [x] Export: `uv run python -m export_synth.main --mode decoder-har --buckets 3s,10s -o coreml` (2026-04-15).
- [x] **Export gates:** Traced vs Core ML finite on export geometry for both buckets.
- [x] **Explicit validation:** `scripts/compare_decoder_har_post_waveforms.py` — baseline packages copied to `/tmp/...` pre-export when comparing pre/post Conv1d; full `HybridTTSPipeline()` so `_select_bucket_seconds` matches production; same text/voice/speed/`torch.manual_seed(0)`. Pearson **> 0.99**, SNR **> 40 dB**, max abs Δ **< 1e-2** for 3s and 10s.
- [x] `uv run pytest tests/test_mlpackage_exports.py -q` — pass.

**Verification:** Both packages save; smoke path works; Pearson **> 0.99**, **SNR ≥ 40 dB**, **max abs Δ ≤ 1e-2** (float32 metric tensors) on real tensors.

---

### Phase 3: Benchmark and compare

**Goal:** Quantify wall or ANE-segment improvement vs baseline.

**Tasks (harness path — preferred once landed):**

- [ ] Run [kokoro-bakeoff-v2.md](kokoro-bakeoff-v2.md) harness Config A when `scripts/bakeoff_harness.py` exists; compare **`t_coreml_predict_s`** to baseline manifest.
- [x] Re-run Phase 0 MIL tally on **new** `3s` package: **2353** ops, **`linear` 0**, **`conv` 99** (baseline was 2207 ops, linear 48, conv 51).
- [x] `outputs/bakeoff/ane_optimization_results.json` written (gitignored) with MIL + waveform metrics + `placement_evidence: unavailable`.
- [x] **Results log** table below updated in-repo.

#### Benchmark fallback (pre-harness)

Use until `scripts/bakeoff_harness.py` exists:

- [x] Same machine / default Core ML compute units (`MLComputeUnits.all` via coremltools load) — document OS/build in JSON when publishing numbers.
- [x] **`scripts/bench_decoder_har_post_predict.py`** — warmup + median of N `predict()` calls; tensors from `build_decoder_har_post_inputs_np` (identical path to production). Optional `--baseline /path/to/old.mlpackage` for A/B median ms and `%` speedup.
- [ ] Optional: `powermetrics` / Instruments / `MLComputePlan` (still recommended for ANE placement proof).
- [x] Results merged into `outputs/bakeoff/ane_optimization_results.json` with `"benchmark_mode": "fallback_loop"` (gitignored); harness will use `"bakeoff_harness"`.
- [x] **Scope note:** Fallback measures **predict-only** wall time (maps to `t_coreml_predict_s`); full hybrid RTF still dominated by CPU stages — do not over-claim end-to-end speedup from this script alone.

**Verification:** Documented improvement meeting the **merge rule** in Open Questions, **or** documented revert. **Phase 3 DoD:** Include **placement evidence** (`MLComputePlan` / Instruments summary) **or** explicit `placement_evidence: unavailable` in the results JSON with reliance on wall-clock only.

---

## Success Criteria

### Hard requirements

- [x] Phase 0 Conv1d/Conv2d MIL equivalence gate passed (Conv1d confirmed to lower to `conv` MIL op).
- [x] `scripts/count_mil_ops.py` checked in and produces reproducible op histograms.
- [x] `AdaIN1d.fc` is `nn.Conv1d(kernel_size=1)` in committed code.
- [x] Hook matches PyTorch’s pre-hook API; checkpoints with 2D `fc.weight` load.
- [x] Hook round-trip test passes: both 2D→3D upgrade and 3D→3D no-op paths verified.
- [x] New pytest for Linear/Conv equivalence; full suite passes in project venv (`uv run pytest`).
- [x] `tests/test_export_wrappers_shapes.py::test_synthesizer_model_forward_runs_and_returns_1d_audio` still passes (full Generator path through Conv1d `AdaIN1d`).
- [x] `kokoro_decoder_har_post_3s` and `_10s` re-exported; smoke + `compare_decoder_har_post_waveforms.py` gates pass (Pearson, SNR, max Δ).
- [x] Decoder smoke test (`tests/test_adain1d_decoder_smoke.py`) passes.
- [x] Benchmark result logged: MIL + waveform + **fallback** `t_coreml_predict_median_ms` via `scripts/bench_decoder_har_post_predict.py` (`ane_optimization_results.json`); bakeoff harness still future work.

### Definition of Done

- [x] Phases 0–3 complete with logs (**Phase 3:** MIL + waveform + fallback predict benchmark + results JSON; bakeoff harness + optional powermetrics / `MLComputePlan` still open).
- [x] No stale `torch.cat` removal claims.
- [x] Plan references: harness command remains documented as **when file exists**; fallback + MIL path used here.
- [x] Re-export performed from clean `mlpackage` sources (no stale `.mlmodelc` in repo).
- [x] [§ Results log](#results-log-commit-this) table filled for baseline + after Conv1d.

## Open Questions

### Resolved

- **Does `GeneratorFromHar` trace include concat from old AdaIN padding?** No — padding branch removed; assert-only path in `AdaIN1d`.
- **Does `IdentityAdaIN` affect this plan?** It replaces `AdainResBlk1d` norms on the shared `kmodel` for **every** export `--mode` before trace, but those layers **do not appear** in the **`decoder-har`** JIT graph (only `GeneratorFromHar` does). This plan does not change that policy.
- **Checkpoint compatibility?** Yes, via pre-hook `fc.weight` unsqueeze from 2D checkpoints; tensors already 3D are left unchanged.
- **Conv1d(k=1) vs Conv2d(1×1) MIL lowering?** Phase 0 `--probe-conv-lowering`: both minimal traces yield the same MIL type set `{cast, const, conv}` — **Conv1d retained** for `AdaIN1d`.

### Unresolved

- **Does MIL already map legacy `linear` to ANE-optimal paths?** Unknown without `MLComputePlan` / Instruments — `placement_evidence: unavailable`. **Wall-clock:** fallback median `predict` ms is in `ane_optimization_results.json`; bakeoff **`t_coreml_predict_s`** and optional `--baseline` A/B still the publication path.
- **Merge rule (closed for this PR):** Hard requirements + waveform gates passed; **(b)** satisfied — `linear` MIL ops **48 → 0** on `3s` (taxonomy: `op.type` from `scripts/count_mil_ops.py`). **(a)** Controlled `predict()` median is logged via `scripts/bench_decoder_har_post_predict.py` (re-run with `--baseline` on a saved pre–Conv1d `.mlpackage` for strict before/after %). Full bakeoff **`t_coreml_predict_s`** still deferred until `scripts/bakeoff_harness.py` exists. **Revert procedure** unchanged if a future benchmark shows regressions without MIL benefit.

## References

### Internal

- [CoreML Compute Unit Scheduling Guide](../Guides/apple-silicon/CoreML-Compute-Unit-Scheduling-guide.md)
- [Bakeoff Plan](kokoro-bakeoff-v2.md)
- [Debug Notes](../Notes/debug-notes.md)
- [Learnings](../learnings.md)

### External

- [Apple: Deploying Transformers on the ANE](https://machinelearning.apple.com/research/neural-engine-transformers)
- [Orion: Characterizing Apple's Neural Engine](https://arxiv.org/abs/2603.06728)
- [Pre-built packages (distribution)](https://huggingface.co/mattmireles/kokoro-coreml) — optional consumer artifacts; **git** remains source of truth for exporter and pinned behavior.

## Risks and Mitigations

| Risk | Mitigation |
| ---- | ---------- |
| Core ML already optimizes linear → no speedup | Phase 0 MIL + Phase 3 numbers; document outcome. |
| Conv1d MIL lowering differs from Conv2d | Phase 0 hard gate: compare Conv1d(k=1) vs Conv2d(1,1) MIL ops before Phase 1. Switch to Conv2d if they differ. |
| FP16 numeric drift | Pearson on real tensors; export `--precision fp32` diagnostic if needed. |
| Singleton-axis / padding interactions (guide) | Phase 0 notes on `(B,C,1)` prevalence; both old and new paths produce same transient shape; verify MIL does not materialize a buffer. Kokoro `num_features` values: 256 (first level), 128 (deeper). |
| Hook API misuse | Copy verbatim signature from this plan; review against PyTorch docs for pinned version. |
| Synthetic export gate non-finite | Use real `x_pre`/`har` for acceptance; see `export_synth/convert.py` decoder-har branch. |
| Conv1d AdaIN1d used in future full-decoder export | `IdentityAdaIN` replaces `AdainResBlk1d.norm1/norm2` before trace for all modes, so Conv1d `AdaIN1d` is dead code in full/decoder paths. If `IdentityAdaIN` swap is ever removed, Conv1d `AdaIN1d` may re-surface MIL broadcast issues — document this in `IdentityAdaIN` docstring. |

## Files Likely to Change

| File | Change |
| ---- | ------ |
| `kokoro/istftnet.py` | `AdaIN1d`: Conv1d + hook + update class docstring (`Linear` → `Conv1d`) |
| `scripts/count_mil_ops.py` | **New** MIL op-type histogram (Phase 0 required deliverable) |
| `scripts/compare_decoder_har_post_waveforms.py` | Baseline vs candidate waveform gates |
| `scripts/bench_decoder_har_post_predict.py` | Phase 3 fallback: median `predict()` ms |
| `tests/test_adain1d_linear_vs_conv1d.py` | **New** equivalence test + hook round-trip (2D→3D and 3D→3D) |
| `tests/test_adain1d_decoder_smoke.py` | **New** `AdainResBlk1d` fan-in smoke |
| `coreml/kokoro_decoder_har_post_3s.mlpackage` | Rebuild |
| `coreml/kokoro_decoder_har_post_10s.mlpackage` | Rebuild |
| `outputs/bakeoff/ane_optimization_results.json` | Benchmark log (gitignored) |

## Results log (commit this)

When the optimization PR merges, **append a row here** (or replace the placeholder table) so the repo keeps a durable summary. Full JSON may stay gitignored under `outputs/bakeoff/`.

| Run | git SHA | coremltools | Hardware | `t_coreml_predict_s` or fallback median (ms) | Pearson (3s / 10s) | Max abs Δ / SNR | MIL note (linear→conv?) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline (pre–Phase 2 export) | parent of Conv1d commit | 8.3.0 | dev machine | not measured | — | — | 3s: `linear` **48**, `conv` 51 |
| After Conv1d | see `ane_optimization_results.json` | 8.3.0 | dev machine | ~19 ms median `predict` (3s, fallback script; not harness) | ~0.999996 / ~0.999995 | Δ~2e-3, SNR~50 dB | 3s: `linear` **0**, `conv` **99** |
