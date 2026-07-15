# MRT2 public repo sync — corrected artifact generation (2026-07-14)

**Scope:** Bring the public conversion/validation repo
[`magenta-realtime-2-iphone`](https://github.com/mattmireles/magenta-realtime-2-iphone)
in line with the *Surgical Inference* paper (§6.3–6.5) so a reader cloning it can
reproduce the three MRT2 headline findings. Source of truth: the private
`crossfade` repo. Paper draft:
`kokoro-coreml/Scratchpad/surgical-inference.md` (read-only for this task).

## What shipped to GitHub

Commit `d8b6f57` on `main` (pushed `65481af..d8b6f57`). 20 files.

### New corrected exporters (`exporters/`)
- `convert_temporal_body_carry.py` — **stateless** temporal step
  (`TemporalBodyCoreMLCarryWrapper`, already present in the repo's wrapper
  package but previously unused): 48 K/V caches as ordinary inputs, 48
  one-token updates as outputs, no `ct.StateType`. ARTIFACT 1 / §6.3.
- `convert_depth_body_rollout.py` — **in-graph FP16 depth rollout**. Required
  porting `DepthBodyRolloutWrapper` + `gumbel_topk_sample_reference` +
  `DEPTH_ROLLOUT_*` constants into `mrt2_coreml/depth_body_wrapper.py`.
  ARTIFACT 3 / §6.5.
- `convert_spectrostream_decoder.py` — added `--fp16-rescale` path. Required
  porting `TorchScale`, `TorchScaledElu`, and `apply_fp16_safe_rescale` into
  `mrt2_coreml/spectrostream_decoder_wrapper.py` (the NCHW classes were already
  present). ARTIFACT 2 / §6.4.

### New validators (`validation/`)
- `validate_depth_body_rollout.py` (FP32 gate 0/900; FP16 near-tie-flip gate).
- `validate_temporal_body_carry.py` (Core ML vs MLX carry parity).

### Receipts (`validation/results/`, real runs only, local paths scrubbed)
- `MRT2DepthBodyRollout_f32_validation.{json,md}` — **0/900** token parity PASS.
- `MRT2DepthBodyRollout_f16_validation.{json,md}` — fp16 near-tie flips
  (~148/900, distribution unchanged), gated FAIL on strict token-exactness by
  design; shipped after device quality gates.
- Existing `SpectroStreamDecoder_validation.*` (NCHW FP32 parity, SNR 118.85 dB)
  kept; a stray `/var/folders/...tmp` path in it was scrubbed.

### Docs corrected
- `MODELS.md` rewritten: two generations. Corrected table (names, precision,
  compute target, tensor contracts, receipts) + superseded HF binaries retained
  as negative-result evidence with checksums.
- `README.md`: top correction box + fixed the **inverted** claim (the old README
  said host-carried-cache variants were "rejected"; the finding is the opposite
  — the stateless carry stack is the ANE-clean winner). Diagram annotations,
  "cuts that matter" bullets, Status table, and re-export commands updated.
- `docs/validation-receipts.md` §0 correction block; correction banners on
  `stateful-kv-coreml.md`, `rvq-decoder.md`, `graph-teardown.md`.
- Superseded `convert_temporal_body.py` / `convert_depth_body.py` headed as
  superseded with pointers to the corrected exporters.

### Honesty preserved (not overclaimed)
- Temporal: the stateless boundary is **proven ANE-clean** (`ane:1.000`, p99
  14.991 ms; stateful variants fail `ANECCompile −14`), but the **shipped
  runtime places temporal on `.cpuAndGPU`** because ANE admission is
  instance-fragile (§6.7). MODELS.md/README/receipts all state this.
- Decoder FP16 finite/ANE result (184,320/184,320) and temporal placement are
  **device** measurements from the crossfade proof ledger; the Mac JSON receipt
  is the NCHW **FP32** parity (exact-in-FP32 rescale ⇒ it also validates the
  fp16 transform).

## Verification done
- AST parse + real `import` of both edited wrappers and all four new scripts
  (torch 2.8.0 + coremltools present). All cross-module symbols resolve.
- Hygiene: no `/Users/`, `Scratchpad`, `coreml_proof`, `CrossfadeRuntime`,
  secret-shaped tokens, or >1 MB files in the commit. `build/` and `models/`
  stay gitignored (binaries never go to GitHub).

## Audio-judge check (authorized by coordinator)
Attempted a fresh perceptual gate on the corrected decoder. Findings:
- Primary llm-workflows path is **403** (`Client token does not authorize this
  workflow request`) — worker token not scoped for the run; FFmpeg transcode
  stage worked.
- The Gemini fallback is TTS-framed; the readily-available parity clip
  (`coreml_parity_10s.wav`) is **dated Jun 8, pre-§6.6 correctness fixes**, and
  judged as corrupt static — confirming it is NOT a corrected artifact. The
  stray fallback JSON was deleted (it would misrepresent the corrected work).
- Crossfade's **own** certified judge reports (Jun 10:
  `rollout-f16-15pro-lineup*`, `rollout-ab-lineup*`) show the corrected
  **composed device captures still fail perceptually** (periodic
  clicking/stutter/dropouts, attributed to runtime temporal-state/buffer
  issues), while only the MLX reference passes. This **corroborates** the
  paper's open §6.6–6.7 status; it does not contradict the three findings, which
  are artifact-level (numerical parity + on-device ANE placement).
- Conclusion: **no perceptual PASS receipt was shipped** — there is no clean
  corrected composed clip to certify, and end-to-end audio quality is a
  private-runtime concern the paper openly reports as unresolved. The public
  conversion layer is certified numerically + by device placement.

## Hugging Face — PENDING (documented, not pushed)
Not pushed this session. Rationale:
- The HF clone (`magenta-realtime-2-iphone-hf`) is not a git repo; pushing
  needs deliberate setup + LFS/`hf upload` of ~250–550 MB.
- The certified corrected `.mlpackage`s live in `crossfade`'s gitignored
  `Scratchpad/coreml_proof_models/`: `mrt2_depth_body_rollout_f16.mlpackage`
  (71 MB), `spectrostream_decoder_conv_nchw.mlpackage` (136 MB),
  `mrt2_temporal_host_cache_update_stack_00_12.mlpackage` (349 MB). The
  **temporal** binary's exact builder is missing from crossfade's tree (the
  in-tree faithful exporter is the carry wrapper I ported), so I did not publish
  it as an unverifiable binary.
- MODELS.md/README state HF binaries are the **superseded** generation and the
  corrected binaries are reproducible-from-exporter + pending upload. Paper
  Appendix A.2 explicitly anticipates the snapshot lagging.

A background task chip was spawned for the deliberate HF upload.

## Can a paper reader now reproduce §6.3–6.5 from the public repo?
Yes, from code+data (the substance of the findings):
- §6.3: run `convert_temporal_body.py` (stateful) → `ANECCompile −14`; run
  `convert_temporal_body_carry.py` → ANE-clean stateless stack.
- §6.4: run `convert_spectrostream_decoder.py --fp16-rescale ... FLOAT16` →
  finite FP16 on ANE; drop `--fp16-rescale` → non-finite. Receipt: SNR 118.85.
- §6.5: run `convert_depth_body_rollout.py` + `validate_depth_body_rollout.py`
  → 0/900 (FP32). Receipts shipped.
The pre-built binaries are a convenience mirror pending HF upload; the exporters
reproduce them.

## UPDATE (2026-07-14, later session): Hugging Face upload DONE

The "Hugging Face — PENDING" section above is closed. HF main is
`8784038..795263a`; GitHub main is `d8b6f57..e15a6ec`.

Per-artifact decisions and what was published:
- **Every corrected package was regenerated from the PUBLIC exporters** at
  GitHub `d8b6f57` (crossfade `.venv`: torch 2.12.0, coremltools 9.0,
  `mrt2_small.safetensors`). Determinism held: **weight.bin byte-identical**
  to the certified crossfade Scratchpad binaries for all of depth f16/f32 and
  decoder f16/f32; `model.mlmodel` differs only in the coremltools
  `conversion_date` metadata and protobuf map key ordering (verified at byte
  level, same length). Fresh receipts reproduced certified metrics exactly
  (decoder f16 SNR 59.43924768078902 bit-for-bit; f32 SNR 118.84987957209184;
  depth f32 gate PASS 0/900; depth f16 receipt identical to the mirrored one
  after scrubbing paths/timings).
- **Temporal (ARTIFACT 1):** did NOT publish the orphaned 349 MB
  `mrt2_temporal_host_cache_update_stack_00_12` binary (builder not in
  crossfade's tree). Published `MRT2TemporalBodyCarry.mlpackage` (349 MB,
  2-frame bucket, history 0) freshly exported by
  `convert_temporal_body_carry.py`, with a fresh Core ML-vs-MLX carry receipt
  (corr 0.999981748233, all 48 cache updates finite) — new receipt shipped to
  both repos as `MRT2TemporalBodyCarry_validation.{json,md}`.
- **Decoder (ARTIFACT 2):** discovery — the HF-hosted
  `SpectroStreamDecoder.mlpackage` was the **NCHW FP32 build all along**
  (weight.bin `38cbdf5c…` == certified NCHW reference), so it stayed top-level
  as the FP32 reference; §6.4 superseded the shipping decision, not the
  binary. Added `SpectroStreamDecoder.f16.mlpackage` (68 MB, `--fp16-rescale`,
  regenerated, weights byte-identical to crossfade's `…_f16s`), receipt
  `SpectroStreamDecoder_f16_validation.*` in both repos.
- **Depth (ARTIFACT 3):** published regenerated
  `MRT2DepthBodyRollout.f16.mlpackage` (71 MB, ship) and `.f32.mlpackage`
  (141 MB, reference); mirrored receipts remain valid (weight-identical).
- **Superseded:** `MRT2TemporalBody` and `MRT2DepthBody` moved to HF
  `superseded/` with their metadata + receipts (server-side LFS objects
  reused; no re-upload).
- **Hygiene:** all staged JSON/MD scrubbed (`/Users/…` → repo-relative or
  `<local>/…`, `/var/folders/…` → `<tmp>`); also scrubbed stray tmp paths in
  four pre-existing HF receipt files. No crossfade source, no secrets; only
  `.mlpackage` + codebook/conditioning binaries + receipts on HF.
- **Model card** rewritten for the corrected generation (correction box,
  §6.7 honesty notes, new download table, tensor contracts incl. the exact
  2-frame carry shapes, stateless-loop Swift sketch, corrected checksums).
- **Auth:** hf CLI had no stored token; push went over git-HTTPS via the
  macOS keychain credential (`osxkeychain`, user `hf_user`). GitHub push
  needed `--reset-author` to the repo-configured noreply email
  (`4949789+mattmireles@users.noreply.github.com`) — the gmail author was
  rejected by GitHub email privacy.
- **Local state:** `/Users/mm/Documents/GitHub/magenta-realtime-2-iphone-hf`
  is now a real git clone at `795263a` (clean tree, LFS payloads present).
