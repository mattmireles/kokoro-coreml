# Hugging Face Re-upload of iSTFT-Fixed har_post Packages (2026-07-14)

Date: 2026-07-14 (upload commits landed 2026-07-15 UTC)
Author: Ilya (persona) via kokoro-coreml agent loop

## Why

The audio-quality fixes captured in
[`cs1-audio-quality-evaluation-2026-07-14.md`](cs1-audio-quality-evaluation-2026-07-14.md)
("Root causes and fixes") re-exported the five
`kokoro_decoder_har_post_{3,7,10,15,30}s.mlpackage` generators with the corrected
one-sided iSTFT scaling (interior bins `2/n_fft` doubling + OLA window-power
normalization) in `kokoro/custom_stft.py`. The fixed packages were only present
locally in the gitignored `coreml/` tree. The Hugging Face model repo
`mattmireles/kokoro-coreml` was last modified **2026-06-30** and still served the
**pre-fix** spectral-tilt binaries. Anyone hydrating models via
`scripts/download_models.py` (SDK downloaded-resource mode, `HostedManifest.json`)
therefore still got the half-amplitude-mid-band bug. This note records making the
distributed artifacts match the fixed source.

## Affected package set (verified)

Only the generator/iSTFT packages are affected. Confirmed by a full local-vs-HF
SHA-256 diff over every full-profile `coreml/` file (66 LFS-tracked files
compared): **exactly** the five `har_post` packages differ (both `model.mlmodel`
and `weight.bin`); all `decoder_pre`, `duration_t*`, and `f0ntrain_t*` packages
are byte-identical between local and the previously published HF revision.
Duration / F0Ntrain / DecoderPre carry no iSTFT weights and were left untouched.

`weight.bin` is byte-identical across all five buckets (only the per-bucket
`model.mlmodel` fixed input shape differs), consistent with the fix living in the
shared inverse-DFT buffers.

### Old -> new SHA-256 (first 16 hex)

| Bucket | file | old (HF 2026-06-30) | new (fixed) |
| --- | --- | --- | --- |
| all 5 | `weights/weight.bin` | `d184cac257165a29` | `e4ada8b28c56a4ac` |
| 3s  | `model.mlmodel` | `5f86977304…` | `6050b421ac1b3785` |
| 7s  | `model.mlmodel` | `2952a767fe…` | `76bdb21faa362869` |
| 10s | `model.mlmodel` | `caba5fd23e…` | `ecac9febb39839f6` |
| 15s | `model.mlmodel` | `2b360357ad…` | `47a73915a21b2275` |
| 30s | `model.mlmodel` | `5453b011a7…` | `46a985077da79cdc` |

Full new 15s `weight.bin` digest:
`e4ada8b28c56a4acda6a88e7c6d076aa65a39051841597bc0c4c07a60afe5ac2` (39,353,848 B).

## Ship gate — per-bucket numerical parity vs PyTorch

Every shipped package was validated before upload with
`scripts/check_coreml_generator_from_dump.py` (GeneratorFromHar Core ML run on a
PyTorch-reference tensor dump captured by `scripts/capture_audio_parity_tensors.py`
from the frozen `outputs/swift_bench_inputs/{key}.json`), compute units
`cpuAndGPU`. Gate thresholds `corr >= 0.99`, `snr_db >= 35`. Reports under
`outputs/audio-parity/gate/report_{bucket}.json`.

| Bucket | correlation | SNR (dB) | max_abs_error | passes |
| --- | --- | --- | --- | --- |
| 3s  | 0.9999865 | 45.62 | 0.00318 | yes |
| 7s  | 0.9999896 | 46.74 | 0.00266 | yes |
| 10s | 0.9999892 | 46.59 | 0.00315 | yes |
| 15s | 0.9999894 | 46.66 | 0.00296 | yes |
| 30s | 0.9999897 | 46.78 | 0.00253 | yes |

Consistent with the fix note's underlying math parity (iSTFT vs `torch.istft`
max abs 8.9e-8; `disable_complex` generator vs complex generator 4.9e-7) and its
recorded 3s Core ML score (corr 0.99999 / SNR 45.7 dB).

## Upload — atomic binaries, then regenerated manifests

Ordering chosen so binaries land first and manifests reference exactly the
uploaded hashes. During the brief window between the two commits, a consumer
using the old manifest against new binaries would get a SHA-256 **mismatch and
reject** (fail-safe), never silent corruption.

1. **Binaries (commit `2e878c6a33c56b40de094ef8237bf15a83d233c5`)** — the five
   fixed packages uploaded in one commit via `huggingface_hub.HfApi.upload_folder`
   (`folder_path="coreml"`, `path_in_repo="coreml"`,
   `allow_patterns=["kokoro_decoder_har_post_{3,7,10,15,30}s.mlpackage/**"]`).
   10 LFS files (5 `model.mlmodel` + 5 `weight.bin`).
   Commit: `.../commit/2e878c6a33c56b40de094ef8237bf15a83d233c5`.

2. **Provenance download manifests** at the new revision (verifies HF now matches
   local; local `har_post` confirmed NOT clobbered — still `e4ada8b2`):

   ```bash
   REV=2e878c6a33c56b40de094ef8237bf15a83d233c5
   python scripts/download_models.py --repo-id mattmireles/kokoro-coreml \
     --revision $REV --sdk-profile starter --manifest-out /tmp/kokoro-dl-starter.json
   python scripts/download_models.py --repo-id mattmireles/kokoro-coreml \
     --revision $REV --sdk-profile full   --manifest-out /tmp/kokoro-dl-full.json
   ```

3. **Bundle build** (HF provenance verified against the download manifests;
   `hf_provenance_verified: true`, `hf_revision: 2e878c6a…`). Compile skipped —
   `compiled/` is excluded from every published manifest:

   ```bash
   node scripts/build_sdk_bundle.mjs --profile starter --output /tmp/kokoro-sdk-starter \
     --repo-id mattmireles/kokoro-coreml --revision $REV --download-manifest /tmp/kokoro-dl-starter.json
   node scripts/build_sdk_bundle.mjs --profile full --output /tmp/kokoro-sdk-full \
     --repo-id mattmireles/kokoro-coreml --revision $REV --download-manifest /tmp/kokoro-dl-full.json
   ```

4. **Metadata publish (commit `32399b333e809044c404c518cb3807a488e8f47d`)** —
   regenerated `HostedManifest.json`, `KokoroRuntimeManifest.json`,
   `sdk/{starter,full}/KokoroRuntimeManifest.json`, `sdk/SDKReleaseManifest.json`,
   README via the repo's own tooling (no hand-edited JSON):

   ```bash
   python scripts/prepare_hf_sdk_metadata.py --repo-id mattmireles/kokoro-coreml \
     --starter-bundle /tmp/kokoro-sdk-starter --full-bundle /tmp/kokoro-sdk-full \
     --output /tmp/kokoro-hf-sdk-metadata --upload
   ```

### Manifest hash regeneration evidence

- Top-level (starter) `HostedManifest.json`: version `starter-95d50d139587`; 15s
  `weight.bin` entry now `e4ada8b28c56a4ac` (was `d184cac257…`).
- `sdk/SDKReleaseManifest.json`: `sdk_commit 95d50d1395879e4c8d18d4afbf6db88d2bd81e53`
  (was `1a07578942…`), `hf_revision 2e878c6a…` (was `c02933e1…`).
- `sdk/full/KokoroRuntimeManifest.json` `model_packages` tree hashes refreshed:

  | Bucket | old tree_sha256 | new tree_sha256 |
  | --- | --- | --- |
  | 3s  | `f8fbcc5649f52d77` | `83c92b4854929713` |
  | 7s  | `d0f7b5d6edcd3aa1` | `00f944548c54b3e3` |
  | 10s | `4d95fac9a5723456` | `4245c65012624d93` |
  | 15s | `6fbf34e0c09090c0` | `156fbd526c9eac2f` |
  | 30s | `0650af6862c8e775` | `8702d88e909ca2f6` |

The model card (`README/hf-model-card.md` -> HF `README.md`) is a shape/architecture
reference with no checksums or bug-specific known-issue text; it was republished
unchanged by the metadata pipeline (no edits needed).

## Post-upload verification (end-to-end)

- HF `repo_info.lastModified` advanced **2026-06-30 -> 2026-07-15 01:19 UTC**;
  head sha `32399b33…`.
- **Forced fresh re-download** of `coreml/kokoro_decoder_har_post_15s.mlpackage/
  Data/com.apple.CoreML/weights/weight.bin` from HF (`hf_hub_download`,
  `force_download=True`) hashes to
  `e4ada8b28c56a4acda6a88e7c6d076aa65a39051841597bc0c4c07a60afe5ac2` — matches
  the fixed local export.
- Freshly published top-level `HostedManifest.json` 15s hash = `e4ada8b2` (fixed).
- Release gates: `node scripts/validate_sdk_bundle.mjs` passed for both starter
  and full bundles; `node scripts/check_sdk_drift.mjs` -> "SDK drift check passed".

## Left undone / follow-ups

- No 10s frozen listening input still (unchanged pre-existing follow-up); the 10s
  package was parity-gated here against its own `outputs/swift_bench_inputs/10s.json`
  tensor dump.
