# Kokoro Drop-In SDK v1 Plan

**Date:** 2026-06-28
**Status:** Implemented locally; final audit fixes in progress

## Executive Summary

Build a real Swift SDK that lets an app developer add local Kokoro TTS to an
iPhone or macOS app with one high-level API call. Use the proven Gist iOS
pattern: native Swift text prep with MisakiSwift, checked Kokoro vocab, voice
row lookup, manifest-backed model download/compile/cache, and the existing
Core ML synthesis runtime. Botnet's Node prep remains the parity oracle and
developer diagnostic path, not an app runtime dependency.

## Problem Statement

- **Symptom:** The model card promises `synthesize(text:voice:)`, but the
  current Swift library only accepts pre-tokenized `inputIds`, `attentionMask`,
  and `refS`.
- **Root Cause:** Text normalization, phonemization, vocab lookup, voice row
  selection, long-text chunking, model resource discovery, and developer-facing
  examples live in scripts or Botnet worker glue instead of a supported SDK
  surface.
- **Impact:** A developer can technically embed the Core ML pipeline, but cannot
  drop it into an iOS or macOS app without reverse-engineering scripts, ignored
  model assets, voice assets, and current worker conventions.

## Mode Definitions

| Mode | Behavior | Why it matters |
| --- | --- | --- |
| Prepared input mode | Caller passes `inputIds`, `attentionMask`, and `refS` directly to `KokoroPipeline`. | Existing tests and benchmark fixtures keep working. |
| Raw text SDK mode | Caller passes text, voice, and speed to `KokoroTTS`; SDK prepares inputs and synthesizes PCM. | This is the drop-in developer experience. |
| App-bundled resources | App target or package resource bundle contains models, voices, vocab, and hn-NSF weights. | Required for offline iOS/macOS use and App Store-friendly behavior. |
| Directory resources | SDK loads source `.mlpackage` files from an explicit URL and may reuse source-hash-matched `.mlmodelc` caches. | Keeps local development, benchmarks, and downloaded model bundles simple without trusting standalone compiled artifacts. |
| Hugging Face resources | Repo tools hydrate model and voice assets from `mattmireles/kokoro-coreml`. | Keeps large binaries out of git while giving developers one canonical download source. |
| Gist-style downloaded resources | App downloads a hashed manifest of `.mlpackage` directories, voices, vocab, and hn-NSF weights over a trusted channel, then compiles and caches `.mlmodelc` locally. | This is the shipping iOS precedent and avoids coupling release artifacts to one OS-compiled `.mlmodelc`; production callers must pin the release manifest or HF revision they trust. |
| Node oracle mode | Node runs Botnet-compatible prep in tests, scripts, CI, or macOS developer diagnostics. | Useful for drift detection; forbidden as a required iOS app dependency. |

## Goals and Non-Goals

### Goals

- [x] Make `kokoro-coreml` self-host the Botnet prep contract without requiring
      Botnet's `packages/kokoro-coreml-runtime` layout.
- [x] Add a native Swift SDK API:
      `let tts = try await KokoroTTS.load(...); try await tts.synthesize("Hello")`.
- [x] Ship a Swift text-prep layer, following Gist's `KokoroG2P` pattern, that
      produces the same prepared-input fields needed by `KokoroSynthesisRequest`.
- [x] Keep `KokoroPipeline` compatible with iOS 16+ and macOS 13+ while making
      the sibling `swift-tts/` package explicitly target the newest OS floor
      that makes raw-text synthesis reliable.
- [x] Support explicit resource discovery, background model compilation, typed
      errors, and no main-thread stalls.
- [x] Provide an example iOS/macOS app and a CLI smoke test that a new developer
      can run without reading benchmark internals.
- [x] Add parity/drift tests against Botnet's JS prep for representative text
      edge cases and existing bakeoff fixtures, with known Misaki-vs-eSpeak
      differences recorded instead of hidden.
- [x] Document a release artifact workflow for the large model assets instead of
      pretending they belong in git.
- [x] Make `scripts/download_models.py` and Hugging Face revision pinning the
      default way to hydrate large model/voice assets for SDK bundles.
- [x] Prove MisakiSwift packaging on macOS and iOS before exposing a public
      raw-text SDK API, reusing the Gist iOS bundle-fix decision where needed.
- [x] Record at least one physical-device iPhone raw-text synthesis smoke
      before claiming iOS SDK readiness.

### Non-Goals

- Do not redesign the Core ML model graph or bucket geometry.
- Do not require Python, Node, JavaScriptCore, or a worker daemon in the
  production iOS app path.
- Do not ship a hidden WebView/WASM bridge as the default iOS phonemizer.
- Do not chase native eSpeak-NG as the V1 app path unless MisakiSwift proves
  unusable for the product target.
- Do not promise non-English raw-text synthesis in V1; Gist's MisakiSwift path
  is English-only.
- Do not make benchmark or paper claims without rerunning the existing gates.

## Scope and Constraints

- **Scope:** `swift/`, `scripts/kokoro-prepare-input.*`, `kokoro.js/src`,
  `swift-tts/`, tokenizer/voice assets, SDK docs, example apps,
  package/resource manifests, Gist iOS audio precedents, Hugging Face artifact
  metadata, and tests.
- **Constraints:** `coreml/` is about 2.6 GB and ignored by git; voice binaries
  under `kokoro.js/voices/` are about 27 MB and ignored by git; SwiftPM resource
  bundling must use explicit target resources and `Bundle.module` access;
  Core ML compilation is expensive and must not run on the main thread.
- **Guardrails:** Keep `KokoroPipeline.synthesize(inputIds:attentionMask:refS:)`
  stable for benchmarks and worker reuse. Add a higher-level SDK above it
  instead of breaking the existing contract.

## Target Boundary Rules

- **`KokoroPipeline` target:** Remains the low-level prepared-input synthesis
  target. It owns `KokoroSynthesisRequest`, `KokoroModelProvider`,
  `executeKokoroSynthesis(...)`, Core ML tensor binding, DSP, and existing
  benchmark compatibility. It must not grow raw-text prep, HF download logic,
  app-bundle discovery, or public SDK convenience API.
- **`swift-tts/` package:** New higher-level SwiftPM package that depends on
  the existing `swift/` package and exposes the `KokoroTTS` library product. It
  owns raw-text prep, MisakiSwift/native phonemizer binding, resource manifest
  validation, SDK model/provider cache, public `KokoroTTS` facade, diagnostics
  policy, and AVFoundation conveniences.
- **Platform split:** `KokoroPipeline` keeps its current low-level platform
  floor inside `swift/`. `swift-tts/` advertises the newer iOS/macOS floor
  required for reliable MisakiSwift and Core ML behavior. Prefer a reliable
  iOS 18+ raw-text SDK over a fragile older-OS compatibility story. Do not
  silently raise the low-level runtime's platform floor.
- **Provider seam:** App-bundle resources, source-hash-matched compiled caches,
  lazy loading, `prewarm(...)`, cache eviction, and compute policy belong in a
  new SDK model provider conforming to `KokoroModelProvider`, not in the
  existing `KokoroPipeline` class.
- **Resource source of truth:** Small runtime files are package-owned and
  checked in; large model and voice binaries are downloaded from pinned HF
  revisions and copied into generated SDK bundles.
- **Downloaded-resource seam:** The SDK should support Gist's manifest-backed
  download/compile/cache pattern as a first-class provider, not only
  app-bundled resources.

## Ground Truth Contracts (Do Not Violate)

- **Prepared input contract:** `input_ids`, `attention_mask`, `ref_s`, `voice`,
  `speed`, and `canonical_duration_s` must stay compatible with
  `KokoroSynthesisRequest` and Botnet's `PreparedKokoroInput`.
- **Token cap:** Active tokens are capped at
  `PipelineConstants.maxCallerChunkTokens == 450`; padding length is not the
  active-token count.
- **Duration token buckets:** Prepared inputs pad to one of
  `[32, 64, 128, 256, 320, 384, 512]`.
- **Voice embedding:** `ref_s` is exactly 256 floats selected from the voice
  `.bin` rows by clamping `phonemeCount - 1`.
- **Runtime assets:** Models, vocab, voices, and hn-NSF weights must be tied
  together by a manifest with versions and hashes.
- **G2P contract:** V1 raw-text prep uses MisakiSwift plus a checked Kokoro vocab
  table, matching the Gist app pattern. Botnet/eSpeak parity is a drift signal,
  not a requirement for byte-identical phoneme strings in V1.
- **Chunking contract:** V1 text chunking starts from Botnet's Mac fleet
  chunker at
  `/Users/mm/Documents/GitHub/botnet/apps/kokoro-worker/Sources/KokoroWorkerCore/TextChunker.swift`.
  The SDK may apply the same deliberate iPhone/Gist cap change
  (`maxChunkSeconds` defaulting to 15 seconds instead of Botnet's static
  30 seconds), but the sentence-boundary, abbreviation, protected punctuation,
  word-packing, character-window, whitespace-normalization, speed, and
  `maxCharacters` fallback heuristics must remain fleet-identical unless a
  phase audit approves a documented divergence.
- **Manifest provenance:** Runtime manifests must include HF repo ID, HF
  revision, per-`.mlpackage` tree digests, file counts, byte sizes, vocab hash,
  voice hashes, hn-NSF weights hash, bundle profile, SDK commit, and minimum
  platform versions.
- **Resource path safety:** Manifest paths must be relative, canonical, and
  confined to the bundle root. Reject absolute paths, `..`, symlinks that escape
  the bundle, and non-canonical model package names before hashing or loading.
- **Core ML loading:** SDK must require source `.mlpackage` directories and may
  reuse `.mlmodelc` caches only when a sidecar proves they were compiled from
  the verified source package tree.
- **On-device compilation:** Source `.mlpackage` directories may be downloaded
  and compiled on device with `MLModel.compileModel(at:)`; hosted downloads do
  not distribute platform-compiled `.mlmodelc` caches.
- **Compute policy:** iPhone production default follows staged loading:
  decoder-pre on `.cpuAndNeuralEngine`; duration, F0Ntrain, and generator on
  `.cpuAndGPU`, with a session-sticky `.cpuOnly` degradation path after Core ML
  load or predict failures.
- **Threading:** Resource loading, model compilation, and synthesis must be safe
  to call from Swift concurrency without blocking the main actor.
- **Diagnostics privacy:** SDK diagnostics must never persist raw text or
  phoneme mirrors by default. Default diagnostics are counters, hashes, and
  typed error codes; raw diagnostic payloads require explicit caller opt-in.
- **iOS readiness:** Simulator builds prove compile/resource wiring only. iOS
  SDK readiness requires physical-device raw-text synthesis evidence.

## Already Shipped (Do Not Re-Solve)

- **Swift pipeline:** `swift/Package.swift` exposes the `KokoroPipeline` library
  for iOS 16+ and macOS 13+.
- **Core ML orchestration:** `swift/Sources/KokoroPipeline/KokoroPipeline.swift`
  loads duration, F0Ntrain, decoder-pre, and generator packages and synthesizes
  from prepared inputs.
- **Shared executor:** `executeKokoroSynthesis(...)` is the single path used by
  the library and benchmark runners.
- **PCM stitching:** `swift/Sources/KokoroPipeline/PcmJoiner.swift` already joins
  chunked PCM with a small crossfade.
- **Botnet prep copy:** `scripts/kokoro-prepare-input.mjs` and
  `scripts/kokoro-prepare-input.py` are byte-for-byte identical to Botnet today.
- **Tokenizer assets:** `kokoro.js/src/phonemize.js`, `splitter.js`, `voices.js`,
  and voice `.bin` files exist locally.
- **iOS resource precedent:** `ios-bench/project.yml` bundles Core ML packages
  as app resources, but this SDK V1 still requires source `.mlpackage`
  manifests for validation and treats compiled output as a cache.
- **Gist iOS app precedent:** `/Users/mm/Documents/GitHub/gist/packages/ios-app`
  ships local listen mode with `KokoroG2P.swift`, `SynthEngine.swift`,
  `KokoroModelStore.swift`, `VoiceTable.swift`, and `SlideAudioStore.swift`.
  It uses MisakiSwift plus `kokoro-vocab.json`, downloads model artifacts from
  a manifest, compiles `.mlpackage` directories on device, caches `.mlmodelc`,
  uses the shared Swift `KokoroPipeline`, and falls back to shared cache/server
  synth only when local synth is unavailable or fails.

## Fresh Baseline (Current State)

- **Architecture:** Core ML synthesis is Swift-first, but raw-text prep is not a
  public SDK surface.
- **Current API gap:** `README/hf-model-card.md` shows
  `pipeline.synthesize(text: "Hello world", voice: "af_heart")`; the actual
  Swift API requires prepared tensors.
- **Current script gap:** Running `node scripts/kokoro-prepare-input.mjs` from
  this repo fails by default because it looks for
  `packages/kokoro-coreml-runtime`, the Botnet runtime layout.
- **Dependency gap:** Even with `KOKORO_COREML_ROOT` set, JS prep requires the
  `phonemizer` npm package to resolve from the current environment.
- **Asset gap:** `coreml/`, `outputs/`, and `kokoro.js/voices/*.bin` are ignored
  by git; the SDK needs a model/asset artifact story.
- **HF download baseline:** `scripts/download_models.py` already targets
  `mattmireles/kokoro-coreml` and downloads the curated Core ML packages plus
  voice `.bin` files.
- **HF live snapshot:** On 2026-06-28 the live HF repo was
  `c02933e179932e51909ff3b29466a7debac7d0e6`, last modified
  2026-06-10, with 23 curated `coreml/*.mlpackage` packages and voice assets.
- **HF SDK gap:** The live HF file list does not expose an SDK runtime manifest,
  SDK bundle profiles, `hnsf_weights.json`, or tokenizer/vocab config as public
  runtime files. It is a model/voice artifact repo today, not yet a drop-in SDK
  artifact repo.
- **Gist iOS raw-text baseline:** Gist already solved the app-path question in
  Swift: MisakiSwift G2P, checked vocab, fleet-compatible voice-row lookup,
  15-second chunking, one synthesis in flight, first-class model download, and
  on-device CPU fallback after Core ML failures. It deliberately does not embed
  Node in the iOS app.
- **Botnet chunker baseline:** Botnet's canonical production chunker lives at
  `/Users/mm/Documents/GitHub/botnet/apps/kokoro-worker/Sources/KokoroWorkerCore/TextChunker.swift`.
  Gist's `TextChunker.swift` is explicitly documented as a copy of that fleet
  chunker with only the iPhone 15-second cap changed from Botnet's static
  30-second cap. The SDK implementation must preserve this source-of-truth
  relationship.
- **Platform-floor baseline:** Gist keeps MisakiSwift at the app layer because
  it raises the platform floor beyond the lower-level fleet runtime. This SDK
  should preserve the same split: raw-text convenience can have a higher floor;
  prepared-input `KokoroPipeline` should not.
- **Config gap:** `checkpoints/config.json` is tracked as a symlink to an
  external user path. SDK bundle generation must reject this even on machines
  where the symlink resolves; V1 should use a real checked-in vocab resource
  derived from `_kokoro_vocab.json` or a copied upstream Kokoro config with
  recorded provenance.
- **hn-NSF weights gap:** The generated verified weights live under ignored
  `outputs/swift_bench_inputs/hnsf_weights.json`, while the tracked root
  `hnsf_weights.json` currently carries `weights_sha256: "unverified"` and does
  not match the generated verified file. V1 must promote or regenerate a
  verified checked-in copy before any clean-checkout bundle claim.
- **Model set drift:** Local `coreml/` contains experimental and legacy packages
  not present on HF. V1 SDK bundles should depend only on the curated HF runtime
  set unless a missing local package is intentionally published and documented.
- **Known good smoke:** The same JS prep script succeeds in Botnet and emits a
  32-token padded input, 32-entry attention mask, and 256-float `ref_s` for
  "Hello world".

## Solution Overview

Use a thin adapter over the working pipeline and the Gist iOS app's proven
shape. Do not fight Core ML, do not move dynamic text prep into the model, do
not invent a second synthesis path, and do not require Node in an iPhone app.

```
App text
  |
  v
KokoroTextProcessor
  - normalize / MisakiSwift phonemize
  - vocab lookup
  - voice row lookup
  - chunk and token-budget guard
  |
  v
PreparedKokoroInput[]
  |
  v
KokoroTTS actor
  - resource manifest or app bundle
  - lazy model provider with staged compute policy
  - background download / compile / load
  |
  v
KokoroPipeline / executeKokoroSynthesis
  |
  v
24 kHz mono PCM + AVAudioPCMBuffer convenience
```

## Implementation Phases

> Do one phase at a time. Verify before proceeding.

### Required Skills

Use the workflow skills explicitly when executing this plan:

- **Whole plan:** `execute-plan` for normal phase-by-phase implementation.
  Use `execute-plan-hardcore` only if the user explicitly asks for the full
  post-execution audit/fix loop.
- **Every phase:** `phase-audit` before moving to the next phase. If delegated
  review is unavailable, run the local phase-audit rubric.
- **Plan and docs edits:** `markdown` for this plan, `write-notes` for
  implementation evidence under `README/Notes/`, and `david-ogilvy` only for
  reader-facing SDK copy polish after the API is real.
- **Final release:** `deploy`, `git-commit`, and `git-push` only when the user
  authorizes release/push side effects.

| Phase | Required skills | Why |
| --- | --- | --- |
| Phase 0 | `botnet`, `debug`, `phase-audit` | Compare the copied prep layer against Botnet and prove the local oracle works. |
| Phase 1 | `ilya-sutskever`, `coreml`, `debug`, `phase-audit` | Keep the Swift package boundary simple while proving the raw-text backend and platform floor. |
| Phase 2 | `coreml`, `debug`, `phase-audit` | Make runtime assets canonical, hashed, and safe for clean checkouts. |
| Phase 3 | `ilya-sutskever`, `coreml`, `documentation`, `phase-audit` | Port text prep carefully and document the public prep contracts. |
| Phase 4 | `coreml`, `debug`, `phase-audit` | Build reproducible HF/model bundle tooling with stable hashes and manifests. |
| Phase 5 | `ilya-sutskever`, `coreml`, `debug`, `documentation`, `phase-audit` | Add the drop-in SDK facade without contaminating the low-level pipeline. |
| Phase 6 | `coreml`, `audio-judge`, `phase-audit` | Validate app integration, generated audio, and real-device behavior. |
| Phase 7 | `markdown`, `write-notes`, `david-ogilvy`, `deploy`, `phase-audit` | Publish accurate docs, notes, model-card updates, and release artifacts. |

### Phase 0: Make the Copied Prep Layer Self-Hosting

**Goal:** `kokoro-coreml` can prepare text without Botnet's directory layout,
and the JS prep path becomes the parity oracle for the SDK.

**Required skills:** `botnet`, `debug`, `phase-audit`.

**Tasks:**

- [x] Fix `scripts/kokoro-prepare-input.mjs` so its default runtime root is the
      current repo root when `kokoro.js/src/phonemize.js` is present.
- [x] Add an explicit `--runtime-root` argument to both prep scripts; environment
      variables remain optional, not required.
- [x] Add `--botnet-root` to the parity script; default to
      `/Users/mm/Documents/GitHub/botnet` only as a local convenience.
- [x] Add a small top-level package script or documented `npm --prefix kokoro.js`
      command so `phonemizer` installs where the script can resolve it.
- [x] Add `scripts/compare_botnet_prepare_input.mjs` that runs this repo and
      the Botnet script selected by `--botnet-root` on the same fixture list.
- [x] Compare the full prepared-input contract: `key`, `text`, `voice`, `speed`,
      `input_ids`, `attention_mask`, `ref_s`, `canonical_duration_s`, optional
      `num_tokens`, and optional `hnsf_weights_sha256`. Fields intentionally
      absent from the JS oracle must be listed explicitly.
- [x] Add fixtures for empty text, whitespace, abbreviations, initials, numbers,
      currency, URLs/emails, quotes, punctuation runs, emoji, long text near
      450 active tokens, `af_heart`, and one British voice.
- [x] Record the exact current behavior for unsupported or unknown phonemes:
      unknown chars are dropped during vocab lookup, then BOS/EOS and padding
      still apply unless the phonemizer returns empty output.

**Verification:** `node scripts/kokoro-prepare-input.mjs --runtime-root
/Users/mm/Documents/GitHub/kokoro-coreml --text-file ...` works from this repo
without `KOKORO_COREML_ROOT`; `node scripts/compare_botnet_prepare_input.mjs
--botnet-root /Users/mm/Documents/GitHub/botnet --fixtures
tests/fixtures/kokoro-text-prep/*.json --compare full` passed 12 fixtures
against Botnet; `uv run python scripts/kokoro-prepare-input.py --runtime-root
/Users/mm/Documents/GitHub/kokoro-coreml ...` generated the expected 32-token
smoke input; `npm --prefix kokoro.js test` passed 276 tests.

---

### Phase 1: `swift-tts/` Package and Gist-Proven Phonemizer Spike

**Goal:** Prove the Gist iOS raw-text backend and target boundaries before
public SDK API work starts.

**Required skills:** `ilya-sutskever`, `coreml`, `debug`, `phase-audit`.

**Tasks:**

- [x] Leave `swift/Package.swift` as the low-level prepared-input package for
      `KokoroPipeline`; do not add MisakiSwift or raw-text targets there.
- [x] Create `swift-tts/Package.swift` with a `KokoroTTS` library product that
      depends on `../swift` and imports `KokoroPipeline`.
- [x] Add a minimal `swift-tts/Sources/KokoroTTS` scaffold, but do not expose
      `KokoroTTS.synthesize(text:)` yet.
- [x] Add MisakiSwift as the V1 raw-text phonemizer dependency in the
      `swift-tts/` package only. Do not add it to `KokoroPipeline`.
- [x] Reuse the Gist packaging decision where applicable: pin the fork/revision
      or upstream tag, include license notices, and confirm resource layout does
      not break iOS code signing.
- [x] Add `KokoroMisakiPhonemizer` behind a `KokoroPhonemizer` protocol so a
      future eSpeak backend can be added without changing the public API.
- [x] Add a short probe executable or test target that phonemizes the same
      fixture texts on macOS using MisakiSwift, and compile-validates the
      package for iOS Simulator.
- [x] Compare probe output against Botnet's Node/eSpeak oracle and write a
      drift table: exact matches, acceptable phoneme differences, dropped
      characters, empty-output behavior, and voice-row-selection consequences.
- [x] Record the dependency, license, platform floor, bundle resource, and App
      Store decision to `README/Notes/kokoro-drop-in-sdk-v1.md`.
- [x] Define the SDK diagnostics policy: default diagnostics are counters,
      stable hashes, and typed error codes; raw text and phoneme strings require
      explicit caller opt-in and are never persisted by the SDK.

**Verification:** `swift test --package-path swift` still builds the low-level
package without MisakiSwift; `swift test --package-path swift-tts` builds the
new raw-text package and imports `KokoroPipeline`; the MisakiSwift probe builds
and runs offline on macOS; the package compile-validates for generic iOS
Simulator; the notes record that MLX Swift does not support iOS Simulator
runtime evaluation and therefore physical-device proof remains mandatory;
the notes record platform floor and license/provenance decisions; and Phase 1
produces a reviewed drift table against the
Botnet JS/eSpeak oracle. If packaging, offline operation, or code signing
fails, execution stops before Phase 2. Exact phoneme-string equality with
eSpeak is not required for V1 unless the drift causes tokenization or
audio-quality failures.

---

### Phase 2: Runtime Asset Source of Truth

**Goal:** Clean checkouts have real small runtime assets, and generated bundles
cannot accidentally absorb machine-local symlinks or stale files.

**Required skills:** `coreml`, `debug`, `phase-audit`.

**Tasks:**

- [x] Add checked-in SDK runtime resources under
      `swift-tts/Sources/KokoroTTS/Resources/KokoroRuntime/`:
      - `kokoro-vocab.json`, copied or regenerated from the same Kokoro vocab
        source used by the Gist iOS app, with recorded SHA/provenance.
      - `hnsf_weights.json`, regenerated or promoted from the verified source
        so `weights_sha256` is not `"unverified"`.
- [x] Add `scripts/verify_runtime_assets.py` to reject symlinked configs,
      compare vocab hashes, compare hn-NSF weights hashes, and prove the package
      resources are the canonical SDK inputs.
- [x] Update prep scripts only if needed so local oracle paths can use the same
      real vocab source as the SDK without relying on `checkpoints/config.json`.
- [x] Leave `checkpoints/config.json` out of SDK bundle generation entirely
      unless it has been replaced with a real copied file and provenance.
- [x] Add a small runtime asset manifest fragment with hashes for vocab and
      hn-NSF weights; later bundle manifests must embed these exact hashes.

**Verification:** `python3 scripts/verify_runtime_assets.py` passes from a clean
checkout; the script fails on symlinked vocab config, missing hn-NSF weights,
`weights_sha256: "unverified"`, or hash drift between package resources and any
generated benchmark copies.

---

### Phase 3: Native Swift Text Prep and Public Types

**Goal:** The production SDK can prepare raw text in-process on iOS/macOS using
the Gist-proven Swift phonemizer path and checked runtime assets.

**Required skills:** `ilya-sutskever`, `coreml`, `documentation`,
`phase-audit`.

**Tasks:**

- [x] Create `swift/Sources/KokoroPipeline/KokoroPreparedInput.swift` with the
      low-level prepared-input contract only: token IDs, attention mask, `refS`,
      speed, optional `canonicalDurationSeconds`, optional `numTokens`, optional
      `hnsfWeightsSHA256`, and metadata needed for existing fixtures.
- [x] Create `swift-tts/Sources/KokoroTTS/KokoroVoiceID.swift`,
      `KokoroSynthesisOptions.swift`, `KokoroAudio.swift`,
      `KokoroTextProcessor.swift`, and `KokoroPhonemizer.swift` in the
      `swift-tts/` package.
- [x] Adapt Gist's `KokoroG2P.swift` behavior: MisakiSwift phonemization,
      checked Kokoro vocab lookup, unknown-vocab drop behavior, BOS/EOS token
      framing, and phoneme UTF-16 count for voice-row selection.
- [x] Adapt Gist's `VoiceTable.swift` behavior: little-endian float32 `.bin`
      loading, 256-float rows, supported voice list, and
      `rowIndex = clamp(phonemeCount - 1, 0, rowCount - 1)`.
- [x] Port Botnet's canonical
      `/Users/mm/Documents/GitHub/botnet/apps/kokoro-worker/Sources/KokoroWorkerCore/TextChunker.swift`
      into `swift-tts/Sources/KokoroTTS/TextChunker.swift`, preserving fleet
      sentence-boundary, abbreviation, protected punctuation, word-packing,
      character-window, whitespace-normalization, speed, and `maxCharacters`
      fallback behavior.
- [x] Apply only the Gist/iPhone-specific chunk cap change on top of the Botnet
      chunker: `maxChunkSeconds` is configurable and defaults to 15 seconds for
      the SDK starter profile instead of Botnet's static 30 seconds.
- [x] Add a chunker parity fixture/test set against the Botnet source behavior
      for abbreviations, initials, decimals, punctuation runs, protected
      commas, hyphenated words, whitespace normalization, long text, speed
      changes, and `maxCharacters` recursive token-budget fallback.
- [x] Port deterministic Botnet-compatible non-phonemizer prep logic to Swift:
      language gating, duration token-budget guard, metadata preservation, and
      typed validation.
- [x] Define unknown-token behavior as Gist-compatible by default: characters
      without vocab entries are dropped during tokenization; throw only when no
      model tokens remain after phonemization/tokenization.
- [x] Keep Node/WASM and Python prep as test/dev oracles only; they must not be
      required by an iOS app.

**Verification:** `swift test --package-path swift-tts --filter KokoroText` passes;
Swift-prepared fixtures match the Gist path for phonemes, token IDs, selected
voice row, `ref_s`, and metadata fields; `TextChunker` fixtures match Botnet's
fleet chunker behavior except for the documented configurable 15-second SDK cap;
Botnet JS/eSpeak comparison produces the approved drift report; no app code
path imports Node, Python, or Botnet.

---

### Phase 4: Reproducible HF Downloads and SDK Bundle Builder

**Goal:** Developers can generate or download a starter SDK runtime bundle from
pinned artifacts with verifiable provenance.

**Required skills:** `coreml`, `debug`, `phase-audit`.

**Tasks:**

- [x] Extend `scripts/download_models.py` with explicit `--repo-id`,
      `--revision`, `--manifest-out`, and SDK-oriented download modes so a clean
      checkout hydrates exactly the model/voice revision used to build a bundle.
- [x] Add `scripts/inspect_hf_artifacts.py` or equivalent to record HF repo ID,
      revision, last-modified time, model package list, file counts, byte sizes,
      and missing SDK metadata.
- [x] Add `scripts/hash_mlpackage_tree.py` to compute stable per-package digests
      over every file inside each `.mlpackage`.
- [x] Add `KokoroRuntimeManifest.json` schema covering SDK commit, HF repo ID,
      HF revision, model package names, per-package digests, file counts, byte
      sizes, bucket set, vocab hash, voice hashes, hn-NSF weights hash,
      supported languages, minimum platform versions, and bundle profile.
- [x] Add `scripts/build_sdk_bundle.mjs` to assemble a release-ready starter
      bundle from pinned `coreml/`, `kokoro.js/voices/`, and checked-in runtime
      assets. The builder may hydrate missing model/voice files via
      `scripts/download_models.py`, but must fail on symlinks, path escapes,
      missing hashes, or digest mismatches.
- [x] Add a Gist-style hosted-manifest output mode that emits
      `{ version, files: [{ path, bytes, sha256 }] }` for file-by-file hosting
      of `.mlpackage` directories, voice `.bin` files, vocab, hn-NSF weights,
      and `KokoroRuntimeManifest.json`.
- [x] Decide whether `models.gist.is/coreml/v1` remains only a Gist app
      endpoint or becomes a generated example of the public SDK-hosted manifest
      shape. Do not hard-code Gist infrastructure into the SDK.
- [x] Support bundle profiles in this order:
      - `starter`: smallest documented bundle for a demo app, one default voice
        and the minimum required bucket set, matching Gist's single-bucket v1
        shape unless measurements require otherwise.
      - `custom`: explicit voices and buckets selected by the developer.
      - `full`: all production buckets and all supported English voices, only
        after starter/custom behavior is proven.

**Verification:** A clean checkout can run the documented pinned HF download
command, generate a starter bundle, generate a hosted manifest, validate
per-package and per-file digests against the manifests, and intentionally fail
when a model file, voice file, vocab, or hn-NSF weights file is modified.

---

### Phase 5: SDK Model Provider and Drop-In API

**Goal:** A developer can add the package, add resources, and synthesize speech
without reading benchmark code or touching the low-level pipeline.

**Required skills:** `ilya-sutskever`, `coreml`, `debug`, `documentation`,
`phase-audit`.

**Tasks:**

- [x] Add `KokoroResourceProvider` in `KokoroTTS` with cases for explicit
      generated-bundle directory URL, app bundle, package bundle, and downloaded
      cache roots. Generated bundle roots may include a `compiled/` directory;
      V1 does not support precompiled-only roots without source `.mlpackage`
      manifests.
- [x] Add a downloaded-resource provider modeled on Gist's `KokoroModelStore`:
      fetch remote manifest, compare local version and file sizes/hashes,
      download missing files with retry, exclude cache from iCloud backup,
      compile `.mlpackage` directories with `MLModel.compileModel(at:)`, and
      cache `.mlmodelc` next to source packages.
- [x] Add `KokoroSDKModelProvider` in `KokoroTTS` that conforms to
      `KokoroModelProvider`, validates the manifest, owns lazy model loading,
      handles `.mlpackage` off-main-thread compilation, reuses source-hash-
      matched `.mlmodelc` caches, caches selected buckets, and exposes
      `prewarm(...)`.
- [x] Keep the existing `KokoroPipeline` class backward-compatible; do not move
      app resource discovery, HF download, or SDK cache policy into it.
- [x] Add `KokoroTTS` as an actor or otherwise concurrency-safe facade over
      `KokoroTextProcessor`, `KokoroResourceProvider`, and
      `KokoroSDKModelProvider`.
- [x] Public API target:
      `let tts = try await KokoroTTS.load(resources: .appBundle(.main))`
      and `let audio = try await tts.synthesize("Hello world", voice: .afHeart)`.
- [x] Add `prepare(_:voice:options:)` for developers who want to inspect or
      cache prepared inputs without running Core ML.
- [x] Add internal chunk stitching for text longer than one duration model can
      accept; use `PcmJoiner` for PCM output and expose raw chunks only as an
      advanced API.
- [x] Add `AVAudioPCMBuffer` convenience creation while keeping the core return
      type as raw `[Float]` plus sample rate and metadata.
- [x] Add clear default compute policy and an escape hatch for `MLComputeUnits`
      per stage without exposing benchmark-only complexity in the common path.
- [x] Make the default iPhone compute policy match the Gist app: decoder-pre on
      `.cpuAndNeuralEngine`, duration/F0Ntrain/generator on `.cpuAndGPU`, and a
      session-sticky `.cpuOnly` retry after Core ML load or predict failure.
- [x] Add typed errors for missing model, missing voice, bad hash, path escape,
      unsupported voice, empty text, invalid speed, input too long, empty
      phonemizer output, Core ML load failure, synthesis cancellation, and
      invalid audio.

**Verification:** `swift test --package-path swift-tts` passes 40 tests
(2 MLX-runtime tests skipped by design unless `KOKORO_RUN_MISAKI_RUNTIME_TESTS=1`
is set); `swift test --package-path swift` passed 46 tests after the Phase 5
surface landed; the Xcode-built `kokoro-sdk-smoke` synthesizes
`"Hello world."` from `/tmp/kokoro-sdk-starter-compiled` with
`samples=37800 sampleRate=24000 duration=1.575`. Direct SwiftPM CLI execution
of the smoke currently fails before synthesis because upstream MLX Swift does
not copy `mlx-swift_Cmlx.bundle/default.metallib` into `.build`; the Xcode app
build embeds that resource bundle and is the supported app-validation path for
this phase. Fresh consumer fixture, app UI, and physical-device validation are
tracked in Phase 6.

**Phase audit:** Read-only Codex cross-agent audit on 2026-06-28 initially
graded this phase Architecture C / Correctness risk C / Complexity debt B and
blocked commit-readiness. Fixes applied after that audit: downloaded hosted
manifest paths reject absolute, `..`, empty, and backslash components before
local writes or remote URL construction; hosted version changes clear the
compiled cache; the dead public `KokoroTTS()` initializer was removed; missing
or malformed voice assets map to public `KokoroError` cases; and the resource
provider wording no longer implies precompiled-only bundle support.
The second cross-agent pass found compiled-cache trust, parent-symlink,
cancellation, model-set consistency, and ignored `maxChunkSeconds` issues. Those
were fixed by making `KokoroSDKModelProvider` internal, validating compiled
cache and runtime/cache/model path components, preserving hosted-download
cancellation, failing fast on missing duration or bucket stage packages, honoring
`KokoroSynthesisOptions.maxChunkSeconds`, requiring every manifest
`duration_token_sizes` entry to have a matching duration model, and adding
targeted tests.
The final focused cross-agent pass found unsafe bundle-output deletion,
self-consistent but unverified HF provenance, host-distributed compiled caches,
root-symlink escapes, unsupported manifest-schema acceptance, stale starter
voice constants, and British voices using the U.S. Misaki path. Those were fixed
by adding a destructive-output guard and bundle marker, requiring
`--download-manifest` unless local provenance is explicitly allowed, excluding
`compiled/` from `HostedManifest.json`, rejecting symlinked roots, enforcing
schema version 1, aligning starter constants to `af_heart`, and selecting the
British Misaki phonemizer for `b*` voices.
The startup path is now explicitly app-safe: `KokoroTTS.load` runs synchronous
manifest/digest/vocab/hn-NSF validation inside `Task.detached`, does not compile
Core ML models, and no longer initializes Misaki/MLX until text prep is
actually requested. `KokoroFacadeTests/testFacadeLoadDefersModelCompilationAndMisakiSetup`
loads a fake-model bundle from `MainActor`; it would fail if load compiled Core
ML packages or touched eager Misaki runtime setup. `swift test --package-path
swift-tts` passed 42 tests with 2 opt-in Misaki runtime skips after this fix.

---

### Phase 6: Examples and macOS/iOS Validation

**Goal:** Prove the SDK works as an app dependency, not only as local scripts.

**Required skills:** `coreml`, `audio-judge`, `phase-audit`.

**Tasks:**

- [x] Add `examples/KokoroDemoApp` with a minimal iOS UI: text field,
      voice picker, synthesize button, progress/error display, and playback.
- [x] Include two resource modes in the example when feasible: bundled starter
      resources for offline demos and downloaded manifest resources for the
      Gist-style app flow. The iOS demo app now has Download/Bundled modes;
      bundled mode loads `.appBundle(.main, subdirectory:)` when an app target
      includes a generated runtime directory.
- [x] Extend the Phase 5 `swift-tts` `kokoro-sdk-smoke` executable into a
      Phase 6 app/fixture smoke that writes a WAV and exercises bundled and
      downloaded resources.
- [x] Add a fresh consumer fixture outside `swift/` that depends on the package
      as a developer would and uses only public API.
- [x] Build and run macOS smoke with a generated starter bundle.
- [x] Build iOS simulator smoke for compile/resource validation, even though
      simulator performance is not meaningful.
- [x] Build the raw-text demo against the actual `KokoroTTS` platform floor
      chosen for reliability; do not imply iOS 16 raw-text support if the
      dependable path is iOS 18+.
- [x] Run mandatory physical-device smoke before iOS readiness: first call,
      warm call, long text chunking, background/foreground transition,
      cancellation, and memory pressure behavior.
- [x] Add the approved bounded memory-pressure recovery harness for physical
      devices because `devicectl sendMemoryWarning` is flaky on iOS 26. The
      accepted release standard is: allocate and touch a bounded pressure
      target, log baseline/peak/released physical footprint, release pressure,
      then complete a post-pressure recovery synthesis.
- [x] Compare SDK raw-text preparation against the legacy prepared-input
      request boundary for the JS phonemizer output of `Hello world.`.
- [x] Record device and Mac validation evidence in `README/Notes/`, not
      `README/Guides/`.

**Verification:** macOS fixture green; iOS app build green; physical-device
raw-text smoke green with evidence before any iOS SDK claim; no performance
claim is made without target-device timing. If no device is available, mark the
plan blocked for iOS release rather than complete.

**Progress:** `swift build --package-path examples/KokoroConsumerFixture`
passes; Xcode-built `KokoroConsumerFixture` runs on macOS from the starter
bundle and writes `/tmp/kokoro-consumer-fixture.wav`; Xcode-built
`kokoro-sdk-smoke` writes WAV output in both `--bundle` and
`--manifest-url file:///.../HostedManifest.json --cache-dir ...` modes; generic
iOS Simulator builds for `KokoroConsumerFixture` and `KokoroDemoApp` pass.
`KokoroDemoApp` builds, installs, and launches on paired iPhone 15 Pro Max
`8A12AEE8-0136-50BE-8EB3-91650E467F15`; first-call downloaded-manifest
synthesis completed with
`KOKORO_DEMO_DONE samples=37800 sampleRate=24000 duration=1.575`. Physical
device scenario `all` completed on the final hardened harness with first call,
warm call, long text, typed SDK cancellation, and memory-footprint logging:
`KOKORO_DEMO_PID pid=4372 scenario=all`,
`all-first elapsedSeconds=7.4879`, `all-warm elapsedSeconds=2.0751`,
`all-long duration=83.345 elapsedSeconds=71.8878`,
`KokoroError.synthesisCancelled`, and
`physicalFootprintBytes=976439296`. `devicectl` suspend/resume succeeded for
PID 4372, but `sendMemoryWarning` failed in an earlier attempt with
`NSPOSIXErrorDomain 2`, so it is not counted as memory-pressure evidence.
At that point, physical readiness remained open pending a real
background/foreground transition and memory-pressure proof run. Prepared-input
parity is covered by
`KokoroTextProcessorTests/testPreparedInputBoundaryMatchesLegacyPreparedRequestContract`,
which uses the JS phonemizer output for `Hello world.` and proves Swift prep
produces the legacy request-contract padded IDs, mask, voice row, speed, and
`KokoroSynthesisRequest` when the phonemes and voice row match. It does not
claim perceptual audio parity. `swift test --package-path swift-tts` now passes
42 tests with 2 Misaki runtime tests skipped after this coverage landed.
`KokoroDemoApp` now emits `KOKORO_DEMO_SCENE_PHASE` sentinels on appear and
scene-phase changes; generic iOS build with `CODE_SIGNING_ALLOWED=NO` passed
after this change. A physical-device rerun against paired iPhone 12 Pro
`F383FC46-FD64-5346-AEC6-59E3E2F8C9CA` could not proceed because Xcode reported
the device was locked and unavailable. A later retry on paired iPhone 15 Pro Max
`8A12AEE8-0136-50BE-8EB3-91650E467F15` running iOS 26.5 completed the current
downloaded-manifest `all` scenario after a fresh signed build/install:
`KOKORO_DEMO_PID pid=4736 scenario=all resourceMode=downloaded`,
`all-first elapsedSeconds=7.865571`, `all-warm elapsedSeconds=2.022877`,
`all-long duration=83.345 elapsedSeconds=71.031005`,
`KokoroError.synthesisCancelled`, and
`physicalFootprintBytes=1015416856`. Activating `com.apple.Preferences` and
then returning to `com.mattmireles.KokoroDemoApp` produced real scene-phase
logs: inactive, background, inactive, active. `devicectl device process
sendMemoryWarning --pid 4736` still failed with `NSPOSIXErrorDomain 2` even
though `devicectl device info processes` confirmed PID 4736 was the demo app,
so the release gate now uses the approved safer replacement harness rather than
that flaky Apple tool. The demo app added `--scenario memory-pressure`, which
allocates and touches bounded memory, logs baseline/peak/released footprint,
releases the pressure, and then requires a recovery synthesis to pass. On the
same iPhone 15 Pro Max, fresh PID 4782 passed `--scenario memory-pressure
--memory-pressure-megabytes 384`: baseline `178193496`, peak `514590096`,
released `258508032`, recovery synthesis `37800` samples, and
`KOKORO_DEMO_SCENARIO_DONE scenario=memory-pressure`.
The demo app also now exposes both resource modes: `--resource-mode downloaded
--manifest-url ...` for hosted-manifest validation and `--resource-mode bundled
--bundle-subdirectory KokoroRuntime` for apps that embed a generated runtime
directory. Generic iOS build passed after this UI/automation change.
A later physical-device retry still found iPhone 12 Pro
`F383FC46-FD64-5346-AEC6-59E3E2F8C9CA` locked; the signed device build timed out
waiting for that destination. That older locked-device attempt is superseded by
the successful iPhone 15 Pro Max lifecycle and bounded memory-pressure recovery
runs above.

---

### Phase 7: Release Artifact and Documentation Pass

**Goal:** The SDK is publishable without local tribal knowledge.

**Required skills:** `markdown`, `write-notes`, `david-ogilvy`, `deploy`,
`phase-audit`.

**Tasks:**

- [x] Add `README/SDK.md` with install, resource bundle, API, playback, and
      troubleshooting steps.
- [x] Add `README/Notes/kokoro-drop-in-sdk-v1.md` during implementation to
      capture local decisions and any rejected tokenizer/backend paths.
- [x] Add release checklist commands for generating starter/full bundles,
      verifying manifests, running Swift tests, running JS parity tests, and
      building the example app.
- [x] Add `scripts/check_sdk_drift.mjs` or equivalent to verify Swift constants,
      JS/Python prep constants, runtime manifest keys, SwiftPM package identity,
      SDK docs, and model-card snippets agree on token caps, bucket sets, voice
      dimension, sample rate, and public API.
- [x] Add `scripts/prepare_hf_sdk_metadata.py` to assemble the lightweight HF
      metadata payload from validated starter/full SDK bundles, including the
      top-level starter manifests, profile manifests, release manifest,
      checksums, and model-card README.
- [x] Update the HF repo only after the public SDK API compiles: upload the SDK
      manifest, bundle-profile metadata, checksums, and model-card README that
      match the released SDK commit.
- [x] Add a rollback/deprecation note for the old model card snippet until a
      release tag includes the new API.
- [x] Decide whether this repo publishes only source + release assets or also a
      SwiftPM binary/resource artifact. Do not choose binary/resource artifact
      until package size and Xcode behavior are measured.

**Progress:** `README/SDK.md` now documents the current local-path SwiftPM
integration, bundled and downloaded resources, playback, starter/full bundle
commands, release gates, troubleshooting, and the V1 decision to publish source
plus downloadable resource bundles rather than a SwiftPM binary/resource
artifact. `README/hf-model-card.md` now points app developers at `KokoroTTS`
and marks old `KokoroPipeline` snippets as low-level.
The physical iOS readiness gate is now closed by iPhone 15 Pro Max evidence.
The HF metadata payload helper shares the downloader's env, repo `.env`, and
Hugging Face cache token lookup. The final hardened HF metadata upload reached
live revision `ccaa3199213acb163b14477e1eaee8b896740e69`.
`scripts/inspect_hf_artifacts.py` reports no missing SDK metadata and no
unresolved top-level `HostedManifest.json` files. The directly hydratable
starter manifest has 34 files and the release manifest points at SDK commit
`8a278df2d3d10ecb62c3eb65516cd09e31a182f3`. It records top-level hosted files
`runtime/hnsf_weights.json`, `runtime/kokoro-vocab.json`, and
`voices/af_heart.bin`; starter profile `starter-8a278df2d3d1` has 10 models / 1
voice and full profile `full-8a278df2d3d1` has 22 models / 28 supported English
voices.
Verification after the docs/drift slice: `node
scripts/check_sdk_drift.mjs`, `swift test --package-path swift-tts`, `swift
test --package-path swift`, `node scripts/compare_botnet_prepare_input.mjs
--botnet-root /Users/mm/Documents/GitHub/botnet --fixtures
tests/fixtures/kokoro-text-prep/*.json --compare full`, `python3
scripts/verify_runtime_assets.py`, and `git diff --check` all passed.
After the first Phase 7 audit found a bad SwiftPM package identity, an external
temp package using `.package(path: "/Users/mm/Documents/GitHub/kokoro-coreml/swift-tts")`
and `.product(name: "KokoroTTS", package: "swift-tts")` built successfully, and
`scripts/check_sdk_drift.mjs` now asserts that install contract.

**Verification:** README links all required assets and does not require Botnet
inside an iOS/macOS app; HF snapshot provenance is reproducible from a checked
command or script; SDK constants/docs/model-card drift check passes. Clean
checkout artifact build/run proof comes from the Phase 4-6 bundle and smoke
evidence, and must be rerun before remote HF publication.

## Executable Memory

- Regression test: `node scripts/compare_botnet_prepare_input.mjs --botnet-root /Users/mm/Documents/GitHub/botnet --fixtures tests/fixtures/kokoro-text-prep/*.json --compare full`
- Regression test: `python scripts/kokoro-prepare-input.py --runtime-root <repo-root> --text-file <fixture> --output /tmp/kokoro-prep.json --key smoke --voice af_heart --speed 1`
- Regression test: `python scripts/verify_runtime_assets.py`
- Regression test: `python3 scripts/inspect_hf_artifacts.py --repo-id mattmireles/kokoro-coreml --revision <released-sha> --verify-hosted-digests --output /tmp/kokoro-hf-artifacts.json`
- Regression test: `python scripts/download_models.py --repo-id mattmireles/kokoro-coreml --revision <released-sha> --sdk-profile starter --manifest-out /tmp/kokoro-download-manifest.json`
- Regression test: `node scripts/build_sdk_bundle.mjs --profile starter --compile-models 1 --output /tmp/kokoro-sdk-starter-compiled --repo-id mattmireles/kokoro-coreml --revision <released-sha> --download-manifest /tmp/kokoro-download-manifest.json`
- Regression test: `python3 scripts/prepare_hf_sdk_metadata.py --repo-id mattmireles/kokoro-coreml --starter-bundle /tmp/kokoro-sdk-starter-compiled --full-bundle /tmp/kokoro-sdk-full-compiled --output /tmp/kokoro-hf-sdk-metadata`
- Regression test: `python scripts/hash_mlpackage_tree.py coreml/kokoro_duration_t512.mlpackage`
- Regression test: `swift test --package-path swift`
- Regression test: `swift test --package-path swift-tts --filter KokoroText`
- Regression test: `swift test --package-path swift-tts --filter KokoroMisaki`
- Regression test: Xcode-built `swift-tts` `kokoro-sdk-smoke <bundle-root> "Hello world"` prints finite samples, sample rate, and duration.
- Phase 6 target: extend the smoke fixture to write WAV output and exercise hosted-manifest cache hydration.
- Regression test: `node scripts/check_sdk_drift.mjs`
- Regression test: `xcodebuild` or XcodeBuildMCP build of `Examples/KokoroDemoApp` on simulator.
- Manual release gate: physical iPhone raw-text smoke with first call, warm
  call, long text, cancellation, background/foreground, and memory-pressure
  evidence recorded in `README/Notes/`. `devicectl sendMemoryWarning` is not a
  release gate because it fails before the app receives a warning on iOS 26;
  the accepted safer default is the bounded pressure/release/recovery synthesis
  harness implemented by `--scenario memory-pressure`.

## Success Criteria

### Hard Requirements (Must Pass)

- [x] Fresh developer flow does not require Botnet, Python, Node, JavaScriptCore,
      or a worker daemon in an iOS/macOS app.
- [x] Public raw-text API is not exposed until MisakiSwift packaging, offline
      operation, license notices, platform floor, and reviewed Botnet drift
      evidence are proven.
- [x] Raw text API exists and works on iOS/macOS from public Swift types.
- [x] Prepared-input API remains backward-compatible for benchmarks.
- [x] Raw-text SDK reliability takes precedence over older iOS compatibility;
      V1 may require iOS 18+ if that is the stable Gist/MisakiSwift path.
- [x] SDK validates model, voice, vocab, hn-NSF, HF revision, and per-package
      `.mlpackage` digest compatibility before synthesis.
- [x] SDK can use app-bundled resources and Gist-style downloaded resources
      without hard-coding Gist infrastructure.
- [x] Long text is chunked deterministically and never silently truncates.
- [x] Missing assets produce actionable typed errors.
- [x] First-run model compilation cannot block the main actor.
- [x] Documentation and model card usage match the actual SDK package identity
      and public API.
- [x] iOS readiness includes physical-device background/foreground and
      memory-pressure evidence, not only synthesis completion.
- [x] SDK diagnostics are privacy-safe by default and never persist raw text or
      phoneme strings without caller opt-in.
- [x] Raw-text package floor is documented separately from the low-level
      prepared-input pipeline floor.

### Definition of Done

- [x] All Swift tests pass.
- [x] JS/Botnet parity fixtures pass.
- [x] Example app builds.
- [x] Smoke WAV is generated and finite.
- [x] SDK docs explain resource bundle setup and size tradeoffs.
- [x] Starter SDK bundle can be generated and verified from a clean checkout and
      pinned HF revision.
- [x] Hosted-manifest SDK cache path can download, compile, reuse, and invalidate
      a starter bundle.
- [x] Physical iPhone smoke evidence is recorded or the iOS release is marked
      blocked, not complete.
- [x] Plan checkboxes updated with real evidence before execution is called
      complete.

## Open Questions

### Resolved

- **Q:** Should V1 put raw text prep inside Core ML?
- **A:** No. Keep dynamic text prep on CPU and keep fixed-shape model buckets.

- **Q:** Should the SDK depend on Botnet at runtime?
- **A:** No. Botnet remains a parity oracle and source of copied prep logic.

- **Q:** Should all model assets be committed to git?
- **A:** No. `coreml/` is ignored and too large; release artifacts and manifests
  are the right boundary.

- **Q:** Should the public SDK break `KokoroPipeline.synthesize(...)`?
- **A:** No. Add `KokoroTTS` above it.

- **Q:** Can we download the model from
  [mattmireles/kokoro-coreml on Hugging Face](https://huggingface.co/mattmireles/kokoro-coreml)?
- **A:** Yes. `scripts/download_models.py` already downloads the curated Core ML
  packages and voice `.bin` files from that repo. Treat HF as the default binary
  source, but do not treat the current HF snapshot as a complete SDK runtime
  bundle until the manifest/config work lands.

- **Q:** Which native iOS phonemizer backend exactly matches Botnet's
  eSpeak/WASM output?
- **A:** V1 follows the Gist iOS app: MisakiSwift plus a checked Kokoro vocab in
  the app/SDK path. Botnet's eSpeak/WASM output remains a drift oracle, not a
  byte-equality requirement. Phase 1 must prove MisakiSwift packaging, offline
  operation, platform floor, license/App Store redistribution, macOS build, iOS
  simulator build, and reviewed drift evidence before public raw-text API work
  proceeds.

- **Q:** Why not embed the same small Node bridge Botnet uses?
- **A:** Botnet is controlled macOS worker infrastructure, so Node is a fine
  operational dependency there. Gist iOS proves the app pattern is different:
  native Swift G2P and Core ML runtime in the app, with server/cache fallback as
  product degradation. The SDK may keep Node as a dev/parity oracle, but a
  drop-in iOS SDK must not require Node, JavaScriptCore, a daemon, or a WebView
  bridge.

- **Q:** Should V1 copy the Gist app's model download/compile/cache pattern?
- **A:** Yes. Support app-bundled resources for offline demos and explicit
  bundle distribution, but make the hosted-manifest download path first-class:
  fetch manifest, validate hashes, download `.mlpackage` directories and
  sidecars, compile on device, and cache `.mlmodelc`.

- **Q:** Why use a separate `swift-tts/` package instead of adding `KokoroTTS`
  to `swift/`?
- **A:** Simpler and safer. `swift/` stays the prepared-input Core ML runtime
  with its current platform floor and no MisakiSwift dependency. `swift-tts/`
  can depend on `../swift`, own raw-text APIs and resources, and advertise the
  real MisakiSwift platform floor without risking accidental breakage for
  Botnet, benchmarks, or prepared-input users.

- **Q:** Should V1 publish a SwiftPM binary/resource artifact?
- **A:** Not yet. V1 publishes source plus generated downloadable resource
  bundles. A binary/resource artifact waits until package size and Xcode
  resource behavior are measured.

- **Q:** Should `hnsf_weights.json` and vocab config live in git, HF, or both?
- **A:** Keep small source-of-truth runtime files in git under the SDK target
  resources, include hashed copies in generated bundles, and record the same
  hashes in HF SDK metadata. Large models and voices stay in HF/downloaded
  artifacts.

- **Q:** Can iOS physical-device validation be deferred?
- **A:** No for an iOS-ready SDK. Simulator builds prove compile/resource wiring
  only; physical-device raw-text synthesis is a release gate. If no device is
  available, the iOS claim is blocked or downgraded.

- **Q:** Should unknown phonemes throw when vocab lookup drops every token?
- **A:** Default V1 behavior matches Botnet: throw only when the phonemizer
  returns empty output. Unknown vocab symbols are dropped, BOS/EOS still apply,
  and diagnostics report counters without persisting raw text or phonemes.

### Resolved Decisions

- **Q:** Should release bundles include every voice by default?
- **A:** No. The starter bundle ships `af_heart`. The full bundle ships all
  supported English voices (`af_*`, `am_*`, `bf_*`, `bm_*`) because V1 raw-text
  synthesis is English-only. Non-English voice prefixes are rejected by the SDK
  even when a custom bundle includes their embedding files.

- **Q:** Should the package publish resource bundles through SwiftPM?
- **A:** Not in V1. V1 publishes source plus generated downloadable resource
  directories. A SwiftPM binary/resource artifact waits until package size and
  Xcode resource behavior are measured.

- **Q:** What exact platform floor should `KokoroTTS` raw-text V1 advertise?
- **A:** Use the newest OS floor that makes the system reliable, even if that
  means iOS 18+. Do not spend V1 complexity on older-OS compatibility.
  `swift-tts/` documents the real floor while `swift/` keeps `KokoroPipeline`
  lower for prepared-input users.

## References

### Internal

- [Runtime boundary wiki](../Wiki/runtime-boundary.md)
- [CoreML export wiki](../Wiki/coreml-export.md)
- [Runtime boundary note](../Notes/kokoro-runtime-boundary.md)
- [Swift prefix rewrite plan](swift-prefix-rewrite-v1.md)
- [iPhone performance plan](kokoro-iphone-performance-v1.md)
- [Model card draft](../hf-model-card.md)
- `swift/Package.swift`
- `swift-tts/Package.swift`
- `swift/Sources/KokoroPipeline/KokoroPipeline.swift`
- `scripts/kokoro-prepare-input.mjs`
- `kokoro.js/src/phonemize.js`
- `/Users/mm/Documents/GitHub/botnet/scripts/kokoro-prepare-input.mjs`
- `/Users/mm/Documents/GitHub/botnet/apps/kokoro-worker/Sources/KokoroWorkerCore/KokoroWorkerCore.swift`
- `/Users/mm/Documents/GitHub/botnet/apps/kokoro-worker/Sources/KokoroWorkerCore/TextChunker.swift`
- `/Users/mm/Documents/GitHub/gist/packages/ios-app/App/Sources/Audio/KokoroG2P.swift`
- `/Users/mm/Documents/GitHub/gist/packages/ios-app/App/Sources/Audio/SynthEngine.swift`
- `/Users/mm/Documents/GitHub/gist/packages/ios-app/App/Sources/Audio/KokoroModelStore.swift`
- `/Users/mm/Documents/GitHub/gist/packages/ios-app/App/Sources/Audio/VoiceTable.swift`
- `/Users/mm/Documents/GitHub/gist/packages/ios-app/App/Sources/Audio/TextChunker.swift`
- `/Users/mm/Documents/GitHub/gist/packages/ios-app/project.yml`

### External

- [SwiftPM resource bundling docs](https://github.com/swiftlang/swift-package-manager/blob/main/Sources/PackageManagerDocs/Documentation.docc/BundlingResources.md)
- [Core ML compileModel(at:) docs](https://developer.apple.com/documentation/coreml/mlmodel/compilemodel%28at%3A%29)

## Error Handling and Edge Cases

| Scenario | Behavior | Fallback |
| --- | --- | --- |
| Empty or whitespace text | Throw `KokoroError.emptyText`. | Caller can skip synthesis. |
| Unknown voice | Throw `KokoroError.unsupportedVoice`. | Caller chooses another voice. |
| Voice asset missing | Throw `KokoroError.missingVoice`. | Bundle generator or app resources must be fixed. |
| Voice `.bin` corrupt length | Throw `KokoroError.malformedVoice`. | Rebuild resource bundle. |
| Unsupported language voice | Treat as unsupported voice unless the bundled backend explicitly supports it. | Use supported English voice. |
| Text exceeds token cap | Chunk before synthesis; if a single chunk still exceeds cap, throw. | Caller can shorten or SDK can split more aggressively. |
| Unknown phoneme not in vocab | Match Gist behavior: drop unknown token and record privacy-safe diagnostic counters. | BOS/EOS still apply; throw only if no model tokens remain. |
| Diagnostics requested | Return counters, hashes, and typed error codes by default. | Raw text or phonemes require explicit caller opt-in and are never persisted by SDK code. |
| Missing duration bucket | Throw with required model filename. | Rebuild bundle with required bucket. |
| `.mlpackage` first-run compile | Compile through the SDK model provider and cache compiled output when the root is writable. | Reuse source-hash-matched local caches; hosted downloads distribute source packages only. |
| Simulator run | Functional smoke only; no performance claim. | Use physical device for placement and latency. |
| No physical iPhone available | Mark iOS release blocked or downgrade claim. | Do not mark iOS SDK readiness complete. |
| Manifest path escape | Reject before hashing or loading. | Rebuild bundle with canonical relative paths. |
| Symlinked vocab/config asset | Reject before bundle generation. | Use checked-in SDK runtime vocab resource with recorded provenance. |
| Model digest mismatch | Reject before synthesis. | Re-download pinned HF revision or rebuild bundle. |
| Memory pressure | Lazy-load bucket models and keep recovery observable through the app harness. | App can discard and reload its `KokoroTTS` instance later; V1 does not expose a public model-release API. |
| Concurrent syntheses | Serialize model prediction or bound concurrency. | Expose queue/concurrency option later if needed. |
| Cancellation | Check cancellation between prep, model stages, and chunks. | Return partial nothing; no corrupt state. |
| Audio NaN/Inf | Validate finite PCM before returning. | Throw `KokoroError.invalidAudioOutput`. |
| Long text stitching | Use deterministic chunk order and `PcmJoiner`. | Expose raw chunks for advanced callers. |

## Degradation and Rollback

**Degradation Modes:**

- **If MisakiSwift packaging fails:** keep the lower-level prepared-input API
  and mark the raw-text SDK plan blocked until the plan is revised. Do not ship
  a raw-text iOS claim that needs hidden Node, WebView, or daemon behavior.
- **If Misaki-vs-eSpeak drift is audible or breaks tokenization:** keep Botnet's
  prepared-input oracle and add a new backend decision phase before public API.
  Do not pretend the phonemizer is byte-parity-compatible when it is not.
- **If hosted-manifest download fails:** use already-ready verified cache when
  hashes and manifest version prove it is safe; otherwise surface a typed model
  unavailable error.
- **If Core ML staged policy fails on device:** latch `.cpuOnly` for the session
  and retry locally before surfacing failure.
- **If full resource bundle is too large:** ship starter/custom bundle flow
  first and document full bundle as optional.
- **If compiled-cache discovery is flaky:** require explicit resource URL
  injection and fall back to source package compilation.
- **If HF update is delayed:** keep `scripts/download_models.py` pointed at the
  last known good model/voice revision and ship SDK source/docs separately; do
  not update the model card to the new raw-text API until the matching SDK
  release is available.
- **If physical iPhone evidence is unavailable:** ship macOS-only or
  prepared-input-only claims as appropriate; do not mark iOS SDK readiness
  complete.
- **If runtime asset provenance fails:** block bundle generation until the
  symlink, unverified hash, or digest mismatch is fixed.

**Rollback Plan:**

- **How to revert:** revert SDK facade files and docs; keep existing
  `KokoroPipeline` prepared-input API untouched.
- **Time to rollback:** one normal git revert if phases stay additive.
- **Data recovery needed:** No; all resources are generated or downloaded
  artifacts.

## Rollout and Gates

- **Feature flag:** none for library code; public API appears only when tests
  and docs are ready.
- **Rollout strategy:** local smoke -> fresh consumer fixture -> example app ->
  physical iPhone/macOS validation -> release artifact.
- **Kill switch:** developers can keep using `KokoroPipeline` prepared-input API
  directly if `KokoroTTS` text prep is not ready.
- **Release blocker:** iOS release requires physical-device raw-text synthesis
  evidence. Without it, the public claim must be macOS-only or explicitly
  experimental for iOS.

## Files Likely to Change

| File | Change Type | Notes |
| --- | --- | --- |
| `scripts/kokoro-prepare-input.mjs` | Modify | Self-host current repo root and `--runtime-root`. |
| `scripts/kokoro-prepare-input.py` | Modify | Same runtime-root behavior for Python oracle. |
| `scripts/compare_botnet_prepare_input.mjs` | Create | Fixture parity against Botnet prep with `--botnet-root` and full-contract comparison. |
| `scripts/verify_runtime_assets.py` | Create | Validate checked vocab/hn-NSF resources and reject symlinked configs. |
| `scripts/inspect_hf_artifacts.py` | Create | Record HF repo/revision/package-list provenance. |
| `scripts/hash_mlpackage_tree.py` | Create | Stable per-package tree digests for manifests. |
| `scripts/build_sdk_bundle.mjs` | Create | Assemble and hash SDK resource bundles from pinned artifacts. |
| `scripts/check_sdk_drift.mjs` | Create | Verify constants/docs/manifest/model-card snippets agree. |
| `scripts/prepare_hf_sdk_metadata.py` | Create | Assemble and optionally upload the lightweight HF SDK metadata payload. |
| `scripts/download_models.py` | Modify | Add repo/revision pinning and SDK download modes. |
| `swift/Package.swift` | Maybe modify | Keep `KokoroPipeline`; only add shared low-level types if required. Do not add MisakiSwift. |
| `swift-tts/Package.swift` | Create | Higher-level package exposing `KokoroTTS`, depending on `../swift` and MisakiSwift. |
| `swift/Sources/KokoroPipeline/KokoroPreparedInput.swift` | Create | Low-level prepared-input type only. |
| `swift-tts/Sources/KokoroTTS/Resources/KokoroRuntime/kokoro-vocab.json` | Create | Checked SDK vocab resource matching the Gist/Kokoro table. |
| `swift-tts/Sources/KokoroTTS/Resources/KokoroRuntime/hnsf_weights.json` | Create | Checked verified hn-NSF weights resource. |
| `swift-tts/Sources/KokoroTTS/KokoroTextProcessor.swift` | Create | Native text prep facade. |
| `swift-tts/Sources/KokoroTTS/KokoroPhonemizer.swift` | Create | Protocol boundary for MisakiSwift now and future backends later. |
| `swift-tts/Sources/KokoroTTS/KokoroMisakiPhonemizer.swift` | Create | Gist-style MisakiSwift phonemizer wrapper. |
| `swift-tts/Sources/KokoroTTS/VoiceTable.swift` | Create | Gist-style voice `.bin` loader and row selection. |
| `swift-tts/Sources/KokoroTTS/TextChunker.swift` | Create | Port Botnet fleet chunker with only the documented configurable 15-second SDK cap change. |
| `swift-tts/Sources/KokoroTTS/KokoroTTS.swift` | Create | Drop-in public API. |
| `swift-tts/Sources/KokoroTTS/KokoroResourceProvider.swift` | Create | Bundle/directory/downloaded resource loading. |
| `swift-tts/Sources/KokoroTTS/KokoroDownloadedModelStore.swift` | Create | Gist-style manifest download, compile, cache, and invalidation. |
| `swift-tts/Sources/KokoroTTS/KokoroSDKModelProvider.swift` | Create | Lazy `.mlpackage` provider with source-hash-matched `.mlmodelc` cache reuse implementing `KokoroModelProvider`. |
| `swift-tts/Sources/KokoroTTS/KokoroErrors.swift` | Create | Typed SDK errors. |
| `swift/Tests/KokoroPipelineTests/*` | Modify/Create | Prepared-input compatibility stays green. |
| `swift-tts/Tests/KokoroTTSTests/*` | Create | Text prep, resource, manifest, SDK API, and diagnostics tests. |
| `Examples/KokoroDemoApp/**` | Create | Drop-in demo app. |
| `README.md` | Modify | SDK quickstart. |
| `README/hf-model-card.md` | Modify | Usage snippet must match actual API. |
| `README/Notes/kokoro-drop-in-sdk-v1.md` | Create | Implementation decisions and evidence. |
| `hnsf_weights.json` | Modify | Replace unverified root copy or point tooling to the verified SDK resource. |
| `_kokoro_vocab.json` | Read/Maybe modify | Source for checked SDK vocab resource if upstream config is not copied. |

## Risks and Mitigations

- **Misaki-vs-eSpeak drift:** Audio quality or token IDs may differ from Botnet.
  -> Produce a reviewed drift table in Phase 1 and audio/tokenization checks
  before public API. Exact phoneme equality is not the V1 requirement.
- **MisakiSwift packaging risk:** Resource layout, dynamic-library embedding, or
  platform floor may not fit the SDK distribution target.
  -> Reuse Gist's packaging fix where appropriate and treat license/provenance,
  platform floor, code signing, and iOS simulator build as release gates.
- **Resource bundle bloat:** App integration becomes impractical.
  -> Provide starter/custom bundle profiles and explicit asset sizing.
- **Unreproducible bundle:** Local symlinks, ignored outputs, or modified
  `.mlpackage` files leak into release artifacts.
  -> Reject symlinks/path escapes and require pinned HF revisions plus
  per-package tree digests.
- **Main-thread stalls:** First use blocks UI.
  -> Async load/prewarm API and tests that assert no main-actor-only compile path.
- **Hidden dependency creep:** SDK silently needs Node/Python/JavaScriptCore.
  -> CI/smoke fixture must run iOS/macOS path without those dependencies in app
  code.
- **Docs overpromise:** Model card drifts from actual API.
  -> Compile the documentation snippet in a consumer fixture.
- **Compute-unit regressions:** Convenience API masks stage policy.
  -> Keep existing compute policy defaults and benchmark controls available.
- **Skipped iPhone proof:** Simulator smoke passes but real-device first load or
  phonemizer resource packaging fails.
  -> Physical-device raw-text smoke is a release blocker for iOS readiness.

## Progress Tracker

### Phase 0: Make the Copied Prep Layer Self-Hosting

- [x] Fix JS default runtime root.
- [x] Add `--runtime-root`.
- [x] Add `--botnet-root`.
- [x] Add dependency install/run path.
- [x] Add full-contract Botnet parity script and fixtures.

### Phase 1: SDK Boundaries and Gist-Proven Phonemizer Spike

- [x] Create `swift-tts/Package.swift` and `KokoroTTS` product.
- [x] Keep `swift/Package.swift` free of MisakiSwift and raw-text targets.
- [x] Add MisakiSwift only to the `swift-tts/` package.
- [x] Prove MisakiSwift packaging/offline behavior on macOS and compile
      validation for generic iOS Simulator.
- [x] Record Botnet/eSpeak drift table.
- [x] Record license/provenance/App Store decision.
- [x] Define privacy-safe diagnostics policy.

### Phase 2: Runtime Asset Source of Truth

- [x] Add checked SDK vocab resource.
- [x] Add checked verified hn-NSF weights resource.
- [x] Add runtime asset verifier.
- [x] Reject symlinked configs and unverified hashes.

### Phase 3: Native Swift Text Prep and Public Types

- [x] Add public prepared-input and options types.
- [x] Adapt Gist `KokoroG2P`, `VoiceTable`, and `TextChunker` behavior.
- [x] Port deterministic non-phonemizer prep logic.
- [x] Prove Swift prep parity against Gist and approved drift against JS oracle.

### Phase 4: Reproducible HF Downloads and SDK Bundle Builder

- [x] Add repo/revision-pinned HF downloader mode.
- [x] Add HF artifact inspection/provenance script.
- [x] Add per-`.mlpackage` tree hashing.
- [x] Add starter/custom/full bundle generator.
- [x] Add hosted-manifest generator.
- [x] Add runtime manifest validation.

### Phase 5: SDK Model Provider and Drop-In API

- [x] Add resource provider.
- [x] Add downloaded-resource provider.
- [x] Add SDK model provider using `KokoroModelProvider`.
- [x] Support `.mlpackage` plus source-hash-matched `.mlmodelc` cache reuse.
- [x] Add `KokoroTTS` facade.
- [x] Add chunk synthesis and PCM conveniences.
- [x] Add Gist-style staged compute policy and `.cpuOnly` retry.

### Phase 6: Examples and macOS/iOS Validation

- [x] Add fresh consumer fixture.
- [x] Add example app and smoke executable.
- [x] Run macOS smoke.
- [x] Build iOS simulator app.
- [x] Run mandatory physical-device smoke or mark iOS release blocked. Current
      status: iPhone 15 Pro Max proved synthesis, cancellation, long-text
      chunking, background/foreground, and the approved bounded
      memory-pressure recovery harness.

### Phase 7: Release Artifact and Documentation Pass

- [x] Add SDK docs.
- [x] Add implementation notes.
- [x] Add release checklist.
- [x] Add drift check.
- [x] Update README and model card.
- [x] Decide measured distribution format.

## Debug Notes

Append real issues encountered during implementation with fixes.

### 2026-06-28 - Planning Baseline

**Problem:** `scripts/kokoro-prepare-input.mjs` is present in this repo but
fails by default because it still assumes Botnet's runtime package layout.
**Root Cause:** The script was copied from Botnet without changing
`DefaultRuntimeRoot`.
**Fix:** Planned Phase 0 fix: self-host the current repo root and add an
explicit `--runtime-root`.
**Files:** `scripts/kokoro-prepare-input.mjs`

---

### 2026-06-28 - Cross-Agent Plan Audit Fixes

**Problem:** Cross-agent audit graded the draft C overall because native
phonemizer ownership, resource provenance, iOS physical-device proof, and SDK
target boundaries were under-specified.
**Root Cause:** The first draft mixed public API work with backend discovery,
used generated/ignored runtime assets in clean-checkout claims, allowed iOS
device proof to be deferred, and risked growing `KokoroPipeline` into a
high-level SDK god object.
**Fix:** Split the plan into stricter phases: prove raw-text phonemizer
packaging before public API, add a separate `swift-tts/` package/provider
boundary, promote checked runtime assets, require pinned HF revisions and
per-package digests, make physical-device iPhone smoke a release gate, and add
drift/provenance checks.
**Files:** `README/Plans/kokoro-drop-in-sdk-v1.md`

---

### 2026-06-28 - Gist iOS Pattern Update

**Problem:** The post-audit plan still treated native eSpeak parity as the V1
app-path gate, even though the shipping Gist iOS app already uses pure Swift
MisakiSwift G2P plus checked Kokoro vocab, Core ML runtime, and manifest-backed
model download/cache.
**Root Cause:** The plan over-weighted Botnet's controlled macOS Node bridge as
the runtime shape instead of separating Botnet as an oracle from Gist as the
iOS product precedent.
**Fix:** Reoriented V1 around the Gist app pattern: MisakiSwift in `KokoroTTS`,
Botnet JS/eSpeak as drift oracle only, Gist-style voice table/chunking/model
store, hosted-manifest downloads, staged compute defaults, and explicit
platform-floor documentation.
**Files:** `README/Plans/kokoro-drop-in-sdk-v1.md`

---

### 2026-06-28 - `swift-tts/` Safer Default

**Problem:** The plan still left room to start with same-package `KokoroTTS`
and only split later if SwiftPM platform floors leaked.
**Root Cause:** That made an avoidable packaging question part of execution
risk, even though the desired boundary is already clear.
**Fix:** Make `swift-tts/` the default from Phase 1. `swift/` remains the
prepared-input package; `swift-tts/` depends on `../swift`, owns MisakiSwift,
resources, hosted-manifest download/cache, public raw-text APIs, and its own
platform floor.
**Files:** `README/Plans/kokoro-drop-in-sdk-v1.md`

---

### 2026-06-28 - Plan Skill Routing

**Problem:** The implementation phases named tasks and verification gates but
did not tell the executor which repo skills to invoke for each phase.
**Root Cause:** The plan followed the template structure but omitted the
workflow routing needed by `execute-plan`, `phase-audit`, domain skills, notes,
and release/documentation skills.
**Fix:** Added a required skills section under `Implementation Phases` and a
`Required skills` line to every phase, with side-effect boundaries for
`execute-plan-hardcore`, `deploy`, `git-commit`, and `git-push`.
**Files:** `README/Plans/kokoro-drop-in-sdk-v1.md`

---

### 2026-06-28 - Phase 0 Prep Self-Hosting

**Problem:** The copied JS prep script still defaulted to Botnet's
`packages/kokoro-coreml-runtime` layout, so this repo could not prepare text
from its own checkout without `KOKORO_COREML_ROOT`.
**Root Cause:** The script looked for Botnet's generated runtime tree before
checking this repo's `kokoro.js/src`, `_kokoro_vocab.json`, and
`kokoro.js/voices` assets.
**Fix:** Added explicit `--runtime-root` handling to both prep scripts, made the
JS script default to the current checkout when runtime assets are present,
added local vocab/voice fallbacks, added a Botnet full-contract comparison
harness, and added tokenizer edge-case fixtures.
**Verification:** JS `Hello world` smoke passed; Botnet comparison passed 12
fixtures; `uv run` Python smoke generated a 32-token padded input; `npm
--prefix kokoro.js test` passed 276 tests.
**Files:** `scripts/kokoro-prepare-input.mjs`,
`scripts/kokoro-prepare-input.py`, `scripts/compare_botnet_prepare_input.mjs`,
`tests/fixtures/kokoro-text-prep/*.json`,
`README/Notes/kokoro-drop-in-sdk-v1.md`

---

## Critical Reminder

> SIMPLER IS BETTER. If you are adding complexity, justify it. Most of the
> time, the simplest solution wins.
