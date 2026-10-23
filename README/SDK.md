# KokoroTTS Swift SDK

Drop local Kokoro speech into an iPhone or Mac app with no server, Node or
Python at runtime. The SDK owns raw-text preparation, Misaki phonemization,
sentence chunking, Core ML model loading, and 24 kHz mono PCM output.

## Contract

| Item | Value |
| --- | --- |
| Swift product | `KokoroTTS` |
| Platforms | `iOS 18.0+`, `macOS 15.0+`; build with Xcode 26+ (Swift 6.2) |
| Output | 24 kHz mono Float PCM |
| Sample rate | `24000` |
| Starter voice | `af_heart` |
| Starter bucket | `15` seconds |
| Full buckets | `3,7,10,15,30` seconds |
| Duration token size | `128` (fixed padded shape; larger text is split) |
| Max caller chunk tokens | `450` |
| Voice embedding dimension | `256` |
| Default chunk cap | `15.0` seconds on macOS, `10.0` on iOS |

The platform floor is intentionally high. A reliable iOS 18/macOS 15 SDK is
better than a broader SDK that fails under real app conditions.

## Platform Scope

Apple Silicon is the supported Mac target: the fast path needs the Neural
Engine. Intel Macs have no ANE and are untested with the current pipeline.

The SDK supports iOS 18.0+ for iPhone/iPad app builds. It does not support
watchOS in V1. The raw-text SDK depends on MisakiSwift/MLX, and the current MLX
dependency does not build for watchOS (`IOSurface/IOSurfaceRef.h` is unavailable
in the watchOS SDK). Do not add a watchOS release claim until the phonemizer
dependency is isolated or replaced and a watchOS build plus device smoke pass.

## Install

Depend on the repo from GitHub. Pin a commit; the root `Package.swift` exposes
`KokoroTTS`, plus the low-level `KokoroPipeline` and `KokoroG2P`:

```swift
dependencies: [
    .package(url: "https://github.com/mattmireles/kokoro-coreml", revision: "<commit>")
]
// target:
.product(name: "KokoroTTS", package: "kokoro-coreml")
```

A local checkout works too: `.package(path: "../kokoro-coreml")`.

Try it without writing code: `swift run -c release kokoro-say "Hello"` from the
repo root writes `./kokoro.wav` (see the README quickstart).

## Hosted Models

`KokoroResourceProvider.huggingFace()` downloads the starter bundle (one 15 s
bucket, voice `af_heart`, about 210 MB) from an immutable Hugging Face revision
and checks the hosted manifest's SHA-256 before trusting any file in it. The
revision and digest live in one place,
[`KokoroHostedModels.swift`](../swift-tts/Sources/KokoroTTS/KokoroHostedModels.swift);
a release updates both together. The bundle is the pipeline the paper measured:
split duration and F0Ntrain, ANE-layout DecoderPre, and the hybrid ANE+GPU
generator with the folded all-ANE generator below 7 s and as the fallback.

Bundles use manifest `schema_version` 2. SDK builds from before October 23,
2026 reject them with `unsupportedManifestSchema`, and this SDK rejects older
bundles the same way; pin a matching SDK commit and HF revision.

## Build A Resource Bundle

Download the model snapshot and build a starter bundle:

```bash
uv run python scripts/download_models.py \
  --repo-id mattmireles/kokoro-coreml \
  --revision <hf-artifact-revision> \
  --sdk-profile starter \
  --manifest-out /tmp/kokoro-download-manifest.json

node scripts/build_sdk_bundle.mjs \
  --profile starter \
  --compile-models 1 \
  --output /tmp/kokoro-sdk-starter \
  --repo-id mattmireles/kokoro-coreml \
  --revision <hf-artifact-revision> \
  --download-manifest /tmp/kokoro-download-manifest.json

node scripts/validate_sdk_bundle.mjs /tmp/kokoro-sdk-starter
```

`<hf-artifact-revision>` is the HF commit holding the model artifacts
(`hf_artifact_revision` in the HF repo's `sdk/SDKReleaseManifest.json`). It is where bundle
sources are verified, not a pin for hosted manifests: those land in a later
commit, and consumers pin that one (the commit printed by
`prepare_hf_sdk_metadata.py --upload`).

Profiles are defined in one file, `scripts/sdk_profiles.json`. Use
`--profile full` for every bucket (`3,7,10,15,30`), every English voice, both
folded variants, every hybrid row count and the `KokoroG2P` assets under
`g2p/`. Both the `KokoroTTS` SDK and the low-level `KokoroPipeline` load a full bundle. On HF it is published as
`sdk/full/KokoroRuntimeManifest.json`; build it locally with the commands
above. V1 raw-text synthesis rejects non-English voice prefixes even if a custom
bundle includes their embeddings. For an app-specific bundle, download with
`--sdk-profile custom --sdk-voices af_heart,af_bella --sdk-buckets 15,30` and
build with `--profile custom --voices af_heart,af_bella --buckets 15,30`.

`HostedManifest.json` is for downloaded-resource mode. It intentionally excludes
`compiled/`; each app keeps its own writable compiled-model cache.

## Use Bundled Resources

Bundle the generated resource directory with your app, then load it explicitly:

```swift
import KokoroTTS

// Pass a writable compiledModelsDirectory for app/package bundles so Core ML
// compilation does not attempt to write inside a read-only resource tree.
let resources = KokoroResourceProvider.directory(
    bundleURL,
    compiledModelsDirectory: cacheURL
)
let tts = try await KokoroTTS.load(resources: resources)
try await tts.prewarm(text: "Hello world.", voice: .afHeart)
let audio = try await tts.synthesize("Hello world.", voice: .afHeart)
let buffer = try audio.makePCMBuffer()
```

`KokoroTTS.load` validates manifests, hashes, vocab, and hn-NSF weights without
compiling Core ML models or initializing Misaki/MLX on the caller's main actor.
Bundles must declare verified Hugging Face provenance
(`hf_provenance_verified: true` and `hf_download_manifest_sha256`). Call
`prewarm(...)` from app startup or another background task to compile,
specialize, and prove the acoustic bucket selected by representative text
before the first user-visible synthesis. Full-profile clients that use
materially different utterance lengths should prewarm representative text for
each expected path.

`KokoroAudio.samples` is mono Float PCM. `KokoroAudio.sampleRate` is `24000`.
Use `makePCMBuffer()` when AVFoundation is available.

## Use Downloaded Resources

For apps that stage model assets after install, hydrate a hosted manifest into a
writable cache. The SDK requires the expected hosted-manifest SHA-256 before it
trusts any file list or per-file hashes. Production manifests must be served
over HTTPS; local HTTP is only for explicit development fixtures.

```swift
import KokoroTTS

// The pinned starter bundle (KokoroHostedModels):
let resources = try await KokoroResourceProvider.huggingFace(cacheDirectory: cacheURL)

// Or your own hosted bundle:
// let resources = try await KokoroDownloadedModelStore(
//     manifestURL: URL(string: "https://example.com/<revision>/HostedManifest.json")!,
//     expectedManifestSHA256: "<sha256 of that exact file>",
//     cacheDirectory: cacheURL
// ).hydrate()

let tts = try await KokoroTTS.load(resources: resources)
let audio = try await tts.synthesize(articleText, voice: .afHeart)
```

The downloader verifies the manifest hash before reading file entries, enforces
file-count and byte-count limits, verifies each file hash, rejects path escapes
and symlinked cache roots, and drops stale compiled-model cache entries when the
hosted manifest version changes.

## Playback

For a simple iOS playback path:

```swift
import AVFoundation
import KokoroTTS

let session = AVAudioSession.sharedInstance()
try session.setCategory(.playback, mode: .spokenAudio)
try session.setActive(true)

let engine = AVAudioEngine()
let player = AVAudioPlayerNode()
let buffer = try audio.makePCMBuffer()
engine.attach(player)
engine.connect(player, to: engine.mainMixerNode, format: buffer.format)
try engine.start()
player.scheduleBuffer(buffer)
player.play()
```

## Example App

Generate and build the iOS demo:

```bash
cd examples/KokoroDemoApp
xcodegen generate --spec project.yml
xcodebuild -quiet \
  -scheme KokoroDemoApp \
  -destination 'generic/platform=iOS Simulator' \
  -derivedDataPath /tmp/kokoro-demo-ios-sim-dd \
  build
```

For a physical iPhone, pass your own signing values:

```bash
xcodebuild -quiet \
  -scheme KokoroDemoApp \
  -destination 'id=<physical-device-udid>' \
  -derivedDataPath /tmp/kokoro-demo-device-dd \
  KOKORO_DEMO_DEVELOPMENT_TEAM=<apple-development-team-id> \
  KOKORO_DEMO_BUNDLE_ID=<unique-demo-bundle-id> \
  build
```

The demo uses `NSAllowsLocalNetworking` for local manifest testing. A fresh
install may still require the user to accept the iOS local-network permission
prompt.

The demo reads its resource mode from launch arguments (set them in the Xcode
scheme, or pass them to `xcrun devicectl device process launch`):

- Downloaded: `--resource-mode downloaded --manifest-url <https URL> --manifest-sha256 <sha256>`.
  For a manifest served from your Mac over plain HTTP, also set the environment
  variable `KOKORO_ALLOW_INSECURE_LOCAL_MANIFEST=1`.
- Bundled: `--resource-mode bundled --bundle-subdirectory KokoroRuntime`.

Use bundled mode only after adding a generated runtime directory with
`KokoroRuntimeManifest.json` to the app target.

## Troubleshooting

`missingManifest`: The resource root must contain `KokoroRuntimeManifest.json`.

`missingModel`: The bundle profile does not include the bucket needed for this
input, or the model package path is wrong.

`unsupportedVoice`: Add `voices/<voice>.bin` to the bundle or choose a bundled
voice.

`inputTooLong`: Reduce `maxChunkSeconds` or pass `maxCharacters`.

`unpinnedHostedRevision`: This build has no pinned Hugging Face revision
(`KokoroHostedModels`); use a release commit of `main`, or pass a bundle.

`missingVoice`: The manifest lists a voice whose file is missing from the
bundle; rebuild or re-download it.

`unsupportedManifestSchema`: The bundle and the SDK disagree on manifest
schema. Bundles from October 23, 2026 on are schema 2; pin a matching SDK
commit.

`kokoro: hybrid generator failed (...); using the folded generator until
relaunch` on stderr: the ANE+GPU generator could not run (for example, no
Metal access while an iOS app is in the background). Synthesis continues on the
all-ANE folded generator; set `KOKORO_HYBRID_GENERATOR=0` to use it always.

`KOKORO_SPLIT_DURATION=0`, `KOKORO_SPLIT_F0N=0` or `KOKORO_FOLDED_GENERATOR=0`
make a schema-2 bundle fail to load: they switch off packages it relies on and
it carries no fallback. They are for benchmarks with local model directories.

Build fails compiling MLX Metal shaders on a fresh Xcode: run
`xcodebuild -downloadComponent MetalToolchain` once.

`synthesisCancelled`: The caller cancelled the task. This is expected for app
navigation, queue replacement, and interruption handling.

Local-network download fails on iOS: confirm `NSLocalNetworkUsageDescription`,
use `NSAllowsLocalNetworking` only for development manifests, and accept the
iOS permission prompt on the device.

## Release Checklist

Run these before publishing a new SDK snapshot:

```bash
uv run --with pytest python -m pytest -q
swift test --package-path swift
swift test --package-path swift-tts
uv run python scripts/verify_runtime_assets.py
node scripts/check_sdk_drift.mjs
# Optional, with a sibling botnet checkout:
node scripts/compare_botnet_prepare_input.mjs --botnet-root ../botnet \
  --fixtures tests/fixtures/kokoro-text-prep/*.json --compare full
```

A release is three steps, run back to back (between steps 1 and 3, `main`'s
old hosted manifest names replaced files, so unpinned readers fail closed).
Bundle path = HF path = `coreml/<name>`. Push the release commit to a
`release/<date>` branch, not `main`: `main` gets it, fast-forwarded, together
with the pin commit at the end, so `main` never carries an unpinned
`KokoroHostedModels`.

```bash
# 1. Artifacts (one HF commit). Run from the Mac that has the gitignored
#    coreml-*/ export dirs; writes nothing locally. --replace swaps same-name
#    packages (here the ANE-layout DecoderPre) in the same commit.
ARGS=(coreml-hybrid=coreml/hybrid)
for p in coreml-dur-ane/kokoro_duration_{ane_front,back}_t128.mlpackage \
         coreml-f0n-split/*.mlpackage coreml-fold/*.mlpackage coreml-dpre-ane/*.mlpackage; do
  ARGS+=("$p=coreml/${p#*/}")
done
for b in 3 7 10 15 30; do ARGS+=(--replace "coreml/kokoro_decoder_pre_${b}s.mlpackage"); done
uv run python scripts/upload_hf_artifacts.py --dry-run "${ARGS[@]}"
uv run python scripts/upload_hf_artifacts.py --message "<what changed>" "${ARGS[@]}"
REV=<printed "artifact commit" SHA, or the "already at" SHA if nothing changed>

# 2. Bundles, from a fresh clone of the pushed release branch (clean tree,
#    empty coreml/), then smoke-test the starter bundle.
git clone -b release/<date> https://github.com/mattmireles/kokoro-coreml /tmp/kokoro-release && cd /tmp/kokoro-release
for profile in starter full; do
  uv run python scripts/download_models.py --repo-id mattmireles/kokoro-coreml --revision "$REV" \
    --sdk-profile "$profile" --manifest-out "/tmp/kokoro-download-manifest-$profile.json"
  node scripts/build_sdk_bundle.mjs --profile "$profile" --compile-models 1 \
    --output "/tmp/kokoro-sdk-$profile" --repo-id mattmireles/kokoro-coreml --revision "$REV" \
    --download-manifest "/tmp/kokoro-download-manifest-$profile.json"
  node scripts/validate_sdk_bundle.mjs "/tmp/kokoro-sdk-$profile"
done
swift run -c release kokoro-say --bundle /tmp/kokoro-sdk-starter --runs 2 \
  "Two short sentences are enough. The starter bundle's one bucket is 15 seconds, so the hybrid runs."
#    Expect generator=hybrid, duration=split, f0=split and rms above 0.01.
xcodebuild -quiet -scheme KokoroTTS -destination 'generic/platform=iOS' \
  -derivedDataPath /tmp/kokoro-tts-ios-dd build

# 3. Metadata (one HF commit; prints the SHA to pin). Refuses a dirty sdk_commit.
uv run python scripts/prepare_hf_sdk_metadata.py --repo-id mattmireles/kokoro-coreml \
  --starter-bundle /tmp/kokoro-sdk-starter --full-bundle /tmp/kokoro-sdk-full \
  --output /tmp/kokoro-hf-sdk-metadata --upload
PIN=<printed commit SHA>
shasum -a 256 /tmp/kokoro-hf-sdk-metadata/HostedManifest.json
uv run python scripts/inspect_hf_artifacts.py --repo-id mattmireles/kokoro-coreml --revision "$PIN" \
  --verify-hosted-digests --output /tmp/kokoro-hf-artifacts.json
```

Then set `revision` and `manifestSHA256` in
`swift-tts/Sources/KokoroTTS/KokoroHostedModels.swift` to `$PIN` and that
digest, commit on the release branch, push it, fast-forward `main` to it, and
confirm `swift run -c release kokoro-say "Hello"` works from a fresh clone of
`main` with an empty `~/Library/Caches/kokoro-coreml`.

The SDK never compiles the private e5rt path: it is behind `#if KOKORO_E5`
(benchmarks only: `swift build -Xswiftc -DKOKORO_E5`). Check a release binary
with `strings .build/release/kokoro-say | grep -c -E 'e5rt_|Espresso'`; it must
print 0.

Before claiming iOS readiness, run the physical-device demo smoke and record the
device evidence in [kokoro-drop-in-sdk-v1 notes](Notes/kokoro-drop-in-sdk-v1.md).
Do not claim watchOS readiness unless the SDK builds for watchOS and the
MisakiSwift/MLX dependency boundary has been replaced or made watch-safe.

## What Is Still Low-Level

`KokoroPipeline` remains available for benchmark and graph-level work. New app
integrations should start with `KokoroTTS`; it is the supported raw-text SDK
surface.
