# KokoroTTS Swift SDK

Drop local Kokoro speech into an iPhone or Mac app without running Botnet,
Node, or Python at runtime. The SDK owns raw-text preparation, Misaki
phonemization, Botnet-compatible chunking, Core ML model loading, and 24 kHz
mono PCM output.

## Contract

| Item | Value |
| --- | --- |
| Swift product | `KokoroTTS` |
| Platforms | `iOS 18.0+`, `macOS 15.0+` |
| Output | 24 kHz mono Float PCM |
| Sample rate | `24000` |
| Starter voice | `af_heart` |
| Starter bucket | `15` seconds |
| Full buckets | `3,7,10,15,30` seconds |
| Duration token sizes | `32,64,128,256,320,384,512` |
| Max caller chunk tokens | `450` |
| Voice embedding dimension | `256` |
| Default chunk cap | `15.0` seconds |

The platform floor is intentionally high. A reliable iOS 18/macOS 15 SDK is
better than a broader SDK that fails under real app conditions.

## Install

Use the package under `swift-tts` from a local checkout:

```swift
dependencies: [
    .package(path: "../kokoro-coreml/swift-tts")
]
```

Then add the library product:

```swift
.product(name: "KokoroTTS", package: "swift-tts")
```

SwiftPM uses the directory name `swift-tts` as the package identity for this
local path dependency. For V1 this repo publishes source plus downloadable
resource bundles. It does
not publish a SwiftPM binary/resource artifact yet; choose that only after
package size and Xcode resource behavior are measured. If you need a remote
SwiftPM dependency before that, mirror this repo layout so `swift-tts` can still
resolve its sibling `../swift` package, or publish both packages explicitly.
Mirroring only `swift-tts` is not enough.

The repo also includes `examples/KokoroConsumerFixture` and
`examples/KokoroDemoApp` as integration fixtures.

The current public starter manifest is:

```text
https://huggingface.co/mattmireles/kokoro-coreml/resolve/main/HostedManifest.json
```

## Build A Resource Bundle

Download the model snapshot and build a starter bundle:

```bash
python3 scripts/download_models.py \
  --repo-id mattmireles/kokoro-coreml \
  --revision <hf-revision> \
  --sdk-profile starter \
  --manifest-out /tmp/kokoro-download-manifest.json

node scripts/build_sdk_bundle.mjs \
  --profile starter \
  --compile-models 1 \
  --output /tmp/kokoro-sdk-starter \
  --repo-id mattmireles/kokoro-coreml \
  --revision <hf-revision> \
  --download-manifest /tmp/kokoro-download-manifest.json

node scripts/validate_sdk_bundle.mjs /tmp/kokoro-sdk-starter
```

Use the latest Hugging Face revision for normal development, or pin the
revision recorded in `sdk/SDKReleaseManifest.json` when reproducing a release.

Use `--profile full` for every checked bucket and every supported English voice
file. V1 raw-text synthesis rejects non-English voice prefixes even if a custom
bundle includes their embeddings. Use
`--profile custom --voices af_heart,af_bella --buckets 15,30` for an
app-specific bundle.

`HostedManifest.json` is for downloaded-resource mode. It intentionally excludes
`compiled/`; each app keeps its own writable compiled-model cache.

## Use Bundled Resources

Bundle the generated resource directory with your app, then load it explicitly:

```swift
import KokoroTTS

let resources = KokoroResourceProvider.directory(bundleURL)
let tts = try await KokoroTTS.load(resources: resources)
try await tts.prewarm(text: "Hello world.", voice: .afHeart)
let audio = try await tts.synthesize("Hello world.", voice: .afHeart)
let buffer = try audio.makePCMBuffer()
```

`KokoroTTS.load` validates manifests, hashes, vocab, and hn-NSF weights without
compiling Core ML models or initializing Misaki/MLX on the caller's main actor.
Call `prewarm(...)` from app startup or another background task to compile and
cache the selected models before the first user-visible synthesis.

`KokoroAudio.samples` is mono Float PCM. `KokoroAudio.sampleRate` is `24000`.
Use `makePCMBuffer()` when AVFoundation is available.

## Use Downloaded Resources

For apps that stage model assets after install, hydrate a hosted manifest into a
writable cache. In production, serve this manifest over HTTPS and pin either the
HF revision or the `sdk/SDKReleaseManifest.json` checksum you expect; per-file
hashes protect against transfer corruption, not a malicious replacement
manifest.

```swift
import KokoroTTS

let resources = try await KokoroDownloadedModelStore(
    manifestURL: URL(string: "https://huggingface.co/mattmireles/kokoro-coreml/resolve/main/HostedManifest.json")!,
    cacheDirectory: cacheURL
).hydrate()

let tts = try await KokoroTTS.load(resources: resources)
let audio = try await tts.synthesize(articleText, voice: .afHeart)
```

The downloader verifies byte counts and SHA-256 hashes, rejects path escapes and
symlinked cache roots, and drops stale compiled-model cache entries when the
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

The demo supports both resource paths:

```bash
--resource-mode downloaded --manifest-url http://<mac-ip>:8766/HostedManifest.json
--resource-mode bundled --bundle-subdirectory KokoroRuntime
```

Use bundled mode only after adding a generated runtime directory with
`KokoroRuntimeManifest.json` to the app target.

## Release Checklist

Run these before publishing a new SDK snapshot:

```bash
swift test --package-path swift
swift test --package-path swift-tts
node scripts/compare_botnet_prepare_input.mjs \
  --botnet-root /Users/mm/Documents/GitHub/botnet \
  --fixtures tests/fixtures/kokoro-text-prep/*.json \
  --compare full
python3 scripts/verify_runtime_assets.py
node scripts/check_sdk_drift.mjs
```

Then generate and validate at least the starter bundle:

```bash
python3 scripts/download_models.py \
  --repo-id mattmireles/kokoro-coreml \
  --revision <hf-revision> \
  --sdk-profile starter \
  --manifest-out /tmp/kokoro-download-manifest.json

node scripts/build_sdk_bundle.mjs \
  --profile starter \
  --compile-models 1 \
  --output /tmp/kokoro-sdk-starter \
  --repo-id mattmireles/kokoro-coreml \
  --revision <hf-revision> \
  --download-manifest /tmp/kokoro-download-manifest.json

node scripts/validate_sdk_bundle.mjs /tmp/kokoro-sdk-starter
pushd swift-tts
xcodebuild -quiet \
  -scheme kokoro-sdk-smoke \
  -destination 'platform=macOS,arch=arm64' \
  -derivedDataPath /tmp/kokoro-tts-smoke-dd \
  build
popd

DYLD_FRAMEWORK_PATH=/tmp/kokoro-tts-smoke-dd/Build/Products/Debug/PackageFrameworks \
  /tmp/kokoro-tts-smoke-dd/Build/Products/Debug/kokoro-sdk-smoke \
  /tmp/kokoro-sdk-starter \
  "Hello world."
```

Generate the full bundle before publishing a release that claims full bucket or
multi-voice coverage:

```bash
python3 scripts/download_models.py \
  --repo-id mattmireles/kokoro-coreml \
  --revision <hf-revision> \
  --sdk-profile full \
  --manifest-out /tmp/kokoro-download-manifest-full.json

node scripts/build_sdk_bundle.mjs \
  --profile full \
  --compile-models 1 \
  --output /tmp/kokoro-sdk-full \
  --repo-id mattmireles/kokoro-coreml \
  --revision <hf-revision> \
  --download-manifest /tmp/kokoro-download-manifest-full.json

node scripts/validate_sdk_bundle.mjs /tmp/kokoro-sdk-full
```

Then publish and inspect the lightweight Hugging Face SDK metadata payload:

```bash
python3 scripts/prepare_hf_sdk_metadata.py \
  --repo-id mattmireles/kokoro-coreml \
  --starter-bundle /tmp/kokoro-sdk-starter \
  --full-bundle /tmp/kokoro-sdk-full \
  --output /tmp/kokoro-hf-sdk-metadata \
  --upload

python3 scripts/inspect_hf_artifacts.py \
  --repo-id mattmireles/kokoro-coreml \
  --verify-hosted-digests \
  --output /tmp/kokoro-hf-artifacts.json
```

Before claiming iOS readiness, run the physical-device demo smoke and record the
device evidence in [kokoro-drop-in-sdk-v1 notes](Notes/kokoro-drop-in-sdk-v1.md).

## Troubleshooting

`missingManifest`: The resource root must contain `KokoroRuntimeManifest.json`.

`missingModel`: The bundle profile does not include the bucket needed for this
input, or the model package path is wrong.

`unsupportedVoice`: Add `voices/<voice>.bin` to the bundle or choose a bundled
voice.

`inputTooLong`: Reduce `maxChunkSeconds`, pass `maxCharacters`, or use a larger
bundle profile.

`synthesisCancelled`: The caller cancelled the task. This is expected for app
navigation, queue replacement, and interruption handling.

Local-network download fails on iOS: confirm `NSLocalNetworkUsageDescription`,
use `NSAllowsLocalNetworking` only for development manifests, and accept the
iOS permission prompt on the device.

## What Is Still Low-Level

`KokoroPipeline` remains available for benchmark and graph-level work. New app
integrations should start with `KokoroTTS`; it is the supported raw-text SDK
surface.
