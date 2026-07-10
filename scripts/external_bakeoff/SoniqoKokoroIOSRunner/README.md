# Soniqo Kokoro iOS Runner

Minimal Kokoro-only iOS runner for the connected iPhone 12 Pro. This avoids
Soniqo's full `iOSEchoDemo`, which pulls in ASR, VAD, MLX, and speech-core
dependencies that are not part of the Core ML TTS timing boundary.

## Generate

Set `SPEECH_SWIFT_PATH` to a pinned `soniqo/speech-swift` clone and generate the
runtime manifest source plus the project:

```bash
python scripts/external_bakeoff/generate_ios_runner_manifest.py
cd scripts/external_bakeoff/SoniqoKokoroIOSRunner
SPEECH_SWIFT_PATH=/tmp/kokoro-external-bakeoff/speech-swift xcodegen generate
```

## Build For The Connected iPhone

Run the repo-level preflight before starting a long physical-device build:

```bash
SPEECH_SWIFT_PATH=/tmp/kokoro-external-bakeoff/speech-swift \
  python scripts/external_bakeoff/preflight_ios_runner.py \
  --generate-project \
  --output outputs/external_bakeoff/ios_runner_preflight_latest.json
```

Preflight requirements:

- The iPhone must appear in `xcrun devicectl list devices`.
- `SPEECH_SWIFT_PATH` must point to a pinned `soniqo/speech-swift` clone.
- `DEVELOPMENT_TEAM` must be set to an Apple development team ID.
- `security find-identity -v -p codesigning` must show a valid Apple
  development signing identity.

Once preflight is green, generate the project and build for the connected
iPhone:

```bash
SPEECH_SWIFT_PATH=/tmp/kokoro-external-bakeoff/speech-swift \
  python scripts/external_bakeoff/preflight_ios_runner.py \
  --generate-project \
  --build \
  --output outputs/external_bakeoff/ios_runner_build_latest.json
```

The app loads `KokoroTTSModel.fromPretrained(computeUnits: .all)`, synthesizes
the five runtime bucket inputs from
`outputs/external_bakeoff/runtime_input_manifest.json`, and renders JSON with
one cold call plus five warm calls per bucket. It remains Kokoro TTS only:
Whisper, ASR, VAD, playback, and the full Soniqo echo demo are outside the
measurement path.

When the run finishes, the app copies the rendered JSON to the iPhone
clipboard. Paste that JSON into
`outputs/external_bakeoff/ios_runner_payload_latest.json`, then ingest it into
the common result schema:

```bash
python scripts/external_bakeoff/ingest_ios_runner_result.py \
  --input outputs/external_bakeoff/ios_runner_payload_latest.json \
  --machine-id iphone-12-pro
```

The ingested result records do not include a spot-check WAV or output SHA-256
because the minimal on-device runner currently exposes timing JSON only.
