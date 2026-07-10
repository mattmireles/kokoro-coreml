# External Bakeoff Phase 0 API Audit

**Date:** 2026-06-05
**Plan:** `README/Plans/kokoro-external-bakeoff-v1.md`
**Status:** Phase 0 complete for Mac synthesis proof; physical iPhone access
confirmed, but a signed minimal iOS run target is still needed before benchmark
collection.

## Selected Comparators

| Role | Repo | Pinned SHA | Model ID | Status |
| --- | --- | --- | --- | --- |
| MLX Python | `Blaizzy/mlx-audio` | `862dfbe5338e91df6f74ac986b4df8bede7961a6` | `mlx-community/Kokoro-82M-bf16` | Selected |
| Core ML / iOS | `soniqo/speech-swift` | `0d09a2ed5464c7c94cf4545be59043c21f8775ea` | `aufklarer/Kokoro-82M-CoreML` | Selected |

`mlalma/kokoro-ios` remains excluded from the primary matrix because it is an
MLX Swift implementation, not the Core ML comparator required for the paper
claim.

## MLX Proof

Environment:

```bash
/Users/mm/.local/bin/uv venv /tmp/kokoro-external-bakeoff/mlx-venv
/Users/mm/.local/bin/uv pip install --python /tmp/kokoro-external-bakeoff/mlx-venv/bin/python \
  -e /tmp/kokoro-external-bakeoff/mlx-audio 'misaki[en]' \
  https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
```

Notes:

- Plain `misaki` was insufficient for English Kokoro. The adapter must install
  `misaki[en]` and explicitly install `en_core_web_sm` into the active venv.
- `model.generate(...)` returns `GenerationResult.samples == 1` for this text,
  but `audio` is the real PCM array. The benchmark adapter must compute sample
  count from `np.array(result.audio).size`.
- One-sentence proof using `voice="af_heart"`, `speed=1.0`, `lang_code="a"`
  produced one chunk with audio shape `[78000]`, 24 kHz sample rate, 3.25 s
  duration, RMS `0.034807`, and float32 PCM SHA256
  `4993dfb6a66c1e5e71cb9feda137a789473d115ab269ddcafa1cf4eb75228270`.

Cache path:

```text
/Users/mm/.cache/huggingface/hub/models--mlx-community--Kokoro-82M-bf16
```

## Soniqo Core ML Proof

Source evidence:

- `Package.swift` exposes product `KokoroTTS`.
- `Sources/KokoroTTS/KokoroTTS.swift` defines
  `KokoroTTSModel.fromPretrained(... computeUnits: MLComputeUnits = .all ...)`.
- `Sources/KokoroTTS/KokoroModel.swift` loads `MLModel` with
  `MLModelConfiguration.computeUnits`, searching for `kokoro_5s`,
  `kokoro_10s`, `kokoro_15s`, or `kokoro`.
- `docs/models/kokoro-tts.md` documents Core ML / Neural Engine usage and
  `.cpuAndGPU` fallback.

Verification:

```bash
swift test --package-path /tmp/kokoro-external-bakeoff/speech-swift \
  --filter KokoroTTSTests/testDefaultModelId
swift test --package-path /tmp/kokoro-external-bakeoff/speech-swift \
  --filter E2EKokoroTests/testSynthesizeEnglish
```

Results:

- `KokoroTTSTests/testDefaultModelId` passed after building the package.
- `E2EKokoroTests/testSynthesizeEnglish` passed in 55.099 s.
- The E2E test used `af_heart` and printed `English: 73800 samples (3.08s)`.

Cache paths:

```text
/Users/mm/.cache/huggingface/hub/models--aufklarer--Kokoro-82M-CoreML
/Users/mm/Library/Caches/qwen3-speech/models/aufklarer/Kokoro-82M-CoreML
```

## iPhone 12 Pro Access

`devicectl` can access the connected device:

```text
Device: iPhone 12 Pro, product iPhone13,3, arm64e
UDID: 00008101-001134561A0A001E
Identifier: F383FC46-FD64-5346-AEC6-59E3E2F8C9CA
iOS: 26.5 build 23F77
Transport: wired
Pairing: paired
Developer Mode: enabled
Tunnel: connected
Capabilities: install app, launch app, process control, file transfer, screen view
```

Installed apps at audit time:

```text
Hyperlearn      com.hyperlearn.app
Swift-TTS-iOS   com.lumen-digital.Swift-TTS-iOS
```

The Soniqo `Examples/iOSEchoDemo` project generated successfully with
`xcodegen generate`. A physical-device build was attempted against the iPhone
with isolated DerivedData and package caches:

```bash
xcodebuild -project iOSEchoDemo.xcodeproj -scheme iOSEchoDemo \
  -destination 'id=00008101-001134561A0A001E' \
  -derivedDataPath /tmp/kokoro-external-bakeoff/ios-derived \
  -clonedSourcePackagesDirPath /tmp/kokoro-external-bakeoff/ios-source-packages \
  -packageCachePath /tmp/kokoro-external-bakeoff/ios-package-cache \
  -skipPackagePluginValidation -skipMacroValidation \
  CODE_SIGNING_ALLOWED=NO build
```

It resolved packages and entered the iPhoneOS 26.5 build graph, but stayed
silent for several minutes after build description creation and was interrupted.
Do not count this as an installed-device synthesis proof. Phase 1 should create
a smaller signed iOS runner for Soniqo Kokoro instead of using the full echo demo
dependency graph.

## Hardware Placement

- MLX competitor: source and dependencies route through MLX/Metal. Phase 1 must
  capture runtime GPU evidence during benchmark collection.
- Soniqo Core ML competitor: source loads `MLModel` with configurable
  `MLComputeUnits`; default `.all` is the ANE-preferred path. Phase 1 must
  capture Core ML Instruments or `powermetrics --samplers ane` evidence during
  benchmark collection.
- The connected iPhone is available for physical-device Core ML testing, but
  the first benchmarkable device runner should be a minimal signed target.
