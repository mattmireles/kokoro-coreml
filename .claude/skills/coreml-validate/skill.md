---
name: coreml-validate
description: >-
  Validate Kokoro Core ML exports against their frozen PyTorch/reference
  behavior and audio-quality gates. Use for numerical parity, input/output
  shape drift, export regressions, CPU-fallback signals, SDK bundles, or
  release-readiness evidence.
---

# Validate Kokoro Core ML

1. Freeze the source revision, export configuration, package hashes, fixtures,
   dtypes, shapes, and threshold before observing results.
2. Run `kokoro/coreml_numeric_validate.py` and
   `kokoro/coreml_export_verify.py` or the narrow equivalent for the changed
   export.
3. Test the actual package on the target device when the claim concerns
   scheduling or deployment; separate numerical parity from audio/listening
   quality and performance.
4. Run `scripts/validate_sdk_bundle.mjs` for hosted SDK bundle changes.
5. Fail closed on shape, dtype, output, fallback, package-identity, or fixture
mismatches. Never loosen a preregistered threshold after seeing a result.
