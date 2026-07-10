# iPhone Core ML Device Lab Runbook

This guide ingests the external iPhone Core ML device-lab report. Treat the raw
report as research input, not canonical truth.

Raw report:

- `/Users/mm/Documents/GitHub/llm-workflows/outputs/create-guide/iphone-core-ml-device-lab-runbook-for-signing-profiling-and-evidence-capture-ret/2026-06-06T23-25-36-950Z/raw-report.md`

## Executive Summary

iPhone proof has two separate jobs:

1. prove the device/app/signing path is operational;
2. prove warmed Core ML inference speed after compile/cache effects are gone.

Do not collapse those into one result. A locked phone, missing trust prompt,
disabled Developer Mode, backgrounded GPU path, or code-signing failure is not a
Kokoro model-speed result.

## Device-Lab Preconditions

| Area | Required state |
| --- | --- |
| Device | Unlocked, trusted over USB, Developer Mode enabled. |
| App | Foregrounded for GPU-including compute units; avoid background timing rows. |
| Power | Plugged in; record Low Power Mode and thermal state. |
| Signing | Known development team/profile, unlocked macOS keychain if building headlessly. |
| Timing | Separate install, launch, compile/load, warmup, and warmed prediction loops. |

## Evidence To Capture

- exact device model, OS version, UDID or redacted identifier;
- Xcode and host macOS versions;
- build/sign/install/launch command output;
- model package path and checksum;
- compute-unit matrix;
- raw warmed per-iteration timings for `3s`, `7s`, `10s`, `15s`, and `30s`;
- first-run compile/cache timing stored outside the inference table;
- logs for `ANECompilerService`, Core ML, and app stdout/stderr when a failure
  happens.

## Foreground Policy

If the Kokoro iPhone path uses `.all` or `.cpuAndGPU`, keep the app foregrounded
and the device unlocked while measuring. If a background run is necessary, force
a compute-unit path that does not require a foreground Metal context and label
the row as a background-path experiment.

## Do / Avoid

| Do | Avoid |
| --- | --- |
| Treat iPhone setup failures as lab-state failures. | Reporting them as model failures. |
| Cache/compile the model before warmed timing. | Including first-load compile in inference speed. |
| Preserve raw logs and command lines. | Keeping only the headline median. |
| Run every runtime bucket independently. | Warming one shape and timing another. |

## Related Documentation

- [Apple Silicon warmed-inference benchmark hygiene](Apple-Silicon-warmed-inference-benchmark-hygiene-guide.md)
- [Core ML ANE compiler failure triage](CoreML-ANE-compiler-failure-triage-guide.md)
- [Core ML compute unit scheduling](CoreML-Compute-Unit-Scheduling-guide.md)
- [Kokoro external bakeoff plan](../../Plans/kokoro-external-bakeoff-v1.md)
