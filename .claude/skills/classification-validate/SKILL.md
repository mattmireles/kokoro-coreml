---
name: classification-validate
description: >-
  Validate Kokoro experiment and listening-decision labels, gate categories,
  and result manifests. Use for pass/warn/fail decision records, F0 or audio
  listening reviews, bakeoff classifications, or preventing invented result
  categories. Do not use for model export numerical parity or generic data
  labeling.
---

# Validate Kokoro Decision Labels

1. Read [the repository anchors](references/index.md).
2. Identify the canonical schema or allowed decision values from the script,
   plan, and existing validated artifact.
3. Reject missing or unknown values; do not normalize a label into a new
   category after observing results.
4. Separate hard failures, caveats requiring evidence, and provisional
   observations. Preserve the underlying measurements and reviewer notes.
5. Rerun the owning validator before treating a summary or bakeoff table as
   complete.

## Guardrails

- Do not turn a subjective listening decision into numerical parity evidence.
- Do not rewrite a negative result into a warning to make a gate pass.
- Keep source audio, model revision, device, and artifact identity attached to
  every decision record.

