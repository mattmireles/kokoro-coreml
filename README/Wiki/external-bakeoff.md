---
title: External Bakeoff
last_synced: 2026-06-06
sources:
  - README/Notes/external-bakeoff-phase0-api-audit.md
  - README/Notes/external-bakeoff-phase2-run-log.md
  - scripts/external_bakeoff/README.md
  - scripts/external_bakeoff/verify_external_bakeoff_completion.py
---

# External Bakeoff

## Current Belief

The external bakeoff is not complete because artifacts exist. Completion is
defined by `scripts/external_bakeoff/verify_external_bakeoff_completion.py`.

Machine-checkable progress and human listening decisions are separate gates.
Signing, iOS runner availability, and blank listening decisions must be treated
as live blockers until the verifier proves otherwise.

## Do Not Break

- Do not infer bakeoff completion from generated manifests alone.
- Do not overwrite human listening decisions while regenerating review files.
- Keep Config F and external runner outputs comparable by schema, not by file
  naming vibes.

## Executable Memory

Regression test:

```bash
python scripts/external_bakeoff/verify_external_bakeoff_completion.py
```
