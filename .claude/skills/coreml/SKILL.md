---
name: coreml
description: >-
  Redirect: cross-repo Core ML master router lives in personal skills. Use
  ~/.cursor/skills/coreml/SKILL.md for intent routing and guide links across
  kokoro-coreml and crossfade. Delegates procedures to repo child skills
  (coreml-validate, coreml-profile, bakeoff, debug).
---

# Core ML (repo stub → personal skill)

**Canonical router:** `~/.cursor/skills/coreml/SKILL.md`

**Full guide index:** `~/.cursor/skills/coreml/reference.md`

This checkout is **kokoro-coreml**. The personal skill detects that from the
workspace path and loads Kokoro guides under
`/Users/mm/Documents/GitHub/kokoro-coreml/`.

## What to do

1. Read and follow `~/.cursor/skills/coreml/SKILL.md` end-to-end.
2. Load child skill **procedures** from this repo when delegated:
   - [coreml-validate](../coreml-validate/skill.md)
   - [coreml-profile](../coreml-profile/skill.md)
   - [bakeoff](../bakeoff/SKILL.md)
   - [debug](../debug/SKILL.md)
   - [guide-ingest](../guide-ingest/SKILL.md)

Do not duplicate routing logic in this file — edit the personal skill instead.
