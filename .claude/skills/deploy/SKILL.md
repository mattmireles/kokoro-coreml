---
name: deploy
description: >-
  Clarifies what “ship” means for kokoro-coreml: there is no Cloudflare-style
  multi-worker deploy script. Use for tagging releases, pushing the branch,
  coordinating with the macOS app repo, or verifying exports before handoff.
  Before treating a revision as releasable, run the repo’s primary checks (see
  git-commit / audit). Do not use when the user only wants local experiments
  with no remote or release intent.
---

# Deploy (kokoro-coreml)

## Purpose

This repository is a **PyTorch → Core ML** conversion and model-artifact tree for
Kokoro TTS. It is **not** the Roast Cloudflare monorepo: there are **no**
`pnpm deploy:*` scripts or Workers/Pages pipelines here.

“Deploy” in this context means **release engineering** appropriate to this repo:

- **Git:** push branches, tags, or PRs (often via **`git-push`**).
- **Artifacts:** `.mlpackage` bundles under `coreml/`, checkpoints, and export
  scripts consumed by a separate macOS app (e.g. TalkToMe).
- **Verification:** run **`pytest`** (or targeted export smoke scripts) before
  declaring a revision ready for downstream integration.

## Default interpretation

- **“Ship it” / “deploy”** → confirm what the user means: **push to GitHub**,
  **cut a tag**, or **hand models to the app**—not a cloud deploy unless they
  name another system.

## Pre-ship gate (recommended)

From the **repository root**, when a change touches **Python export, Core ML, or
tests**:

1. **`pytest`** (full test pass when tests are present).
2. If the change is export-only, at minimum run the **relevant** `examples/` or
   `test_*.py` scripts the guides mention.

There is no single CI workflow in `.github/workflows/` for this repo yet; treat
**local green tests** as the main gate before pushing risky changes.

## Procedure (agent)

1. Read **`README.md`**, **`CLAUDE.md`** (agent operating rules), **`README/Wiki/`**
   plus relevant **`README/Guides/`** (PyTorch → Core ML playbooks), and the
   relevant **`README/*.md`** guide for the subsystem (conversion, runtime,
   export).
2. Do **not** run Cloudflare/Roast deploy commands—they do not apply.
3. Run [Pre-ship gate](#pre-ship-gate-recommended) before advising “ready to
   ship.”
4. For **git remote** operations, follow **`git-push`** when the user wants push
   + CI (if added later).

## Anti-patterns

- Copying **Roast** `pnpm deploy:staging` / Workers steps into this repo.
- Pushing large binary or regenerated **`.mlpackage`** changes without noting
  what was regenerated and how to verify.

## Related skills

- **`git-push`:** commit, merge, push, chase CI when CI exists.
- **`git-commit`:** post-commit **`pytest`** awareness (non-blocking heads-up).
- **`audit`:** full review when the user says **`audit`**.
