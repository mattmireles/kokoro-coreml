# In-Repo Memory Guide

This repo uses git-native markdown memory. The wiki is a compiled navigation
layer; the source of truth remains code, scripts, checked-in plans, notes, and
measured outputs.

## First Stop

Start at [README/Wiki/README.md](../Wiki/README.md) for current belief. Then
drill into canonical sources listed on each page.

## Rules

- No vector DB is the default. Add one only after grep, indexes, and source
  links fail on a measured workflow.
- Every current-belief page must have `last_synced` and `sources:`.
- Do not cite `README/Plans/` as durable truth unless the question is about plan
  status. Plans are intent; notes, scripts, outputs, and code are evidence.
- Bug fixes must leave executable memory: a regression command or an explicit
  `Not testable:` reason.
- Hard-to-reverse architecture decisions need an ADR or an explicit ADR
  exception in the plan or note.

## Health Gate

```bash
node scripts/memory-health.mjs --write-coverage
node scripts/memory-health.mjs --strict
```

The coverage index is deterministic. It proves the wiki can route back to every
canonical memory source without pretending the coverage index is current belief.
