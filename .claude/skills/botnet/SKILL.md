---
name: botnet
description: >-
  Master skill for managing the owner-operated Botnet Mac fleet across gist,
  botnet, and kokoro-coreml. Use when checking fleet health, rolling Mac
  workers, debugging scrape.remote_browser, tts.kokoro_coreml, or precompute
  workers, handling LaunchAgents, SSH fan-out, Cloudflare control-plane health,
  Kokoro/Core ML runtime issues, botnet fleet env vars, or writing fleet
  debugging notes.
---

# Botnet Fleet

## Purpose

Use this as the master skill for Botnet fleet operations. It coordinates the
cross-repo system made of:

- `botnet`: control plane, SSH rollout scripts, remote browser worker, native
  Kokoro worker, precompute workers, fleet inventory, and operator notes.
- `gist`: upstream extension/runtime behavior and Gist-facing product contract.
- `kokoro-coreml`: Swift/Core ML model export, runtime assets, profiling, and
  validation evidence for native TTS.

Read [`reference.md`](reference.md) before acting on anything non-trivial. It is
the path catalog for related skills, runbooks, notes, environment variables, and
historical decisions.

## Use When

- The user mentions the Botnet fleet, Mac fleet, `m2-studio`, `irvine-m1`,
  `m2-air`, SSH rollout, `update-fleet`, `update-host`, or fleet health.
- The task involves `scrape.remote_browser`, `tts.kokoro_coreml`, reader-audio
  precompute workers, worker LaunchAgents, queue depth, worker freshness,
  `workersByMachine`, active slots, or per-machine capability readiness.
- The failure crosses `gist`, `botnet`, and `kokoro-coreml`, or depends on which
  repo owns the broken contract.
- The task asks where Botnet fleet env vars, worker keys, admin bearer keys, or
  Cloudflare deploy credentials live.

## First Rule

Start in `botnet` unless there is strong evidence the root cause belongs
elsewhere. `botnet` owns the live control plane and fleet scripts; `gist` and
`kokoro-coreml` are upstream or dependency repos for specific contracts.

## Required Workflow

1. Read [`reference.md`](reference.md), then load only the related skill or note
   for the current failure class.
2. Prove current fleet state before changing anything:
   `cd /Users/mattmireles/Documents/Github/botnet` and run the checked-in
   health workflow from `fleet-health-check` when live status matters. From this
   operator Mac you can SSH directly to `m2-studio` and `irvine-m1` for remote
   inspection and rollouts—see `reference.md` § Operator Workstation.
3. Repair the failing primitive, not the whole system:
   control-plane health, SSH reachability, LaunchAgent state, scraper worker,
   Kokoro worker, precompute worker, or Cloudflare deploy.
4. Use `fleet-update` for rollouts, `fleet-health-check` for health triage,
   `remote-worker` for browser worker incidents, and Kokoro/Core ML skills or
   docs for TTS runtime defects.
5. Verify with the operator command that matches the repair. Do not call the
   fleet healthy from a single local symptom.
6. For debugging that takes real investigation, write or update notes before
   finishing. Prefer `botnet/README/notes/` for fleet incidents; use `gist` or
   `kokoro-coreml` notes only when that repo owns the durable lesson.

## Guardrails

- Do not print secret values from `.env`, worker keys, admin bearer keys, or
  Cloudflare tokens.
- Do not deploy the shared Cloudflare API worker unless the user asked for a
  deploy or the rollout requires `pnpm update-fleet --deploy-api`.
- Do not rewrite imported Gist worker code inside `botnet` casually. Preserve
  upstream import boundaries and sync deliberately.
- Do not treat `gist`'s older Mac mini runbooks as the live source of truth when
  `botnet` has newer Botnet-branded fleet docs.
- Do not change Kokoro model/runtime policy without checking `kokoro-coreml`
  evidence and `botnet` Kokoro throughput notes.
