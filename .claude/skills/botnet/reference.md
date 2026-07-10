# Botnet Fleet Reference

## Repo Roots

- `botnet`: `/Users/mattmireles/Documents/Github/botnet`
- `gist`: `/Users/mattmireles/Documents/Github/gist`
- `kokoro-coreml`: `/Users/mattmireles/Documents/Github/kokoro-coreml`

## Related Skills

Read these first when they match the active failure:

- `botnet/.claude/skills/fleet-health-check/SKILL.md`: health pass, DNS/Bonjour
  triage, SSH reachability, queue/freshness interpretation.
- `botnet/.claude/skills/fleet-update/SKILL.md`: SSH fan-out, `pnpm
  update-fleet`, `pnpm update-host`, rollout verification.
- `botnet/.claude/skills/remote-worker/SKILL.md`: `scrape.remote_browser`,
  worker extension artifact, MV3 keepalive, CDP/debugger recovery, DLQ.
- `botnet/.claude/skills/debug/SKILL.md`: systematic botnet debugging loop.
- `botnet/.claude/skills/write-notes/SKILL.md`: durable botnet notes.
- `gist/.claude/skills/kokoro/SKILL.md`: Gist-facing Kokoro browser/native
  context and sibling repo links.
- `gist/.claude/skills/auto-gist-debug/SKILL.md`: auto-gist or suggestion-bar
  failures after Botnet scrape output reaches Gist.
- `gist/.claude/skills/chrome-extension-debug/SKILL.md`: generic MV3/runtime
  issues not specific to Botnet's worker artifact.
- `kokoro-coreml/.claude/skills/coreml-validate/skill.md`: Core ML validation.
- `kokoro-coreml/.claude/skills/coreml-profile/skill.md`: Core ML profiling.
- `kokoro-coreml/.claude/skills/bakeoff/SKILL.md`: Kokoro bakeoff evidence.

## Source Of Truth

Use `botnet` as the live fleet source of truth:

- `botnet/README.md`: current overview, fleet update flow, env requirements,
  current worker IDs, roles, host paths, and update-host behavior.
- `botnet/.fleet-worker-ids`: canonical expected worker IDs.
- `botnet/.fleet-host-roles`: expected LaunchAgent roles per worker ID.
- `botnet/.fleet-hosts`: SSH fan-out targets and checkout paths.
- `botnet/README/docs/worker-ops.md`: remote browser worker artifact,
  prove-live, operator smoke tests.
- `botnet/README/docs/remote-worker-runbook.md`: macOS browser worker install,
  Botnet.app, profiles, LaunchAgent labels, stable extension path.
- `botnet/README/docs/kokoro-worker-ops.md`: native Kokoro worker install,
  LaunchAgent, runtime bundle, health, canaries, concurrency policy.
- `botnet/README/docs/security.md`: key provisioning and secret handling.

`gist` still matters for upstream contracts:

- `gist/README/Guides/chrome-extension/Remote-browser-worker-guide.md`
- `gist/README/Runbooks/remote-worker-observability.md`
- `gist/README/Runbooks/remote-worker-mac-mini-operations.md`
- `gist/README/Notes/chrome-extension/remote-worker-notes.md`
- `gist/README/Guides/Kokoro-TTS-browser-guide.md`
- `gist/README/Notes/infrastructure/m2-air-kokoro-thermal-soak-notes.md`

`kokoro-coreml` matters for model/runtime evidence:

- `kokoro-coreml/README.md`
- `kokoro-coreml/README/Notes/kokoro-runtime-boundary.md`
- `kokoro-coreml/README/Notes/bakeoff-results-v2.md`
- `kokoro-coreml/README/Guides/apple-silicon/CoreML-Compute-Unit-Scheduling-guide.md`
- `kokoro-coreml/README/Guides/apple-silicon/CoreML-LSTM-Enumerated-Shapes.md`
- `kokoro-coreml/README/Notes/coreml-compute-unit-ablation.md`

## Operator Workstation

The primary operator Mac—the one with `gist` and `botnet` checkouts under
`/Users/mattmireles/Documents/Github/`—can SSH directly into **`m2-studio`**
and **`irvine-m1`**. Treat this machine as the fleet console: run
`pnpm update-fleet`, health passes, and ad hoc remote commands from here without
needing physical access to those workers.

Direct SSH targets from this machine:

```bash
ssh mm@m2-studio.local
ssh mattmireles@irvine-m1.local
```

Use the matching checkout paths from `.fleet-hosts` when running remote
commands (for example `pnpm update-host` on the remote checkout). Do not assume
SSH reachability to every fleet host from every machine—`m2-air` may require
local access or a different operator path.

## Fleet Inventory

Current expected worker IDs:

- `m2-studio`
- `irvine-m1`
- `m2-air`

Current SSH fan-out targets live in `botnet/.fleet-hosts`:

- `mm@m2-studio.local=/Users/mm/Documents/GitHub/botnet`
- `mattmireles@irvine-m1.local=/Users/mattmireles/Documents/GitHub/botnet`
- `mattmireles@M2-Air.local=/Users/mattmireles/Documents/Github/botnet`

Current expected roles live in `botnet/.fleet-host-roles`:

- `m2-studio=scrape,tts,precompute`
- `irvine-m1=scrape,tts,precompute`
- `m2-air=scrape,tts,precompute`

## Environment Variables

Do not print secret values. Read `.env` only when needed to run commands.

Primary env files:

- `botnet/.env`: live fleet/control-plane operations.
- `gist/.env`: Gist app, extension, and upstream integration work.
- `kokoro-coreml/.env`: local Kokoro/Core ML validation or profiling work.
- `botnet/apps/api-worker/.dev.vars`: local Worker dev vars when running
  `wrangler dev`.

Common `botnet/.env` names:

- `WEB_SCRAPER_BASE_URL` or `PROVE_LIVE_BASE_URL`: control-plane URL.
- `INTERNAL_ADMIN_BEARER_KEY`, `PROVE_LIVE_ADMIN_BEARER`, or
  `WEB_SCRAPER_ADMIN_BEARER`: admin health bearer.
- `WEB_SCRAPER_API_KEY`: caller API key for scrape/TTS canaries.
- `REMOTE_WORKER_KEY`: per-worker API key persisted outside git.
- `REMOTE_WORKER_ID`: optional health scope or rollback override.
- `KOKORO_COREML_ROOT`: sibling `kokoro-coreml` checkout path when non-default.
- `KOKORO_INSTALL_MAX_CONCURRENT_JOBS`,
  `KOKORO_INSTALL_HOST_PRESSURE_ADAPTIVE`,
  `KOKORO_INSTALL_P1_EXTRA_CHUNK_CONCURRENCY`, and
  `KOKORO_INSTALL_P1_BURST_JOBS`: installer-preserved TTS capacity policy.
- `TTS_STICKY_ASSIGNMENT_ENABLED`: Cloudflare api-worker rollback fuse.
- `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN`: required only on the
  host that deploys `apps/api-worker`.
- `GIST_BACKEND_URL`, `GIST_ADMIN_API_KEY`, `PRECOMPUTE_SLIDES_FILE`, and
  `WEB_SCRAPER_CHECKOUT`: precompute worker bootstrap inputs.

## Default Commands

Run from `/Users/mattmireles/Documents/Github/botnet`:

```bash
./scripts/fleet-health-pass.sh 90
pnpm update-fleet
pnpm update-fleet --keep-going
pnpm update-fleet --deploy-api
pnpm update-host
pnpm check:worker-health
pnpm check:tts-worker-health
pnpm prove-live --help
pnpm prove-live:tts --help
pnpm operator:scrape-smoke
pnpm operator:tts-smoke
pnpm kokoro-worker:doctor
pnpm kokoro:calibrate
```

Use `--deploy-api` sparingly. It deploys the shared Cloudflare control plane
from the first host in `.fleet-hosts`, then updates the remaining Macs.

## Notes And History

Prefer updating an existing high-level note over creating a one-off file:

- Fleet/Kokoro throughput: `botnet/README/notes/kokoro-throughput-interventions.md`
- Kokoro worker bugs: `botnet/README/notes/kokoro-debug-notes.md`
- Remote browser worker bugs: `botnet/README/notes/remote-worker-debug-notes.md`
- Worker identity migration: `botnet/README/notes/worker-identity-migration.md`
- Plan 014 naming/TCC history: `botnet/README/notes/plan-014-tcc-preflight.md`
  and `botnet/README/notes/plan-014-rename-postmortem.md`
- Gist upstream remote worker notes:
  `gist/README/Notes/chrome-extension/remote-worker-notes.md`
- Kokoro runtime boundary:
  `kokoro-coreml/README/Notes/kokoro-runtime-boundary.md`

When debugging takes more than a quick pass, use the active repo's `write-notes`
skill. Fleet incidents normally belong in `botnet/README/notes/`; upstream Gist
runtime lessons belong in `gist/README/Notes/`; Core ML export/runtime lessons
belong in `kokoro-coreml/README/Notes/`.

## Historical Constraints

- Current Botnet naming uses `Botnet.app`, `com.botnet.scraper.<slug>`,
  `com.botnet.tts.<slug>`, and `com.botnet.precompute.<mode>`, but some
  on-disk directories intentionally still use older `web-scraper-*` names.
- Scrape and TTS health are independent. A degraded browser worker must not gate
  TTS readiness, and a degraded TTS worker must not gate scrape fallback.
- TTS should be null-assigned at enqueue unless the rollback fuse explicitly
  restores sticky assignment.
- Kokoro production workers must use pinned runtime bundles. No deployed worker
  should depend on an unpinned local `kokoro-coreml` checkout.
- Queue reachability, worker freshness, LaunchAgent state, and host memory
  pressure have historically dominated before model-level performance changes.
