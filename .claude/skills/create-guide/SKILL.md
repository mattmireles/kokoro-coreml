---
name: create-guide
description: >-
  Creates technical field guides and debugging manuals from web search and external sources using Gemini Deep Research
  Max via the llm-workflows create_guide_v1 agent. A guide is a reference manual
  for a hard, non-obvious external mechanism a developer would Google for hours or
  ask an SME about and that LLMs handle poorly — not product, UX, architecture, or
  policy decisions. Assembles structured research briefs, launches long-running
  background research, saves raw drafts locally, and hands off to guide-ingest for
  verification and repo integration. Use when the user invokes create-guide or
  wants a new README/Guides field guide on an arcane mechanism; do not use for
  decisions the plan and agent can reason about from training data.
disable-model-invocation: true
---

# Create Guide

## Purpose

Automate the first half of field-guide creation:

1. Assemble a strong Deep Research brief (not a one-liner).
2. Launch `create_guide_v1` in `llm-workflows` (Deep Research Max by default).
3. Poll or resume until raw markdown lands locally.
4. Hand the raw draft to repo-specific ingestion (`guide-ingest` when available).

Raw Deep Research output is **draft material only**. Never treat it as canonical until verified and integrated.

Guide/notes boundary: `README/Guides/` is for externally created reference
manuals after `raw-report.md` exists and has been ingested. A pending
checkpoint, prompt, local summary, or agent-written synthesis is not a guide.
Do not create or update `README/Guides/` unless you can point to the external
raw source path, usually `llm-workflows/outputs/create-guide/**/raw-report.md`.
Local briefs, prompts, repo analysis, verification notes, experiment logs, and
implementation decisions belong in `README/Notes/`, not `README/Guides/`.

## What A Guide Is For (Read This First)

A guide is a **reference manual for a hard, non-obvious mechanism** — the kind
of thing a competent human developer would Google for hours, dig through forums,
WWDC sessions, and source code for, or consult a subject-matter expert about
before or while building. It exists to give the next agent **evidence-backed
answers about how something actually works** that it would otherwise guess at.

A topic qualifies only when **all three** hold:

1. **Non-obvious mechanism.** There is a hidden gotcha, an undocumented platform
   behavior, an internal implementation detail, or a sharp edge you cannot infer
   from first principles.
2. **A human would research it.** A real developer would search the web, read
   issue trackers / WWDC notes / spec docs, or ask an expert before trusting
   their own answer.
3. **Poorly represented in LLM training data.** Niche, new, fast-moving, or the
   kind of thing models confidently hallucinate.

Good guide topics (note the shape):

- "How does iMessage render a rich link preview with an inline playable audio
  button, and what Open Graph / metadata makes that appear?"
- "How does Core ML actually schedule ops across ANE/GPU/CPU, and what causes
  silent fallback?"
- "What exact APNs payload and entitlements wake an ActivityKit Live Activity in
  the background, and what only works on device vs Simulator?"
- "Why can a wildcard provisioning profile not carry HealthKit, and what is the
  minimum signing setup to ship it?"

## Not A Guide

Guides are **not** for product, UX, or architecture decisions. The agent and the
plan can reason about these from training data; sending them to Deep Research
wastes a run and produces a roadmap, not a reference manual. Do **not** write a
guide for:

- Product/scope choices ("which permissions should we ask for, in what order").
- UX or copy decisions (owned by design skills and the plan).
- API/schema/endpoint design ("how should we shape the token endpoint").
- Fallback/timeout/policy decisions (engineering judgment, not arcane mechanism).
- General, well-trodden how-tos already covered well by training data or
  Context7 (standard SwiftUI flows, basic REST, common library usage).

Litmus test: if the hard part is **deciding what to build**, it is not a guide.
If the hard part is **understanding how an external system actually behaves**, it
might be.

## Prerequisites

- `llm-workflows` checked out at `/Users/mm/Documents/GitHub/llm-workflows`
- `GEMINI_API_KEY` in `llm-workflows/.env` (or exported) for the default
  Deep Research Max streaming path
- `WORKFLOW_RUNTIME_TOKEN` in `llm-workflows/.env` (or exported) only when using
  `--workflow-runtime`, `--no-wait`, `--resume`, `--run-id`, or `--result-url`
- Optional: `WORKFLOW_RUNTIME_BASE_URL` (defaults to staging for workflow mode)

## Use When

- User invokes **create-guide**
- The topic clears the three-part test in **What A Guide Is For** — a
  non-obvious external mechanism a human would research or ask an SME about
- User wants to replace manual Gemini UI research with the automated workflow

## Do Not Use When

- The hard part is a product, UX, architecture, or policy **decision** (see
  **Not A Guide**) — let the plan and the agent decide, do not research it
- The topic is a well-trodden how-to already covered by training data or
  Context7 → answer directly or use **Context7**
- Raw export already exists → use **`guide-ingest`**
- Only fixing markdown in an existing guide → use **`markdown`**
- The local learning belongs in the repo, not external research → use
  **`write-notes`**
- Lighter in-session research without Deep Research → use **`research-planning`**

## Workflow

### 1. Gather the brief

Collect from the user (ask if missing):

| Field | Required | Notes |
| --- | --- | --- |
| `topic` | Yes | One-line guide subject |
| `context` | Often | Ground truth the external researcher cannot see |
| `primaryResearchGoal` | Often | Narrow objective |
| `questions` | Often | Bullet list of questions to answer |
| `sourceHints` | Optional | Docs, repos, papers to prioritize |
| `avoidSources` | Optional | Sources to distrust |
| `targetRepo` | Optional | Active repo name, e.g. `kokoro-coreml`, `originals-condom` |
| `targetGuidePath` | Optional | e.g. `README/Guides/apple-silicon/example-guide.md` |

Default `agentMode` to **`max`**. Confirm before launching unless the user explicitly asked for fast/cheap (`default`).

### 2. Assemble the research prompt

Follow the scratchpad pattern (see `Scratchpad/scratchpad.md` in active repo when present):

- Advanced developer field guide
- Best practices, worst practices, common bugs, hidden gotchas, known limitations
- Go light on theory, heavy on practical detail with code
- For complex topics: add Context, Primary research goal, Questions to answer, Output format
- Mark speculative ideas separately from evidence-backed recommendations
- For text-only field guides: explicitly request no charts, images, diagrams,
  or generated visualizations. The runtime should also send
  `agent_config.visualization: "off"` for `create_guide_v1`.

Write a local brief file when helpful, then pass it via `--context-file` or
`--prompt-file`. If the active repo has `README/Notes/`, put the brief there
with a clear prompt/provenance name. Do not place briefs, local analysis, or
unverified drafts in `README/Guides/`.

### 3. Launch the research run

From `llm-workflows`, run Max to completion with the local direct Gemini
Interactions streaming transport:

```bash
cd /Users/mm/Documents/GitHub/llm-workflows

pnpm run research:create-guide \
  --topic "Core ML vs MLX scheduling for ISTFTNet vocoders" \
  --context-file /path/to/brief.md \
  --target-repo kokoro-coreml \
  --target-guide-path README/Guides/apple-silicon/example-guide.md \
  --agent-mode max
```

For Max, do not use `--no-wait`: the local streaming process is the reliable
completion path and must stay alive to write `raw-report.md`. Use
`--workflow-runtime` only when explicitly testing the Cloudflare Workflow
transport.

For a non-Max or workflow-mode run:

```bash
pnpm run research:create-guide \
  --topic "Core ML vs MLX scheduling for ISTFTNet vocoders" \
  --context-file /path/to/brief.md \
  --target-repo kokoro-coreml \
  --target-guide-path README/Guides/apple-silicon/example-guide.md \
  --agent-mode default \
  --workflow-runtime
```

Artifacts land under:

`outputs/create-guide/<slug>/<timestamp>/`

- `input.json` — workflow input
- `prompt.md` — human-readable brief preview
- `checkpoint.json` — resume pointer
- `run.json` — polling status
- `raw-report.md` — final draft (when complete)
- `result.json`, `metadata.json`
- `events.ndjson`, `thought-summaries.md`, `interaction-id.txt` — direct
  Max streaming diagnostics

### 4. Resume if interrupted

```bash
pnpm run research:create-guide \
  --resume \
  --checkpoint outputs/create-guide/<slug>/<timestamp>/checkpoint.json
```

If only the remote run is known, resume with `--run-id <id>` or `--result-url <url>`.
Do **not** restart a Max run if a checkpoint or run id exists. Resume first.

Direct local Max streaming runs cannot currently be resumed after the local
process exits. If interrupted, inspect `events.ndjson`, `interaction-id.txt`,
and `raw-report.md` before starting another run.

### 4.1. Triage provider failures

If the run fails before `raw-report.md` exists:

1. Inspect `run.json`, `checkpoint.json`, and any saved internal attempt
   artifacts before relaunching.
2. If the provider response body is redacted, recover details from the
   Interactions stream when possible: `GET /v1beta/interactions/<id>?stream=true`
   with the same Gemini key and API revision.
3. Do not default to "prompt too large" as the diagnosis. For
   `provider_invalid_response` / `invalid_request` after thought summaries such
   as "Considering Visual Elements", check the request artifact for
   `agentConfig.visualization`. It must be `"off"` for markdown-only guide runs.
4. If the request enabled or omitted visualization, rerun only after the runtime
   or prompt has been fixed to keep the guide text-only.
5. If the request is already text-only and the recovered stream is only
   `interaction.created` / `status_update` / `invalid_request` or provider
   `api_error` with no usable report text, do not launch another Max duplicate.
   Treat the Max guide as **blocked**, not complete. A default-agent rerun is a
   separate, explicitly downgraded deliverable and requires user authorization;
   it must never satisfy a request for Deep Research Max.

Current known failure signature, confirmed 2026-06-18 with raw REST streaming,
the official collaborative-planning flow, a no-`agent_config` stream,
`@google/genai@2.8.0`, and a no-explicit-tools direct stream matching the
documented default Deep Research tool set: Deep Research Max accepts the
interaction, often emits thought summaries through draft/polish/final editing,
then returns stream event
`{"code":"api_error","message":"There was a problem processing your request. You will not be charged."}`
or resume returns `400 invalid_request`, or polling remains stale at the
original `user_input` timestamp before any final text. That is provider-side Max
failure unless a later run produces non-empty `raw-report.md`.

### 5. Hand off to ingestion

When `raw-report.md` exists:

| Repo has `guide-ingest`? | Action |
| --- | --- |
| Yes (e.g. kokoro-coreml) | Invoke **`guide-ingest`** on `raw-report.md` |
| No | Normalize markdown, verify API claims (Context7 when available), add cross-links per local `README/Guides/` conventions |

Ingestion steps (never skip):

0. Provenance gate: confirm and record the external raw source path. If there
   is no raw external artifact, stop and write a note instead.
1. Mechanical cleanup (escapes, data-URI images, broken fences)
2. Context7 verification for library/API claims
3. Outbound links to existing guides/notes
4. Inbound links from related docs
5. Place final file at `targetGuidePath` or agreed location

Leave **Related Documentation** empty during Deep Research; ingestion adds repo cross-links.

## Prompt Writing Guidelines

The prompt must be 100% self-contained. The research agent has no other context than the context that you share with it. It cannot look at your code unless you share your code. It can only search the web (and hit any MCP you wire up).

The more and richer context you give it, the more relevant the guide will be.

## Prompt template (minimal)

```markdown
Create an advanced developer field guide on: [TOPIC]

Best practices? Worst practices? Common bugs and issues? Hidden gotchas?
Idiosyncratic design quirks? Known limitations?

Be extensive. Be comprehensive. Create a reference manual for implementation and debugging.
Go light on theory, heavy on practical detail with code examples.

Context:
[GROUND TRUTH]

Primary research goal:
[GOAL]

Questions to answer:
- [Q1]
- [Q2]

Output format:
- Executive summary first
- Do-this / avoid-this tables
- Failure modes and debugging section
- Concrete profiling commands/tools when relevant
- Numbered references
- Mark speculation separately from evidence-backed recommendations
```

## Handoff rules

| Situation | Skill |
| --- | --- |
| Raw draft ready | **`guide-ingest`** (if available) |
| Markdown lint only | **`markdown`** |
| Where to put a note | **`write-notes`** |
| Lighter research without Deep Research | **`research-planning`** |

## References

- Runner: `/Users/mm/Documents/GitHub/llm-workflows/scripts/run-create-guide-research.mjs`
- Workflow: `create_guide_v1` in `/Users/mm/Documents/GitHub/llm-workflows`
- Deep Research guide: `/Users/mm/Documents/GitHub/llm-workflows/README/guides/gemini/Gemini-Deep-Research-agents-guide.md`
- Ingest skill: `guide-ingest` in repos that ship it
