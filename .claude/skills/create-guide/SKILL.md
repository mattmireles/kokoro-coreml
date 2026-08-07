---
name: create-guide
description: >-
  Creates technical field guides by synthesizing what the public internet says
  about a hard, non-obvious external domain via Gemini Deep Research Max
  (llm-workflows create_guide_v1). A guide is a durable “afternoon of Googling”
  artifact for kokoro-coreml — not dial-a-friend advice, campaign go/no-go, or product decisions.
  Assembles research briefs, launches long-running research, saves raw drafts
  locally, and hands off to guide-ingest. Use when the user invokes create-guide
  or wants a new README/Guides field guide on an arcane mechanism.
disable-model-invocation: true
---

# Create Guide

## Purpose

Automate the first half of field-guide creation for this Core ML TTS repo:

1. Assemble a strong Deep Research brief (not a one-liner).
2. Launch `create_guide_v1` in `llm-workflows` (Deep Research Max by default).
3. Poll or resume until raw markdown lands locally.
4. Hand the raw draft to repo-specific ingestion (`guide-ingest` when available).

A guide is a **synthetic library entry**: what competent practitioners would find
if they spent an afternoon searching docs, issues, WWDC notes, papers, and
forums. It exists so the next agent can reuse that research instead of guessing
or re-Googling.

Raw Deep Research output is **draft material only**. Never place it in
`README/Guides/` until `guide-ingest` has normalized it, checked library/API
claims, annotated firsthand contradictions, and cross-linked the corpus.

Guide/notes boundary: `README/Guides/` is for externally researched reference
manuals after `raw-report.md` exists and has been ingested. A pending
checkpoint, prompt, local summary, or agent-written synthesis is not a guide.
Do not create or update `README/Guides/` unless you can point to the external
raw source path, usually `llm-workflows/outputs/create-guide/**/raw-report.md`.
Local briefs, prompts, repo analysis, verification notes, experiment logs, and
implementation decisions belong in `README/Notes/`, not `README/Guides/`.

## What A Guide Is For (Read This First)

A guide **synthesizes external knowledge about a domain** — mechanisms, gotchas,
failure modes, recipes people report, known limitations. It is the artifact of
research, not a recommendation engine for this repo’s next move.

A topic qualifies only when **all three** hold:

1. **Non-obvious external domain.** Hidden platform behavior, sharp library
   edges, undocumented scheduling, signing/entitlement traps, or similar — not
   something you invent from first principles.
2. **A human would Google it.** A developer stuck or unfamiliar would spend real
   time in docs, issue trackers, WWDC notes, or SME conversations.
3. **Poorly represented in LLM training data.** Niche, new, fast-moving, or the
   kind of thing models confidently hallucinate.

Good guide topics for this repo (shape: *what does the world know about X?*):

- "How does Core ML actually schedule ops across ANE/GPU/CPU, and what causes
  silent fallback?"
- "What ISTFTNet / vocoder conversion traps appear when targeting ANE fixed
  shapes?"
- "How do practitioners split TTS pipelines so sequential BERT/LSTM stays off
  ANE while dense decoder/vocoder stays on it?"
- "What Swift Core ML prediction patterns keep multi-model pipelines warm
  without thrashing the Neural Engine?"

## Not A Guide (Dial-A-Friend / Advice)

**create-guide is not “ask a smarter model what we should do.”** That is
**`second-opinion`**. Do **not** use create-guide when the ask is advice,
judgment, or a campaign decision dressed up as research.

Reject or rewrite briefs that ask:

- "Should we put the decoder on ANE for *this* bakeoff?"
- "Is this Core ML bucket split the right next arm given our RTF?"
- "What should we do about these silent ANE fallbacks?"
- Product/scope choices, UX/copy, API shape, timeout/policy judgment
- General how-tos already covered well by training data or Context7

Litmus tests:

| Ask shape | Skill |
| --- | --- |
| "What does the internet say about how X works / fails / is tuned?" | **create-guide** |
| "Given our plan/numbers, should we do X?" | **`second-opinion`** (or plan/notes for durable campaign decisions) |
| Local measurement / investigation write-up | **`write-notes`** |
| Lighter in-session research without Deep Research | **`write-notes`** |

If the hard part is **deciding what to build next**, it is not a guide.
If the hard part is **stocking external domain knowledge** an agent would
otherwise Google, it might be.

Context in the brief may ground the researcher (pinned versions, hardware, what
you already tried) so the synthesis stays relevant. Context must **not** turn
the run into "advise our campaign." Prefer questions like "what ratios and
failure modes appear in the literature?" over "are we justified to launch?"

## Prerequisites

- `llm-workflows` checked out at `/Users/mm/Documents/GitHub/llm-workflows`
- `GEMINI_API_KEY` in `llm-workflows/.env` (or exported) for the default
  Deep Research Max streaming path
- `WORKFLOW_RUNTIME_TOKEN` in `llm-workflows/.env` (or exported) only when using
  `--workflow-runtime`, `--no-wait`, `--resume`, `--run-id`, or `--result-url`
- Optional: `WORKFLOW_RUNTIME_BASE_URL` (defaults to staging for workflow mode)

## Use When

- User invokes **create-guide**
- The topic clears the three-part test — external domain knowledge a human would
  research when stuck or unfamiliar
- User wants to replace manual Gemini UI research with the automated workflow

## Do Not Use When

- The ask is advice, go/no-go, or "what should we do next" (see **Not A Guide**)
- The topic is a well-trodden how-to already covered by training data or
  Context7 → answer directly or use **Context7**
- Raw export already exists → use **`guide-ingest`**
- Only fixing markdown in an existing guide → use **`markdown`**
- The local learning belongs in the repo, not external research → use
  **`write-notes`**
- Lighter in-session research without Deep Research → use **`write-notes`**

## Workflow

### 1. Gather the brief

Collect from the user (ask if missing):

| Field | Required | Notes |
| --- | --- | --- |
| `topic` | Yes | One-line **domain** subject (mechanism / practice area) |
| `context` | Often | Ground truth the external researcher cannot see (pins, hardware). Not a request for advice. |
| `primaryResearchGoal` | Often | What external knowledge to synthesize |
| `questions` | Often | Bullet list of factual / practice questions — not "should we ship X?" |
| `sourceHints` | Optional | Docs, repos, papers to prioritize |
| `avoidSources` | Optional | Sources to distrust |
| `targetRepo` | Optional | Active repo name, e.g. `kokoro-coreml` |
| `targetGuidePath` | Optional | e.g. `README/Guides/apple-silicon/coreml-ane-scheduling-guide.md` |

Default `agentMode` to **`max`**. Confirm before launching unless the user explicitly asked for fast/cheap (`default`).

If the user’s ask is advice-shaped, **stop and reframe** with them into a domain
synthesis topic, or route away from create-guide. Do not launch Deep Research on
a dial-a-friend prompt.

### 2. Assemble the research prompt

Follow the scratchpad pattern (see `Scratchpad/scratchpad.md` in active repo when present):

- Advanced developer field guide: synthesize public knowledge
- Best practices, worst practices, common bugs, hidden gotchas, known limitations
- Go light on theory, heavy on practical detail with code
- For complex topics: add Context, Primary research goal, Questions to answer, Output format
- Mark speculative / community / "worth trying" ideas separately from
  well-documented behavior — **keep both**; do not demand universal proof
- Explicitly tell the researcher **not** to produce a go/no-go for a specific
  product campaign; produce a reusable reference manual
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
  --target-guide-path README/Guides/apple-silicon/coreml-ane-scheduling-guide.md \
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
  --target-guide-path README/Guides/apple-silicon/coreml-ane-scheduling-guide.md \
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
| Yes | Invoke **`guide-ingest`** on `raw-report.md` |
| No | Normalize markdown, check API claims (Context7 when available), annotate local contradictions, add cross-links per `README/Guides/` conventions |

Ingestion owns: provenance, cleanup, Context7/API corrections, firsthand
contradiction notes, bidirectional links. create-guide does not rewrite the
draft into a campaign verdict.

Leave **Related Documentation** empty during Deep Research; ingestion adds repo cross-links.

## Prompt Writing Guidelines

The prompt must be 100% self-contained. The research agent has no other context than the context that you share with it. It cannot look at your code unless you share your code. It can only search the web (and hit any MCP you wire up).

The more and richer context you give it, the more relevant the guide will be.
Richer context ≠ asking it to decide your roadmap.

## Prompt template (minimal)

```markdown
Create an advanced developer field guide on: [TOPIC]

Synthesize what public docs, papers, issues, and practitioner reports say.
Best practices? Worst practices? Common bugs and issues? Hidden gotchas?
Idiosyncratic design quirks? Known limitations? Recipes people report as
worth trying?

Be extensive. Be comprehensive. Create a reference manual for implementation and debugging.
Go light on theory, heavy on practical detail with code examples.
Do not produce a go/no-go or roadmap for a specific product campaign.
Mark speculation and community heuristics separately from well-documented behavior;
keep both — do not omit "worth trying" material solely because it is unproven.

Context (environment pins only — not a request for advice):
[GROUND TRUTH]

Primary research goal:
[What external knowledge to synthesize]

Questions to answer:
- [Factual / practice Q1]
- [Factual / practice Q2]

Output format:
- Executive summary first
- Do-this / avoid-this tables
- Failure modes and debugging section
- Concrete profiling commands/tools when relevant
- Numbered references
- Mark speculation separately from documented recommendations
```

## Handoff rules

| Situation | Skill |
| --- | --- |
| Raw draft ready | **`guide-ingest`** (if available) |
| Markdown lint only | **`markdown`** |
| Where to put a note / campaign decision | **`write-notes`** |
| Lighter research without Deep Research | **`write-notes`** |
| Advice / second opinion on what to do next | **`second-opinion`** |

## References

- Runner: `/Users/mm/Documents/GitHub/llm-workflows/scripts/run-create-guide-research.mjs`
- Workflow: `create_guide_v1` in `/Users/mm/Documents/GitHub/llm-workflows`
- Deep Research guide: `/Users/mm/Documents/GitHub/llm-workflows/README/guides/gemini/Gemini-Deep-Research-agents-guide.md`
- Ingest skill: [`guide-ingest`](../guide-ingest/SKILL.md)
- Advice skill: [`second-opinion`](../second-opinion/SKILL.md)
