---
name: find-papers
description: >-
  Find and validate research papers for Kokoro CoreML's TTS, vocoder, Core ML,
  Apple Silicon, and on-device deployment work, then ingest each usable paper
  with paper-ingest. Use when the user asks to discover and add relevant
  research papers. Do not use for a general web guide, an implementation plan,
  a literature review, or one already-named paper.
---

# Find Papers

Discover papers remotely; ingest them locally one at a time.
`paper-ingest` owns full-text transcription. Abstracts are not substitutes
for full text.

## Scope the question

Read the relevant current belief in `README/Wiki/`, external research in
`README/Guides/`, and local evidence in `README/Notes/` before launching
research. Write a private brief at
`README/Notes/<slug>-find-papers-brief.md` containing the decision, technical
context, precise questions, source priorities, exclusions, and a maximum paper
count (default 10; hard maximum 30). Do not include credentials, private audio,
or unreleased benchmark outputs in the brief or prompt.

If `README/Papers/` exists, inventory it by title and identifier to avoid
duplicates. If it does not exist, record that. Create the corpus when the first paper
is ingested, not before candidates exist.

If the user already named one paper, use `paper-ingest` directly.

## Run the research workflow

From `/Users/mm/Documents/GitHub/llm-workflows`, run:

```bash
pnpm run research:find-papers -- \
  --topic "<bounded topic>" \
  --context-file /Users/mm/Documents/GitHub/kokoro-coreml/README/Notes/<slug>-find-papers-brief.md \
  --target-repo kokoro-coreml \
  --target-paper-directory README/Papers \
  --max-papers <n> \
  --agent-mode max
```

The workflow uses `find_papers_v1`. Preserve its generated run directory and
`paper-candidates.md`; do not hand-copy an unverified list into repository
documentation. A completed run requires nonempty candidate output and completed
metadata. Resume a failed workflow through its checkpoint rather than launching
an unrelated duplicate run.

## Validate before recommending

For each retained candidate, verify its canonical identifier, title, authors,
year, landing page or official PDF/full HTML, and abstract from a primary
source. Merge a preprint and a published version into one record, preferring the
published version when it contains the relevant method or results. Record why it
matters to the exact TTS/vocoder, Core ML conversion, model architecture, or
Apple hardware decision in the brief.

Exclude papers that only share generic keywords, lack a recoverable primary
source, or cannot support the decision in the brief. Treat reported benchmark
numbers as hypotheses until the relevant model revision, device, OS, runtime,
and measurement method have been reproduced locally.

## Ingest every usable paper

Ingest in two passes: parallel transcription, then one serial linking pass.

**Pass A — transcribe in parallel.** Launch one background subagent per
validated candidate, all in a single message so they run concurrently. Use
**Haiku 5.5** (`claude-haiku-5-5`) for every transcription subagent.
`paper-ingest` owns the transcription. Give each subagent:

- the `paper-ingest` skill path;
- exactly one candidate: title, DOI or arXiv id, and the verified PDF path or
  full-text HTML URL;
- an instruction to write **only** that paper's markdown file, and not to
  edit `README/Papers/README.md`, other papers, guides, notes, or plans;
- a short report back: file path, anything unrecoverable, and 2–3 findings
  relevant to the brief.

Downloads and validation happen before Pass A, in the parent.
One subagent per paper. The parent does not transcribe.

**Pass B — link serially.** After every Pass A subagent has reported, the
parent adds each paper to `README/Papers/README.md` and makes the inbound
and outbound links described by `paper-ingest`, one paper at a time, so
later papers can link to earlier ones.

- Before Pass A, create `ingest-state.json` beside `paper-candidates.md`.
  Record each candidate's title, identifier, selected source, status
  (`pending`, `in_progress`, `ingested`, `duplicate`, `rejected`, or
  `blocked`), and final paper path. The parent owns this file: set
  `in_progress` for every candidate before launching Pass A, and update each
  candidate atomically as its subagent reports.
- Download a verified PDF into `README/Papers/pdfs/` using a unique,
  title-derived filename. Never save generic names such as `main.pdf` or
  overwrite an existing file. Reuse an existing filename only after its hash
  and paper identity match; otherwise append a deterministic DOI/arXiv-derived
  suffix. If no identifier exists or that suffix also collides, append the
  first 12 characters of the verified source SHA-256; always use no-clobber
  creation.
- Invoke `paper-ingest` with the candidate's exact title, DOI/arXiv id, and the
  exact downloaded PDF path. Do not let it select an unspecified PDF from the
  drop folder.
- If no PDF exists, pass only the verified full-text HTML URL—not the canonical
  landing page—to `paper-ingest`. Explicitly bind that URL as the sole ingest
  source and forbid fallback to any PDF already in `README/Papers/pdfs/`.
- Let `paper-ingest` choose or reuse the markdown slug.
- Never transcribe from the candidate abstract alone.

If full text is unavailable, leave that candidate un-ingested and report the
blocker. Do not replace faithful transcription with a summary.

On interruption, resume from the newest `paper-candidates.md` and its
`ingest-state.json`. Do not launch Deep Research again or repeat candidates
already in a terminal state. Reconcile every `in_progress` candidate first:
inspect the expected paper path, source metadata/hash, and reciprocal links. If
complete, mark it `ingested`; otherwise repair or resume that same paper before
advancing. If the final path was not recorded before interruption, locate it by
scanning `Paper metadata` for the candidate DOI, normalized arXiv id, or title;
do not guess a slug.

## Close-out

Report:

- candidate artifact path and workflow mode
- candidates returned, duplicates/rejections, and papers ingested
- final paper paths and source versions
- any paper blocked by unavailable or unverifiable full text
