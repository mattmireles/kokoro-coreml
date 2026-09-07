---
name: find-papers
description: >-
  Find and validate research papers for Kokoro CoreML's TTS, vocoder, Core ML,
  Apple Silicon, and on-device deployment work. Use when the user asks for a
  bounded, source-verified paper set or literature discovery pass; do not use
  for a general web guide, an implementation plan, or ingesting one named paper.
---

# Find Papers

Produce a small, decision-relevant candidate set with primary-source evidence.
Discovery output is not a paper corpus and abstracts are not substitutes for
full-text evidence.

## Scope the question

Read the relevant current belief in `README/Wiki/`, external research in
`README/Guides/`, and local evidence in `README/Notes/` before launching
research. Write a private brief at
`README/Notes/<slug>-find-papers-brief.md` containing the decision, technical
context, precise questions, source priorities, exclusions, and a maximum paper
count (default 10; hard maximum 30). Do not include credentials, private audio,
or unreleased benchmark outputs in the brief or prompt.

If `README/Papers/` exists, inventory it by title and identifier to avoid
duplicates. If it does not exist, record that the repository has no paper
corpus; do not create one as a side effect of discovery.

For a single paper named by the user, retrieve and verify its primary source
directly instead of running broad discovery.

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

## Handoff boundary

Return the verified candidate set and the retained workflow artifact. Do not
create `README/Papers/`, reconstruct papers from abstracts, or claim full-text
review as a discovery side effect. When the user asks to import a retained
paper, hand off to `paper-ingest` after primary-source validation.
