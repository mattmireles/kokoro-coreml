---
name: paper-ingest
description: >-
  Ingest one academic paper into Kokoro CoreML's README/Papers as a faithful
  markdown transcription with provenance and bidirectional links to relevant
  repository material. Use when the user asks to ingest or import a named
  paper, provides a PDF, or asks to repair a raw extraction. Do not use for
  summaries, literature discovery, or a passing citation.
---

# Paper Ingest

Turn one academic paper into a durable local record. This is faithful
transcription, not a recommendation or a rewrite of the authors' claims.

Papers are external source material. Local device measurements, conversion
results, and decisions remain in `README/Notes/`; external synthesis remains
in `README/Guides/` through `create-guide` and `guide-ingest`.

## Source and identity

Use the caller-bound full-text source first: a named PDF, official full-text
URL, DOI, arXiv identifier, or a file in `README/Papers/pdfs/`. Do not invent
a paper from memory. If the bound source cannot yield recoverable full text,
stop and report the gap; do not substitute an unrelated PDF.

Record the exact title, authors, venue/status, date, DOI and/or arXiv version,
canonical URL, and actual ingest source. Hash a local PDF with `shasum -a 256`.
Do not put credentials, private audio, or unreleased benchmark outputs in the
paper record.

The same work is identified by DOI, arXiv identifier ignoring its version
suffix, or normalized title. Replace an existing matching record rather than
minting a duplicate. PDFs belong only in `README/Papers/pdfs/`; the markdown
record belongs at `README/Papers/<slug>.md`.

## Faithful transcription

Repair layout, not meaning: restore reading order, headings, hyphenation,
lists, footnotes, citations, table cells, equation markup, captions, and one
reference per entry. Keep the abstract, body, tables, equations, figure/table
captions, references, and any recoverable appendices. Do not paraphrase or
silently correct the authors. If science-critical sections cannot be recovered,
stop and say which ones are missing.

Use this file shape:

```markdown
# Exact paper title

> **Paper metadata**
>
> - **Authors:** …
> - **Venue/status:** …
> - **Published:** …
> - **Identifier:** DOI and/or arXiv id with version
> - **Canonical source:** [label](url)
> - **Ingest source:** exact path or full-text URL used
> - **Source SHA-256:** hex, when a PDF was used
> - **Ingested:** YYYY-MM-DD

## Repository context

Links to existing local receipts or contradictions only. Omit when none.

## Related repository material

- **Papers:** [title](./other-paper.md) — why
- **Guides:** [title](../Guides/…md) — why
- **Notes:** [title](../Notes/…md) — why
- **Plans:** [title](../Plans/…md) — why

---

## Abstract

Faithful paper text begins here.
```

Omit empty related categories. All repository commentary stays above the rule;
do not alter the paper body or bibliography for local discussion.

## Link the evidence

An ingest is incomplete until it has both directions of a meaningful link:

- Outbound: the paper's `Related repository material` identifies a document
  sharing a concrete method, model, result, plan, evaluation, receipt, or
  specification.
- Inbound: that local document links back to the paper with one clause saying
  why it is relevant.

Search existing papers, guides, notes, plans, evaluations, and evidence by
title, author, identifier, and distinctive terms. Do not link documents merely
because both mention TTS, Core ML, or Apple Silicon.

On the first ingest, create `README/Papers/README.md` exactly as this minimal
index, then add one row per paper with the reason it is present:

```markdown
# Papers

Faithful transcriptions of individual academic papers. These are not guides or
notes. Ingest with `paper-ingest`; source PDFs live in [`./pdfs/`](./pdfs/).

| Paper | Why it is here |
| --- | --- |
```

Add links to an existing `README/Guides/README.md` or
`README/Notes/README.md` only when those indexes exist. Do not create the paper
directory merely for discovery; `find-papers` owns discovery and invokes this
skill only for an explicit import.

## Close-out

Report the paper path, verified source/version, source hash when applicable,
science-critical material that could not be recovered, and every inbound and
outbound link added. Use `markdown` only for metadata, indexes, and related
blocks; use `write-notes` for new local evidence rather than putting it in the
paper transcription.
