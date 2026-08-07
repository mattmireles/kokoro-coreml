---
name: guide-ingest
description: >-
  Ingests offline deep-research exports into README guides for kokoro-coreml: cleans markdown,
  corrects library/API claims with Context7 when docs contradict the draft,
  annotates firsthand repo contradictions without stripping useful “worth
  trying” external synthesis, and adds bidirectional cross-links in
  README/Guides and README/Notes. Use for guide-ingest, research-to-guide, or
  corpus linking of a new guide from Deep Research / browser export.
---

# Guide ingest

## Purpose

Turn raw external research into a **durable repo guide**: a synthetic library
entry that captures what the internet says about a domain (the “afternoon of
Googling” artifact from **create-guide**).

`kokoro-coreml` runs Kokoro TTS on Apple Silicon via Core ML / ANE. Guides stock
external mechanism knowledge (Core ML conversion, ANE scheduling, Metal/MPS,
iSTFT, Swift/Core ML glue). They are not campaign go/no-go decisions.

Ingest must:

1. Match this repository’s markdown conventions
   ([`README/Guides/content/markdown-authoring-guide.md`](../../../README/Guides/content/markdown-authoring-guide.md)).
2. **Correct** documented library/API facts when Context7 (or other primary
   docs) contradict the draft.
3. **Annotate** contradictions from firsthand local knowledge — do not silently
   rewrite the guide into a campaign memo, and do not delete useful external
   synthesis just because it is unproven.
4. Sit correctly in the **bidirectional** web of `README/Guides/` and
   `README/Notes/`.

A guide remains an **external-knowledge artifact**, not a dial-a-friend verdict
for the current plan. Campaign go/no-go, measured latency/quality tables, and “do not launch
arm X” decisions belong in `README/Notes/` (link them); keep the guide reusable
as domain synthesis.

## Use When

- Importing deep-research (or similar) output into `README/Guides/...`.
- The user invokes **guide-ingest**, **research-to-guide**, or the same workflow
  in natural language.
- A new guide needs **outbound** links to notes/guides and **inbound** links from existing docs.

## Do Not Use When

- The task is only a small markdown typo fix (use **`markdown`**).
- The task is only where to put a note (use **`write-notes`**).
- The work is inline code documentation (use **`documentation`**).
- There is no external raw artifact — stop and write a note instead of inventing
  a guide from local reasoning alone.

## Prerequisites

- **Context7 MCP** available for `resolve-library-id` / `query-docs` (or the
  repo’s equivalent Context7 tools). Read the tool schema in the MCP
  descriptors **before** calling tools.
- Raw source: path to `raw-report.md` (or equivalent browser-downloaded export).

## Procedure

### 0. Provenance gate

1. Confirm an external raw source path exists (usually
   `llm-workflows/outputs/create-guide/**/raw-report.md`).
2. Record that path (and SHA-256 when practical) near the top of the ingested
   guide.
3. If there is no raw external artifact, **stop** — use **`write-notes`**, do
   not mint a guide from agent synthesis alone.

### 1. Normalize to proper markdown

1. Read the **`markdown`** skill and the guide index
   [`README/Guides/content/markdown-authoring-guide.md`](../../../README/Guides/content/markdown-authoring-guide.md)
   (and [`README/Templates/guide-template.md`](../../../README/Templates/guide-template.md) when adding a new file)
   for repo tone and structure.
2. Produce guide-shaped markdown:
   - real markdown links, not bare URLs
   - blank lines around headings and lists
   - language-tagged fences where applicable
   - single trailing newline; no unnecessary HTML
   - for messy exports: unescape punctuation, strip data-URI images / broken
     image blocks, and repair Swift/code fences without inventing missing facts
3. Add a short top-of-file purpose blurb if the template or sibling guides use
   one; match the tone of nearby guides in the same folder.
4. Strip or refuse campaign-advice framing from the draft (“you should kill this
   arm”, “no-go for your bakeoff”) when it is clearly advice to *this* repo’s
   plan. Preserve the underlying external mechanism, ratios, failure modes, and
   recipes. Put the campaign decision in a note and link it.

### 2. Correct vs annotate (do not over-prune)

Guides are allowed to include **plausible, unproven, “worth trying”** material
from the public internet. Ingest is not a demand for 100% validation before a
claim may appear.

| Situation | Action |
| --- | --- |
| Context7 / official docs contradict a library API, flag, default, or deprecation | **Correct** the guide to match current docs |
| Obvious hallucination (invented API, fake flag, fabricated citation) | **Remove or rewrite**; note the rejection briefly if useful |
| Community heuristic / paper recipe / “people report X” with no local proof | **Keep**, labeled as community / secondary / worth-trying |
| Firsthand local measurement or pinned-stack run **contradicts** the draft | **Annotate** in-place (correction callout + link to the note/receipt); do **not** delete the external claim without saying why |
| Claim outside Context7 (Core ML / ANE scheduling, Metal, Swift glue) | Prefer primary docs / man pages / source pins when checking; else keep labeled, do not fake “verified” |

Procedure:

1. Extract **verifiable** library/API claims and check them with Context7
   (resolve library ID, then query docs). Prefer current official behavior for
   those factual corrections.
2. Scan `README/Notes/` and related guides for **firsthand** contradictions
   (measured runs, pinned-stack corrections). Add explicit callouts such as
   “Local evidence (link): … contradicts the draft’s claim that …”.
3. Do **not** gut sections solely because they lack a Context7 hit or a local
   A/B test. Unverified ≠ worthless for a synthetic external library.
4. Do **not** rewrite the guide into a qualification-campaign status document
   during ingest. Link status notes; keep the guide domain-scoped.

### 3. Add outbound links (guides and notes)

1. While reading sections, identify concepts that already have a home in
   **`README/Guides/`** or **`README/Notes/`**.
2. Add **inline** markdown links on the first strong mention in each section (or
   a compact “Related Documentation” / “See also” list if inline would clutter).
   Use **repo-relative** paths from the new guide file.
3. Prefer linking to **one canonical** guide per topic rather than duplicating
   long explanations.
4. When local campaign decisions exist, link them as **related notes**, not as
   the guide’s thesis.

### 4. Add inbound links (corpus updates)

Goal: related docs **point back** to the new guide so agents discover it from
both directions.

1. **Discover candidates**: search `README/Guides/` and `README/Notes/` for
   overlapping keywords, library names, and headings (e.g. ripgrep). Include files that already link to adjacent topics.
2. **Edit sparingly**: add a link in “Related”, “See also”, or the most
   relevant paragraph—**minimal** diff, no drive-by rewrites.
3. If a high-level notes file covers the same subsystem, add a short bullet or
   sentence there with a link to the new guide (see **`write-notes`** for
   consolidation habits).
4. Update [`README/Wiki/canonical-source-coverage.md`](../../../README/Wiki/canonical-source-coverage.md)
   (and [`README/Wiki/README.md`](../../../README/Wiki/README.md) when needed) when adding a new guide file.

### 5. Close the loop

1. Rely on the **`markdown`** skill for structure; run **`pytest`** only if the change overlaps Python/runtime code.
2. Give the user a **short summary**:
   - provenance path recorded
   - API/docs corrections (Context7 or primary)
   - firsthand contradiction annotations (and note links)
   - what was kept as unlabeled or labeled “worth trying”
   - inbound link files touched

## Handoff Rules

| Situation | Hand off to |
|-----------|-------------|
| Repo markdown rules, lint, structure | **`markdown`** |
| Campaign decision / measurement write-up | **`write-notes`** |
| Docstrings / inline code documentation | **`documentation`** |
| Need a new external synthesis run | **`create-guide`** |
| Advice / second opinion on what to do next | **`second-opinion`** |

## References

- [Markdown authoring guide](../../../README/Guides/content/markdown-authoring-guide.md)
- [Notes index (`README/Notes/README.md`)](../../../README/Notes/README.md)
- Sibling skills: [`create-guide`](../create-guide/SKILL.md), [`second-opinion`](../second-opinion/SKILL.md)
