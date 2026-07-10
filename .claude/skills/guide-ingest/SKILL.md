---
name: guide-ingest
description: >-
  Ingests offline deep-research exports into README guides: normalizes messy
  exports (escapes, data-URI images, Swift snippets), converts to clean markdown,
  verifies library/API claims with Context7 MCP when needed, adds cross-links
  across README/Guides and README/Notes, and updates related docs to link back.
  Use when adding or refreshing a guide from browser-downloaded research, Deep
  Research output, or similar raw material; when the user says guide-ingest,
  ingest-guide, research-to-guide, or wants corpus cross-linking for a new guide.
---

# Guide ingest

## Purpose

Turn raw, offline research into a **durable repo guide** that matches
[kokoro-coreml markdown conventions](../../../README/Guides/content/markdown-authoring-guide.md),
reflects **current** library and platform facts where Context7 can verify them,
and sits correctly in the **bidirectional** web of `README/Guides/` and
`README/Notes/`.

`README/Guides/` is the shelf for externally created reference manuals: Deep
Research / external-agent output that has already landed as raw source material.
`README/Notes/` is for this repo's own learnings, experiment logs, decisions,
and provenance. Do not invent new guide material from local analysis alone; put
local analysis in `README/Notes/` and link to existing guides when useful.
There is a hard provenance gate: no `raw-report.md` or equivalent externally
generated source artifact means no new `README/Guides/` file. A prompt,
checkpoint, local audit, or Codex-authored synthesis belongs in `README/Notes/`
until the external guide actually lands.

This skill **subsumes** the older mechanical-only `ingest-guide` flow: use
**guide-ingest** for all guide imports here.

## Use When

- Importing deep-research (or similar) output into `README/Guides/...`.
- The user invokes **guide-ingest**, **ingest-guide**, **research-to-guide**, or
  the same workflow in natural language.
- A new guide needs **outbound** links to notes/guides and **inbound** links
  from existing docs.
- The source is a **messy export** (Docs/PDF/chat) with escaped punctuation,
  broken image reference blocks, or placeholder formulas—see **Mechanical
  cleanup** below.

## Do Not Use When

- The task is only a small markdown typo fix (use **`markdown`**).
- The task is only where to put a note (use **`write-notes`**).
- The work is inline code documentation (use **`documentation`**).

## Prerequisites

- **Context7 MCP** available for library/tool docs when verification is in
  scope. Read the tool schema in the MCP descriptors **before** calling tools.
- Raw source: path or pasted content from an externally generated research
  artifact, usually `raw-report.md` under `llm-workflows/outputs/create-guide`.
  If no external source exists, stop and ask whether to run `create-guide`
  instead of creating a guide from scratch.

## Procedure

### 1. Normalize to proper markdown

1. Confirm the source is an externally generated guide/report. Record the source
   path in the guide, preferably the absolute `raw-report.md` path under
   `llm-workflows/outputs/create-guide`. If the content is only repo-local
   investigation, a prompt, a checkpoint, or a local summary, stop and write a
   note, not a guide.
2. Read the **`markdown`** skill and
   [markdown-authoring-guide.md](../../../README/Guides/content/markdown-authoring-guide.md).
3. Produce guide-shaped markdown:
   - real markdown links, not bare URLs
   - blank lines around headings and lists
   - language-tagged fences where applicable
   - single trailing newline; no unnecessary HTML
4. Match the tone and optional top-of-file blurb of sibling guides in the same
   folder (e.g. `README/Guides/apple-silicon/`).

### 2. Mechanical cleanup (exports from Docs, PDF, chat)

Apply when the file has obvious paste/export damage:

1. Remove trailing `[imageN]: <data:image...>` reference definitions; replace
   inline `![][imageN]` with meaningful text (recover numbers/formulas from
   images when practical, or substitute concise prose).
2. Unescape systematically (multi-character sequences first): `\[ \]`, `\!=`,
   `\==`, `\-\>`, `\**`, ` \- `, `\+`, `\>=`, `\<=`, `\>`, `\<`, `\=`, `\#`,
   `\_`, `\(`, `\)`, `\.`, `` \` ``, `\!`, then remaining `\[` / `\]` for
   citations where needed.
3. **Swift:** replace `\\(` with a placeholder **before** applying `\(` → `(`,
   then restore to `\(` (string interpolation).
4. After the main pass: `-\>` → `->`, then `\-` → `-` (CLI flags, LLDB `-n`,
   ObjC `-[...]`).
5. Fix obviously broken snippets (e.g. `os.environ = "1"` →
   `os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"` when that is what the
   prose means).
6. `grep` the file for `\\` after edits; only intentional escapes (e.g. Swift
   `\(`) should remain.

### 3. Verify claims with Context7 (fix stale deep-research)

1. Extract **verifiable** claims: library APIs, CLI flags, framework behavior,
   config keys, deprecation notices, default versions, and similar **documented**
   facts (coremltools, PyTorch MPS, Core ML, Xcode, etc.).
2. For each claim, use Context7: resolve the library/tool ID, then pull the
   relevant docs. Prefer **current** official behavior over the research draft.
3. **Update the guide** when Context7 contradicts the export: correct API
   names, replace deprecated patterns, fix version-specific statements.
4. For claims **outside** Context7’s scope (product strategy, subjective
   opinion, internal project behavior not in public docs), do **not** fake
   verification. Either leave as non-factual narrative, mark as
   opinion/heuristic, or point to a repo note/guide that is authoritative.

### 4. Add outbound links (guides and notes)

1. While reading sections, identify concepts that already have a home in
   **`README/Guides/`** or **`README/Notes/`**.
2. Add **inline** markdown links on the first strong mention in each section (or
   a compact “Related Documentation” / “See also” list). Use **repo-relative**
   paths from the new guide file.
3. Prefer linking to **one canonical** guide per topic rather than duplicating
   long explanations.

### 5. Add inbound links (corpus updates)

Goal: related docs **point back** to the new guide so agents discover it from
both directions.

1. **Discover candidates**: search `README/Guides/` and `README/Notes/` for
   overlapping keywords, product names, library names, and headings (e.g.
   ripgrep). Include files that already link to adjacent topics.
2. **Edit sparingly**: add a link in “Related”, “See also”, or the most
   relevant paragraph—**minimal** diff, no drive-by rewrites.
3. If a high-level notes file covers the same subsystem, add a short bullet or
   sentence there with a link to the new guide (see **`write-notes`** for
   consolidation habits). Notes may also record ingest provenance, verification
   decisions, rejected claims, and experiment outcomes derived from the guide.

### 6. Close the loop

1. Run the repo’s primary checks when you touched more than markdown in passing
   (root **`pytest`** is the main gate here); run markdown lint only if the repo
   defines a command for it (`markdown` skill).
2. Give the user a **short summary**: mechanical fixes applied, what was
   corrected via Context7, which files gained inbound links, and any claims left
   unverified.

## Handoff Rules

| Situation | Hand off to |
| --- | --- |
| Repo markdown rules, lint, structure | **`markdown`** |
| Where a note should live, note sprawl | **`write-notes`** |
| JSDoc / Python docstrings in code | **`documentation`** |

## References

- [Markdown authoring guide](../../../README/Guides/content/markdown-authoring-guide.md)
- [Notes consolidation guide](../../../README/Guides/content/notes-consolidation-guide.md)
- `markdown` skill: [references/index.md](../markdown/references/index.md)
