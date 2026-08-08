---
name: debug
description: >-
  Systematic debugging for kokoro-coreml: consult README/ (guides and
  learnings) and CLAUDE.md first, pull current library docs via Context7 MCP
  when coremltools/PyTorch/API behavior is uncertain, parallelize investigation
  with multiple subagents when stuck, prove fixes before calling success, then
  capture one consolidated note in README/Notes via **write-notes** as the
  **final** step before ending the session. For extreme cases, delegate a
  multi-agent audit via the Claude Code CLI. Use when the user invokes **debug**,
  **use debug**, asks to debug or fix a tricky bug, or work is blocked by unclear
  failure modes after quick local checks.
---

# Debug

## Purpose

Reduce guesswork: **README/ and CLAUDE.md first**, **Context7 when library
behavior is uncertain**, **parallel hypotheses when stuck**, **proof before
“fixed”**, **one write-notes pass at session end**, **CLI multi-agent audit**
only for exceptional escalation.

## Use When

- The user says **debug**, **use the debug skill**, **debug this**, or the task
  is **bug investigation** / **root-cause** analysis that is stalling.
- Failures involve **coremltools**, **PyTorch**, **Core ML runtime**, or **Swift
  integration** where training data may be stale—use Context7 before assuming
  APIs.

## Do Not Use When

- The user only wants a **trivial fix** with an obvious stack trace and no doc
  ambiguity (still skim `README/` if the area is known to be finicky).
- The task is **greenfield feature build** with no defect.

## Workflow (in order)

### 1. Read repo knowledge first (mandatory)

Before writing code or deep-diving the stack:

1. **Skim `README/`** for files that match the subsystem (conversion, export,
   Core ML runtime, tokenizer bridge, `kokoro.js`, etc.). Open the most relevant
   guides and look for **known issues**, **workarounds**, and **checklists**.
2. **Skim `README/Notes/`** (when present) and high-level docs linked from
   `README.md` for past investigations.
3. Read **`README/Wiki/`** and the relevant **`README/Guides/`** for PyTorch →
   Core ML playbook constraints (static shapes, ANE layout, divide-and-conquer).
   **`CLAUDE.md`** is agent operating rules only.

### 2. Context7 MCP (library and platform docs)

When the bug touches **coremltools**, **PyTorch**, **Swift Core ML**, or related
APIs:

1. Use **Context7**: resolve the library ID, then fetch focused docs for the API
   or behavior in question.
2. Prefer Context7 over memory or generic web search **for API shape and
   version-sensitive behavior**.

If Context7 is unavailable, fall back to official docs—**do not** invent APIs.

### 3. Reproduce and narrow

- Confirm **minimal repro** or the exact **export step / predict call / log
  line** that fails.
- State a **one-sentence hypothesis** and what evidence would falsify it.

### 4. Parallel investigation when stuck

If the problem is **hard**, **cross-cutting**, or you have been **going in
circles** after the reproduce-and-narrow pass:

- In **one turn**, spin up **multiple** `Task` subagents with **`readonly: true`**
  and **narrow charters** (e.g. “MIL/export path only”, “Swift shapes only”).
- **Merge** overlapping hypotheses; avoid duplicate deep reads of the same file.

**When in doubt, parallelize** (same posture as **`audit`**, but
hypothesis-driven).

### 5. Exceptionally frustrating bugs — CLI + audit

When **still stuck** after parallel investigation, or blast radius is **high**:

1. Use the **Claude Code CLI** from the repo root to run a **multi-agent `audit`**
   over the **scoped slice** that matters.
2. In the handoff, cite relevant **`README/`** paths, **symptom + ruled-out
   items**, and instruct use of the **`audit`** skill with readonly subagents.

Escalation only—**after** README, Context7, and a fair parallel pass.

### 6. Prove the fix (before claiming success)

Do **not** say **fixed** until there is **objective proof**: failing test passes,
export completes, Core ML predict matches tolerance, bad log line absent—whatever
matches the bug.

### 7. Write notes once (mandatory last step)

**After** investigation, perform **a single** `README/Notes/` update using
[**`write-notes`**](../write-notes/SKILL.md) (consolidate the session).

- One pass: symptom, repro, ruled out, root cause (`TBD` if unknown), fix,
  **verification** (proven / not), pointers to guides.
- **Skip** only when the user explicitly wants no note or the outcome is trivial.

## Output expectations

- **What was read** from `README/` / `CLAUDE.md` (paths).
- **Verification:** what proved the fix (or **unverified** / **blocked**).
- **Which note file(s)** were updated in step 7 (or why skipped).
- **Whether Context7** was used and for which topic.
- **Leading hypothesis(es)** if still open, else **root cause** summary.

## Anti-patterns

- Skipping **`README/`** / **`CLAUDE.md`** to “save time.”
- Assuming API behavior without **Context7 or official docs** when versions
  matter.
- **Serial** thrashing instead of **parallel** charters.
- **Declaring victory** without **verification**.
- **Patching notes throughout** instead of **one** end-of-session pass.

## Relation to other skills

- **`audit`:** structured review; use via CLI escalation when debug maxes out.
- **`write-notes`:** once at end of debug session.
- **`phase-audit`:** plan-phase checks—not the same as production debugging.
