---
name: audit-fix-loop
description: >-
  Runs the full multi-agent audit workflow on the user's exact task scope,
  fixes every in-scope issue that blocks Architecture, Correctness risk, or
  Complexity debt from grade A, re-audits that same scope until all three are
  A, then commits only loop-owned changes via git-commit. Use ONLY when the
  user explicitly invokes audit-fix-loop, $audit-fix-loop, or "use
  audit-fix-loop"—not for standalone audits, plan execution, whole-repo cleanup
  inferred from a narrow task, or implicit routing. Does not replace
  phase-audit or deploy gates.
---

# Audit–Fix Loop

## Purpose

**Audit the task → fix its findings → audit the same task again** until the
**`audit`** rubric shows **A** for **Architecture**, **Correctness risk**, and
**Complexity debt** within that task scope, then make **one final
`git-commit`** containing only the loop's changes.

This is not a license to repair the whole repository. A narrow user task stays
narrow even when repo-wide checks expose unrelated debt.

## Use When

- The user **explicitly** invokes **`audit-fix-loop`**, **`$audit-fix-loop`**,
  or clearly asks to **use audit-fix-loop** (same intent boundary as
  **`execute-plan-hardcore`**—no implicit routing).

## Do Not Use When

- The user only asked for a **read-only audit**—use **`audit`**.
- The user is **executing a checked-in plan** and wants the hardcore gate after
  plan work—prefer **`execute-plan-hardcore`**.
- A **single phase** needs plan rubric review—use **`phase-audit`**.
- Git writes are forbidden and nothing overrides that.

## Authority Model

Explicit invocation of **`audit-fix-loop`** authorizes:

- Running the full **`audit`** procedure **as part of this skill** even if the
  current message does **not** contain the substring **`audit`**—this skill is
  the explicit request for **audit and implementation fixes** until grades reach
  **A** (overrides **`audit`**'s default "surface only unless asked" rule for
  this workflow only).
- **Multiple** audit cycles with **multi-agent** depth per **`audit`** (parallel
  `Task` charters when the harness supports it; sequential charters when not).
- **`git-commit`** once **after** **A / A / A** is achieved, following
  **`git-commit`** message rules but overriding its whole-tree staging default:
  stage only files created or changed by this audit-fix loop for the authorized
  task.

If **`audit-fix-loop`** was only inferred from vague wording—**stop** before
fixes or commit and confirm intent.

## Procedure

1. **Freeze the task scope before auditing:**
   - Translate the user's task into concrete flows, paths, or a git-defined
     diff. Whole-repo scope is allowed only when the user explicitly requested
     a whole-repo audit-fix loop.
   - Record the starting `git status` and diff so pre-existing user/agent
     changes are distinguishable from loop-owned changes.
   - Name relevant tests, contracts, guides, and notes that constrain the task.
   - State the frozen scope in every audit report header.
   - If investigation reveals a separate defect outside that scope, report it
     as an out-of-scope finding; do not fix it or silently expand the task.

2. **Audit pass (read + grade):** Follow **`audit`** end-to-end:
   - Mechanical signals from repo root: **`pytest`** (or `python -m pytest`);
     lint only when the repo defines a standard lint command—otherwise skip and
     note “no configured lint.” When the audited surface includes `kokoro.js/`,
     add targeted checks there (e.g. `npm test`). Repo-wide failures outside the
     frozen task are baseline/out-of-scope evidence, not authorization to repair
     unrelated files.
   - Delegate **multiple readonly** charters when depth warrants it (**when in
     doubt, parallelize** per **`audit`**).
   - Merge, dedupe, assign **P0–P3** severities, and **Architecture** /
     **Correctness risk** / **Complexity debt** grades (**A–F**).
   - Grade the task and its changed surface—not unrelated pre-existing repo
     debt.

3. **Pass condition:** all three grades are **A**. If mechanical checks fail,
   treat failures as **P0** and fix before accepting any **A**.

4. **If any grade is below A:** fix every **in-scope** issue that blocks **A**
   (prioritize **P0**/**P1** and task-related mechanical failures). Re-run
   targeted checks; iterate only on task-owned code until ready for a fresh
   audit. Record unrelated findings without changing them.

5. **Loop:** Return to step **2** with a complete audit of the **same frozen
   task scope** (not only a spot-check) until **A / A / A**. Include any
   additional files the loop itself changed to implement the task, but do not
   absorb unrelated defects merely because they were discovered.

6. **Stuck loop:** If **A / A / A** requires an out-of-scope change, a product
   or architecture tradeoff, or modifying pre-existing user/agent work, stop,
   report grades and the exact boundary, and ask the user whether to expand
   scope—do not spin or expand it yourself.

7. **Final commit:** When step **5** passes, run **`git-commit`** once for the
   files produced by this loop. Use **`git-commit`** for subject/body quality,
   but explicitly override whole-tree staging. Never include unrelated dirty
   files, pre-existing user edits, or another agent's work merely because they
   are present.

## Boundaries

- Do **not** grade-inflate; **`audit`** rubric applies.
- Do **not** lower thresholds to reach **A**—raise quality.
- Do **not** audit or fix the whole repository when the user authorized a
  narrower task.
- Do **not** treat unrelated repo-wide check failures as task failures. Report
  them separately with their pre-existing/out-of-scope status.
- Do **not** revert, rewrite, stage, or commit pre-existing user/agent changes
  outside the frozen task scope.
- **`git-commit`** runs **after** grades are **A / A / A**, not after the first
  audit pass unless the first pass already meets **A** (still follow
  **`git-commit`** for recording).

## Relation to Other Skills

- **`audit`:** Defines the audit procedure, charters, mechanical checks, and
  grading; **`audit-fix-loop`** adds **mandatory fix iterations** and a **final
  commit**.
- **`execute-plan-hardcore`:** Plan execution **plus** audit-to-**A** gate;
  **`audit-fix-loop`** is **only** the audit–fix–commit loop (no plan phases).
- **`git-commit`:** Final step; post-commit behavior in **`git-commit`** is
  **not** a substitute for step **2** above.
