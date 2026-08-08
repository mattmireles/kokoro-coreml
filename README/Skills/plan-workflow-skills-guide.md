# Plan Workflow Skills Guide

Canonical workflow guide for the repo’s plan-oriented skills:
`create-plan`, `execute-plan`, `execute-plan-hardcore`, and `phase-audit`.

## Purpose

These skills turn a repeated manual pattern into a stable workflow:

- create a real plan from repo knowledge
- execute an approved plan one phase at a time
- audit each completed phase before moving on

Canonical knowledge stays in **`README/`** (guides, plans, notes). This file
is the workflow contract; skills should wrap it, not invent a parallel process.

## Shared Rules

- Keep each skill narrow:
  - `create-plan` writes plans
  - `execute-plan` executes plans
  - `execute-plan-hardcore` runs the same execution loop as `execute-plan`, then
    a full **`audit`** until Architecture, Correctness risk, and Complexity
    debt are all **A** (fix and repeat)
  - `phase-audit` reviews completed phases
- If a runtime cannot support delegated review cleanly, the workflow must still
  work with a **local** audit using [phase-audit-rubric.md](./phase-audit-rubric.md).
- Optional cross-agent review (separate Codex / Claude Code CLI threads) is
  **nice to have**, not required for this repo unless the user asks for it.

## Side-Effect Classes

- `create-plan`: repo-write, no git side effects beyond normal file edits
- `phase-audit`: read-only review
- `execute-plan`: git-write workflow that may commit, sync, push, and monitor
  CI
- `execute-plan-hardcore`: same git-write surface as `execute-plan`, plus
  post-push **`audit`** iterations and fixes until **A / A / A** on the audit rubric (see `.claude/skills/execute-plan-hardcore/SKILL.md`)

## Authority Rule

- Explicit invocation of a workflow skill authorizes the side effects
  documented for that skill (e.g. “use execute-plan”, `$execute-plan`, “use
  execute-plan-hardcore”, `$execute-plan-hardcore`).
- Implicit routing does **not** authorize git writes. If `execute-plan` or
  `execute-plan-hardcore` was not invoked explicitly, prepare local changes but
  stop before commit or push and say why.

## Skill: `create-plan`

### Job

Turn a concrete request into a repo-native implementation plan using
[Templates/Plans-template.md](../Templates/Plans-template.md).

### Research Order

1. Related files under **`README/`** (conversion guides, export notes,
   `README/Plans/` neighbors).
2. **`README/Wiki/`** plus **`README/Guides/`** for PyTorch → Core ML
   constraints (see also `CLAUDE.md` for agent operating rules only).
3. **Context7** only when the plan depends on current `coremltools`, PyTorch, or
   Apple API behavior that may have changed.

### Output Contract

- Use the plans template; include phases, verification, hard requirements, and
  rollback where relevant.
- Name concrete files when the path is knowable.

## Skill: `execute-plan`

### Job

Execute an existing checked-in plan end-to-end, **one phase at a time**.

### Required Loop (per phase)

1. Read the phase and linked guides.
2. Implement only that phase’s scope.
3. Audit the phase (`phase-audit` or local rubric); fix findings before
   proceeding.
4. Update plan checkboxes to match reality.
5. Commit the phase with a clear message (**narrow** staging vs default
   `git-commit` whole-tree—see `execute-plan` skill).

After all phases: sync, push, and monitor CI (**`git-push`**) when the user
wants the branch integrated.

For **`execute-plan-hardcore`**, that loop is **Part A** only; **Part B** is a
mandatory full **`audit`** on the plan-execution scope, then fix and re-audit
until all three rubric grades are **A** (see `.claude/skills/execute-plan-hardcore/SKILL.md`).

### Worktree Rule

- Ignore unrelated dirty files unless they conflict with the active phase.
- Never revert unrelated user work.

## Skill: `execute-plan-hardcore`

### Job

Same as **`execute-plan`**, then require a passing full-repo-style **`audit`**
(three dimensions **A / A / A**) on the execution scope, with authorized fix
iterations until grades hold or the user must resolve a tradeoff.

### When to use

- Explicit invocation only (`execute-plan-hardcore`, `$execute-plan-hardcore`).
- User wants plan execution **and** the audit-to-**A** gate, not **`execute-plan`**
  alone.

## Skill: `phase-audit`

### Job

Review a completed phase like a skeptical senior reviewer before the next
phase or push.

### Review Style

- Findings first, severity ordered, concrete file references.
- Read [phase-audit-rubric.md](./phase-audit-rubric.md) before auditing.

## Delegated Audit Fallback

If forked/delegated review is not available, `execute-plan` must still run the
same rubric locally. **`execute-plan-hardcore` Part B** is separate: a full
**`audit`** on the execution scope (not only `phase-audit`).

## Invocation Policy

Prefer explicit `$create-plan`, `$execute-plan`, `$execute-plan-hardcore`, or
`$phase-audit` when forcing a precise handoff.
