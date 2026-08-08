# kokoro-coreml reading list (persona + export work)

Paths are **repo-root relative** (open from the `kokoro-coreml` checkout root).

## Canonical persona and playbook

- `CLAUDE.md` — agent operating rules (Simpler Is Better, Algorithm, memory routing)
- `README/Wiki/README.md` — current belief / Core ML export routing
- `README/Guides/` — PyTorch → Core ML field manuals (ANE layout, scheduling, convert)

## Conversion and export

- `README/Guides/apple-silicon/Kokoro-to-CoreML-conversion.md`
- `README/coreml-conversion-guide.md`
- `README/learnings.md`
- `README/problem-summary.md`

## Apple silicon runtime and performance

- `README/Guides/apple-silicon/CoreML-Compute-Unit-Scheduling-guide.md`
- `README/Guides/apple-silicon/pytorch-mps.md`
- `README/Guides/apple-silicon/HF-transformers-MPS-guide.md`
- `README/Notes/performance-notes.md`
- `README/Plans/002-ane-optimization-plan.md`

## Experiments and comparisons

- `README/Plans/003-kokoro-bakeoff-plan.md`
- Other plans under `README/Plans/` as needed

## Repo process (plan-driven work)

- `README/Skills/plan-workflow-skills-guide.md`
- `README/Skills/phase-audit-rubric.md`

## Related skills (narrower charters)

- `.claude/skills/audit/SKILL.md` — findings-first review; word **audit**
- `.claude/skills/debug/SKILL.md` — systematic defect investigation; word **debug**
- `.claude/skills/execute-plan/SKILL.md` — phased plan execution
