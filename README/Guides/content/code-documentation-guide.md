# Code documentation guide

## Purpose

Standards for **inline** documentation in this repo: docstrings, file headers,
and rationale for constants and tricky tensor shapes—not README or plan files.

## Document what code cannot show

Add or tighten docs only when they capture:

- **Domain knowledge** — Kokoro/Core ML pipeline stages, bucketing, alignment.
- **Non-obvious constraints** — static shapes, dtype, device, sequence limits.
- **Cross-file contracts** — “must stay aligned with `export_*.py` / Swift
  padding” when grep alone is insufficient.
- **State lifecycle** — buffers, caches, export vs runtime.
- **Constant rationale** — why this bucket size or channel count exists.

## Prefer short and durable

- Short module docstrings when the role is not obvious from the path.
- Docstrings that explain **why** or **constraints**, not a rephrasing of the
  signature.
- Avoid manual call graphs and comments that will drift on the next export.

## When the explanation belongs elsewhere

If the real answer is a long operational procedure, put it in **`README/`**
guides and link from a one-line comment in code.

## Related

- `documentation` skill (`.claude/skills/documentation/`; `.cursor/skills` and
  `.agents/skills` symlink to `.claude/skills`)
- [Kokoro-to-CoreML-conversion.md](../apple-silicon/Kokoro-to-CoreML-conversion.md)
