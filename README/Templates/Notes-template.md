# [Domain] Debug Notes

Institutional memory for [domain]-related bugs. Multiple related issues should
usually live in one high-level domain note file, with each issue kept
self-contained.

**Quick filter:** `grep -n "— Active" README/Notes/[filename].md`

---

## Issue: [Short Issue Name] — Active

**First spotted:** YYYY-MM-DD
**Status:** Active

### Summary

What broke and what fixed it (2-3 sentences max). For active issues, describe
current state and next step.

### Symptom

```log
[Actual error or log output]
```

### Root Cause

Why it broke (1-2 sentences). Write "TBD. Not manually confirmed." until verified.

### Related Guides

Links to relevant guides that provide context or solutions:

- [Guide Name](README/Guides/path/to/guide.md) - Brief reason why it's relevant
- [Another Guide](README/Guides/path/to/another-guide.md) - Brief reason why
  it's relevant

### Fix

**Files:**

- `path/to/module.py:123`
- `path/to/other_module.py:456`

```python
# Before
broken_code()

# After
fixed_code()
```

### Verification

```bash
pytest path/to/test_module.py
```

### Investigation Log

**YYYY-MM-DD**

- **Hypothesis:** What we thought was wrong.
- **Tried:** What we did to test it.
- **Outcome:** What we learned. Did it work? What's next?

**YYYY-MM-DD**

- **Hypothesis:** Next theory.
- **Tried:** Next experiment.
- **Outcome:** Result.

### If This Recurs

- [ ] Check [specific thing]
- [ ] Verify [specific condition]

```bash
# Debug command
grep -r "pattern" path/
```

---

## Issue: [Another Issue Name] — Resolved

**First spotted:** YYYY-MM-DD
**Resolved:** YYYY-MM-DD
**Status:** Resolved

### Summary

What broke and what fixed it.

### Symptom

```log
[Error output]
```

### Root Cause

Why it broke (confirmed).

### Related Guides

Links to relevant guides that provide context or solutions:

- [Guide Name](README/Guides/path/to/guide.md) - Brief reason why it's relevant
- [Another Guide](README/Guides/path/to/another-guide.md) - Brief reason why it's relevant

### Fix

**File:** `path/to/module.py:123`

```python
# The fix
```

### Verification

```bash
# Command to verify fix
```

---

## Required Executable Memory

Every resolved issue must include one of these lines:

- Regression test: `[command that proves the fix still works]`
- Not testable: `[why no command can prove it, plus strongest manual proof]`

<!--
USAGE NOTES:

1. Prefer updating an existing high-level domain note file before creating a new
   file. Use a new file only when the topic is durable enough to deserve its own
   entry point.

2. New issue? Copy the "Active" template above and paste at the top (newest
   first).

3. Fixed an issue? Change "— Active" to "— Resolved" and add "Resolved:" date.

4. Investigation Log: Delete ruled-out hypotheses after resolution to keep it
   clean.

5. Optional sections: "If This Recurs" is optional. Skip it for simple bugs.

6. Multiple files? Use bullet list under **Files:**. Single file? Use **File:**
   inline.

7. Quick scan: `grep -n "— Active"` shows all open issues with line numbers.
-->
