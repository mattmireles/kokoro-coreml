# [Feature Name] Plan

**Date:** YYYY-MM-DD
**Status:** Planned | In-Progress | Complete

> This template is the single source of truth for plans. Use the core sections below, and pull from Modules when needed. Delete anything unused.

## Executive Summary

One to three sentences: what we are building and why it matters. Include before and after metrics if applicable.

## Problem Statement

- **Symptom:** What users or developers experience.
- **Root Cause:** Why this keeps happening.
- **Impact:** Cost of inaction (errors, latency, dollars, time).

## Mode Definitions

Only include modes that change behavior or routing.

| Mode | Behavior | Why it matters |
| --- | --- | --- |
| [Mode] | [What happens] | [Why this changes the plan] |

## Goals and Non-Goals

### Goals

- [ ] [Specific, measurable outcome]
- [ ] [Specific, measurable outcome]

### Non-Goals

- [Deliberate exclusion and why]
- [Deliberate exclusion and why]

## Scope and Constraints

- **Scope:** What this plan includes.
- **Constraints:** Hard limits or dependencies.
- **Guardrails:** What must stay stable.

## Ground Truth Contracts (Do Not Violate)

- **[Contract]:** [Why this matters and what breaks if violated]
- **[Contract]:** [Why this matters and what breaks if violated]

## Already Shipped (Do Not Re-Solve)

- **[Component]:** [What it does, where it lives]
- **[Component]:** [What it does, where it lives]

## Fresh Baseline (Current State)

What exists today, with concrete data. Use this to anchor the plan.

- **Architecture:** [Short summary]
- **Metrics:** [Key numbers, p50/p95 if relevant]
- **Known gaps:** [What is missing]

## Solution Overview

High-level approach before diving into phases. Include a simple diagram if helpful.

```
+-----------+     +-----------+     +-----------+
| Component | --> | Component | --> | Component |
+-----------+     +-----------+     +-----------+
```

## Implementation Phases

> Do one phase at a time. Verify before proceeding.

### Phase 0: Prerequisites

**Goal:** [What this phase accomplishes]

**Tasks:**

- [ ] Task with specific file: `path/to/file.ts`
- [ ] Task with specific file: `path/to/file.ts`

**Verification:** [How to know this phase is complete]

---

### Phase 1: [Phase Name]

**Goal:** [What this phase accomplishes]

**Tasks:**

- [ ] Task with specific file: `path/to/file.ts`
- [ ] Task with specific file: `path/to/file.ts`

**Verification:** [How to know this phase is complete]

---

### Phase 2: [Phase Name]

**Goal:** [What this phase accomplishes]

**Tasks:**

- [ ] Task with specific file: `path/to/file.ts`
- [ ] Task with specific file: `path/to/file.ts`

**Verification:** [How to know this phase is complete]

---

### Phase 3: Validation and Cleanup

**Goal:** Verify end-to-end behavior and remove temporary scaffolding.

**Tasks:**

- [ ] [Validation task]
- [ ] [Cleanup task]

**Verification:** [How to know this phase is complete]

## Executable Memory

- Regression test: `[command that proves the core contract]`
- Not testable: `[only use when no command can prove it; explain the manual proof]`

## Success Criteria

### Hard Requirements (Must Pass)

- [ ] [Specific, testable requirement]
- [ ] [Specific, testable requirement]

### Definition of Done

- [ ] All tests passing
- [ ] Code reviewed and merged
- [ ] Deployed to production
- [ ] Monitoring confirms success

## Open Questions

### Resolved

- **Q:** [Question that was answered]
- **A:** [Decision made and rationale]

### Unresolved

- **Q:** [Question still pending]
- **Options:** [A, B, C] and current lean

## References

### Internal

- [Guide Name](../Guides/guide-name.md)
- [Related Plan](../Plans/related-plan.md)

### External

- [API Documentation](https://example.com/docs)

## Modules

Use only what you need. Delete unused modules.

### User Flow

1. User does X.
2. System responds with Y.
3. User sees Z.

### Data Model and Schema Changes

```typescript
interface NewOrModifiedType {
  field: string;        // Description of field
  anotherField: number; // Description of field
}
```

**Storage Keys:**

- `storage.key.name` - What it stores

**Database Changes:**

- Add field `X` to collection `Y`
- Add index on `[field1, field2]`

### API Contracts

**Endpoints:**

- `POST /api/feature/action` - [What it does]
- `GET /api/feature/status` - [What it returns]

**Request:**

```json
{
  "field": "value"
}
```

**Response:**

```json
{
  "status": "ok"
}
```

### Performance and Latency Budget

| Operation | p50 Target | p95 Target | Current |
| --- | --- | --- | --- |
| [Action] | <X ms | <Y ms | Z ms |

### Error Handling and Edge Cases

| Scenario | Behavior | Fallback |
| --- | --- | --- |
| [Error case] | [What happens] | [Recovery action] |
| [Edge case] | [What happens] | [Recovery action] |

### Degradation and Rollback

**Degradation Modes:**

- **If [X fails]:** System does [Y] instead.
- **If [X fails]:** System does [Y] instead.

**Rollback Plan:**

- **How to revert:** `git revert [hash]` or feature flag `X`
- **Time to rollback:** [Estimate]
- **Data recovery needed:** Yes or No, with steps if yes

### Rollout and Gates

- **Feature flag:** [Flag name and default state]
- **Rollout strategy:** [Canary, staged, internal-only, full]
- **Kill switch:** [How to disable quickly]

### Monitoring and Observability

**Metrics to Track:**

- `metric_name` - What it measures
- `metric_name` - What it measures

**Alerts:**

- Alert on [condition] - indicates [problem]

**Dashboards:**

- [Dashboard Name](https://example.com/dashboard)

### Testing and Validation

**Unit Tests:**

- [ ] Test case description
- [ ] Test case description

**Integration Tests:**

- [ ] Test case description
- [ ] Test case description

**Manual QA:**

- [ ] Manual test scenario
- [ ] Manual test scenario

### Migration and Backfill

**Steps:**

- [ ] [Migration step]
- [ ] [Backfill step]

**Verification:**

- [ ] [How to verify the migration]

**Rollback:**

- [ ] [How to revert safely]

### Phase Dependencies

```
Phase 0 --> Phase 1 --> Phase 2
               |
               +--> Phase 3 (parallel OK)
```

### Files Likely to Change

| File | Change Type | Notes |
| --- | --- | --- |
| `path/to/file.ts` | Modify | [What changes] |
| `path/to/file.ts` | Create | [What it does] |
| `path/to/file.ts` | Delete | [Why removing] |

### Risks and Mitigations

- **[Risk]:** [Impact] -> [Mitigation]
- **[Risk]:** [Impact] -> [Mitigation]

### Progress Tracker

#### Phase 0: Prerequisites

- [ ] Task 1
- [ ] Task 2

#### Phase 1: [Name]

- [ ] Task 1
- [ ] Task 2

#### Phase 2: [Name]

- [ ] Task 1
- [ ] Task 2

#### Phase 3: Validation

- [ ] Task 1
- [ ] Task 2

### Debug Notes

Append real issues encountered during implementation with fixes.

#### [Date] - [Issue Title]

**Problem:** [What went wrong]
**Root Cause:** [Why it happened]
**Fix:** [How it was resolved]
**Files:** `path/to/file.ts:123`

---

## Critical Reminder

> SIMPLER IS BETTER. If you are adding complexity, justify it. Most of the time, the simplest solution wins.
