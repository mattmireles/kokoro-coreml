---
name: second-opinion
description: >-
  Gets two independent, read-only opinions on a concrete technical decision,
  diagnosis, plan, or tradeoff from GPT-5.6 Sol at xhigh reasoning and Claude
  Fable 5 at max effort, then reconciles agreement and disagreement without
  treating model consensus as proof. Use when the user asks for a second
  opinion, says dial-a-friend, wants outside advice, or explicitly asks what
  Fable and GPT think. Do not use for broad internet synthesis (create-guide),
  implementation, or ordinary questions answerable directly.
---

# Second Opinion

## Purpose

Ask two strong models for **independent advice on one concrete matter**:

- GPT-5.6 Sol with `xhigh` reasoning
- Claude Fable 5 with `max` effort

This is the dial-a-friend workflow. It answers “Given these facts, what would
you do and why?” It does not create a durable internet-research artifact; use
**`create-guide`** for that.

`kokoro-coreml` runs Kokoro-82M TTS on Apple Silicon via Core ML / ANE. Second opinions usually concern stage placement (ANE vs GPU vs CPU), Core ML conversion traps, latency/quality tradeoffs, or Swift pipeline design — not inventing trainer WER claims for other repos.

## Invocation and authority

Run only when the user explicitly asks for a second opinion, dial-a-friend,
outside advice, both named models, or equivalent. Explicit natural-language
requests count; the literal skill name is not required.

Invocation authorizes two read-only external model calls and local artifacts
under `tmp/second-opinions/`. It does not authorize code edits, commits, pushes,
or other mutations.

## Procedure

### 1. Frame a rich investigation brief

The brief is an **entry point**, not a substitute for repo research. Sparse
context produces generic advice and anchors both reviewers to the parent
agent's possibly stale framing.

Write a detailed markdown brief containing:

1. **North-star outcome:** the higher-level system or product result we are
   actually trying to achieve, why it matters, and how success is measured.
   Include non-negotiable quality constraints. Example: “minimize end-to-end TTS latency on M-series without degrading audio quality by choosing which Kokoro stages run on ANE, GPU, or CPU.”
2. **Tactical question:** the exact decision, diagnosis, or tradeoff currently
   under consideration.
3. **Why this tactic might serve the north star:** the hypothesized causal link
   and why this is believed to be the current bottleneck.
4. **Current understanding:** measured evidence, assumptions, unresolved
   contradictions, and why the question is difficult.
5. **Constraints:** fixed policy, budget, compatibility, safety, and non-goals.
   Never include live credentials, `.env` secrets, or customer scrape payloads
   in the brief.
6. **Options considered:** including rejected paths and why they were rejected.
7. **Repo starting points:** relevant code, tests, plans, notes, guides, notes, plans, receipts, and recent diffs or commits.
8. **Requested output:** multiple potential directions, each direction's
   steelman and strongest counter-argument, a final recommendation, risks,
   confidence, and the evidence that would reverse the recommendation.

Do not hide inconvenient facts. Distinguish measurements from assumptions.
Name starting paths, but **do not curate the evidence boundary so narrowly that
reviewers cannot discover newer or contradictory repo state**.

Explicitly invite both reviewers to:

- restate the north-star outcome before evaluating the tactical question
- test whether the tactical question attacks the real bottleneck or merely
  optimizes a local component
- propose a different lever, decomposition, or architecture when it would
  reach the north star more directly
- orient themselves in the repository before answering
- inspect all code, tests, plans, notes, guides, notes, plans, and receipts relevant to the question—not only files linked in the
  brief
- search for newer evidence that supersedes the brief
- reconcile documentation against implementation and firsthand measurements
- use git status, diffs, and recent history when they affect the answer
- report which repo evidence materially drove the recommendation
- respect secrets: do not quote credentials or raw customer payloads

The agent preparing the brief should inspect the repo first and provide strong
starting points. The delegated reviewers must still perform their own repo
investigation; do not ask them to reason only from the summary.

Save the brief under `tmp/second-opinions/`. Do not place it in
`README/Guides/` or `README/Notes/` unless the user separately asks for a
durable document.

### 2. Assign an expert persona

Give both reviewers the same domain-appropriate persona so the comparison tests
reasoning, not role ambiguity.

For ML, Core ML, ANE, TTS, evaluation, or Apple Silicon questions, default to
the repo's **Ilya Sutskever** persona:

- [persona skill](../ilya-sutskever/SKILL.md)
- [persona source](../ilya-sutskever/ilya-sutskever.md)

Use the persona's actual text—not merely “act like Ilya.” The useful stance is
learning-first and empirical: respect the Bitter Lesson, prefer simple
objectives and decisive experiments, treat data and scale as first-class, and
avoid false certainty.

For other non-ML questions, choose another existing repo persona only when it
genuinely fits. Otherwise use a neutral skeptical principal-engineer persona.
Pass a non-default persona with:

```bash
SECOND_OPINION_PERSONA_FILE=<persona-path> \
  scripts/run-second-opinion.sh <brief-path>
```

### 3. Launch fresh independent reviewers

Run:

```bash
scripts/run-second-opinion.sh <brief-path>
```

The helper launches separate fresh CLI threads in parallel:

- `codex exec --model gpt-5.6-sol` with
  `model_reasoning_effort="xhigh"` and a read-only sandbox
- `claude -p --model claude-fable-5 --effort max` in plan mode with
  `--tools "Read,Glob,Grep,Bash"` — Bash for read-only investigation only

Both receive the same prompt. Neither sees the other opinion. Never seed one
reviewer with the first reviewer’s answer.

The helper defaults to the full Ilya persona and tells each reviewer to perform
a broad, read-only repo investigation before answering. The brief's linked
files are starting points, not a whitelist.

The generated prompt puts the full persona **before** task instructions. It
requires reviewers to reason at two levels:

1. the **north-star outcome** the project is trying to achieve
2. the **tactical question** currently proposed as a way to get there

Reviewers must challenge whether the tactic attacks the true bottleneck and
surface broader alternatives when another lever reaches the north star more
directly. The prompt also states the review's purpose: improve the owner's
decision, provide a fresh perspective, expand the option space, and identify
decisive evidence. Each reviewer must propose multiple plausible directions
when warranted and, for each, provide:

- the strongest steelman
- the strongest counter-argument and failure conditions
- the smallest observation or experiment that discriminates it from alternatives

Reviewers must not invent weak alternatives to meet a quota; they should say
when evidence leaves only one viable path.

The model and effort pins are part of the contract. If either exact model or
effort is unavailable, report that reviewer as unavailable. Do not silently
substitute another model or lower effort.

### 4. Read and adjudicate

Read:

- `gpt-5.6-sol-opinion.md`
- `claude-fable-5-opinion.md`
- `status.txt`

Then produce your own synthesis. Do not concatenate the answers and do not use
majority vote. Check each argument against local evidence and canonical repo
docs (README, notes, guides, Core ML receipts).

Use this compact structure:

```markdown
## North-star outcome

## Shared conclusion

## Where they disagree

## My adjudication

## Unknowns that could change the answer
```

State plainly when:

- both models agree but rely on the same unsupported assumption
- one opinion is stronger because it uses better evidence
- local firsthand evidence overrides either model
- the question remains genuinely unresolved
- one reviewer failed or was unavailable

### 5. Preserve the boundary

The output is advice for the current matter, not source-of-truth documentation.
If the run reveals durable local learning, use **`write-notes`** separately.
If it reveals a missing external-knowledge domain, use **`create-guide`**
separately with an internet-synthesis brief.

## Do Not Use When

- The user wants an “afternoon of Googling” synthesis → **`create-guide`**
- A raw research export needs integration → **`guide-ingest`**
- The user wants implementation → use the appropriate implementation workflow
- The question is routine and a second model would add cost without meaningful
  independent judgment
- The user asks for only one named model → invoke that model directly instead
  of pretending this two-opinion contract ran

## Failure handling

- Missing CLI, auth failure, empty output, parse failure, or timeout: read
  `status.txt` and the per-reviewer opinion stubs the helper wrote; continue
  with the available reviewer and disclose that the result is not a two-model
  opinion.
- Do not silently substitute another model or lower effort.
- Both fail: report the blocker and stop; answer locally only if the user asks
  for a non-delegated opinion.

## References

- Helper: [`../../../scripts/run-second-opinion.sh`](../../../scripts/run-second-opinion.sh)
- Cross-agent precedent:
  [`../execute-plan/SKILL.md`](../execute-plan/SKILL.md)
- Internet-synthesis boundary:
  [`../create-guide/SKILL.md`](../create-guide/SKILL.md)
- Default persona:
  [`../ilya-sutskever/ilya-sutskever.md`](../ilya-sutskever/ilya-sutskever.md)
