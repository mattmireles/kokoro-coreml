#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf '%s\n' \
    "Usage: run-second-opinion.sh <brief-path> [output-dir]" \
    "" \
    "Run independent GPT-5.6 Sol (xhigh) and Claude Fable 5 (max) opinions." \
    "Defaults to the repo's Ilya Sutskever persona. Override with" \
    "SECOND_OPINION_PERSONA_FILE=<path>." \
    "Writes both opinions, raw output, logs, and status to the output directory."
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage >&2
  exit 1
fi

brief_input="$1"
output_input="${2:-}"
timeout_seconds="${SECOND_OPINION_TIMEOUT_SECONDS:-900}"
timeout_exit_code=142

repo_root="$(git rev-parse --show-toplevel 2>/dev/null)" || {
  echo "run-second-opinion.sh must run inside the repo." >&2
  exit 1
}

default_persona="$repo_root/.claude/skills/ilya-sutskever/ilya-sutskever.md"
persona_input="${SECOND_OPINION_PERSONA_FILE:-$default_persona}"
if [[ "$persona_input" = /* ]]; then
  persona_abs="$persona_input"
else
  persona_abs="$repo_root/$persona_input"
fi

if [[ ! -f "$persona_abs" ]]; then
  echo "Persona not found: $persona_input" >&2
  exit 1
fi

if [[ "$brief_input" = /* ]]; then
  brief_abs="$brief_input"
else
  brief_abs="$repo_root/$brief_input"
fi

if [[ ! -f "$brief_abs" ]]; then
  echo "Brief not found: $brief_input" >&2
  exit 1
fi

brief_name="$(basename "$brief_abs")"
brief_slug="${brief_name%.*}"
brief_slug="$(
  printf '%s' "$brief_slug" \
    | tr '[:upper:]' '[:lower:]' \
    | sed 's/[^a-z0-9]/-/g; s/--*/-/g; s/^-//; s/-$//'
)"
timestamp="$(date +%Y%m%d-%H%M%S)"

if [[ -n "$output_input" ]]; then
  if [[ "$output_input" = /* ]]; then
    output_dir="$output_input"
  else
    output_dir="$repo_root/$output_input"
  fi
else
  output_dir="$repo_root/tmp/second-opinions/${brief_slug}-${timestamp}"
fi

mkdir -p "$output_dir"
cd "$repo_root"

request_copy="$output_dir/request.md"
persona_copy="$output_dir/persona.md"
agent_prompt="$output_dir/agent-prompt.txt"
status_file="$output_dir/status.txt"

codex_opinion="$output_dir/gpt-5.6-sol-opinion.md"
codex_raw="$output_dir/gpt-5.6-sol.raw"
codex_log="$output_dir/gpt-5.6-sol.log"
codex_status="$output_dir/gpt-5.6-sol.status"

claude_opinion="$output_dir/claude-fable-5-opinion.md"
claude_raw="$output_dir/claude-fable-5.raw"
claude_log="$output_dir/claude-fable-5.log"
claude_parse_log="$output_dir/claude-fable-5-parse.log"
claude_status="$output_dir/claude-fable-5.status"
claude_mode_file="$output_dir/claude-fable-5.mode"

cp "$brief_abs" "$request_copy"
cp "$persona_abs" "$persona_copy"

cat >"$agent_prompt" <<'EOF'
--- BEGIN PERSONA ---
EOF
cat "$persona_copy" >>"$agent_prompt"
cat >>"$agent_prompt" <<'EOF'
--- END PERSONA ---

You are an independent expert giving a second opinion on a difficult matter.
Apply the persona above throughout the investigation and answer.

Purpose of this review:

- Help the owner make a better technical decision than they could from the
  current framing alone.
- Supply a genuinely fresh perspective, not an affirmation or paraphrase of
  the brief.
- Expand the option space, expose hidden assumptions, and identify the
  smallest decisive evidence or experiment.
- Produce advice grounded in the repository's actual implementation and
  firsthand evidence.

Reason at two levels:

1. **North-star outcome:** What higher-level system or product result is the
   owner actually trying to achieve? What metric should improve, what quality
   must not regress, and why does this outcome matter?
2. **Tactical question:** Is the narrow decision in the brief the best lever
   for that outcome, or is it optimizing the wrong component, assumption, or
   bottleneck?

Restate the north-star outcome before judging the tactic. Trace the proposed
tactic's causal link to that outcome. Challenge the premise when the repository
shows a different bottleneck. If another architecture, model/data strategy,
compute placement, decomposition, or experiment could reach the north star
more directly, surface it even if the brief did not ask about it.

Think broadly and creatively. Look beyond the options already named in the
brief. Propose multiple plausible ideas, solutions, or strategic directions,
including at least one materially different direction when the evidence permits.
For every proposed direction:

1. Steelman it: explain the strongest case, prerequisites, and evidence in its
   favor.
2. Counter-argue it: explain the strongest failure mode, disconfirming evidence,
   opportunity cost, and conditions under which it should be rejected.
3. State what observation or experiment would most efficiently distinguish it
   from the alternatives.

Do not manufacture extra options merely to satisfy a count. If the evidence
collapses the decision to one viable path, explain why the alternatives fail.

Work read-only: do not edit files, create commits, or mutate local/external
state.

The matter below is an initial map, not an evidence boundary. Before answering:

1. Orient yourself in the repository.
2. Read the brief's starting points.
3. Search broadly for all relevant implementation code, tests, plans, notes,
   guides, specifications, status/coordination ledgers, receipts, and recent
   git history or diffs.
4. Look specifically for newer firsthand evidence that contradicts or
   supersedes the brief.
5. Reconcile prose against current code and measured artifacts.

Do not answer from the brief alone. Follow relevant links and search adjacent
repo material until you can explain the strongest evidence on both sides.
Prefer concrete, repo-grounded reasoning over generic advice. Cite the paths
and measured facts that materially drive your recommendation.
Stay inside this repository unless the brief explicitly names an external
artifact. Do not use unrelated user-level memories or other repositories as
evidence.

Return markdown with these exact sections:

## North-star framing

Restate the higher-level outcome, success metrics, non-negotiable quality
constraints, and whether the tactical question targets the real bottleneck.

## Bottom line

## Reasoning

Under this section, compare multiple candidate directions. For each direction,
include its steelman, strongest counter-argument, and discriminating evidence.

## Risks and blind spots

## Recommended next step

## Confidence and unknowns

Matter:

EOF
cat "$request_copy" >>"$agent_prompt"

printf 'brief=%s\n' "$brief_abs" >"$status_file"
printf 'persona=%s\n' "$persona_abs" >>"$status_file"
printf 'output_dir=%s\n' "$output_dir" >>"$status_file"
printf 'timeout_seconds=%s\n' "$timeout_seconds" >>"$status_file"
printf 'gpt_model=gpt-5.6-sol\n' >>"$status_file"
printf 'gpt_reasoning_effort=xhigh\n' >>"$status_file"
printf 'claude_model=claude-fable-5\n' >>"$status_file"
printf 'claude_effort=max\n' >>"$status_file"

classify_codex_failure() {
  if grep -Eiq 'auth|login|credential|token' "$codex_raw" "$codex_log" 2>/dev/null; then
    printf 'auth-failed'
  else
    printf 'failed'
  fi
}

classify_claude_failure() {
  if grep -Eiq 'auth|login|credential|token' "$claude_raw" "$claude_log" 2>/dev/null; then
    printf 'auth-failed'
  else
    printf 'failed'
  fi
}

extract_claude_result() {
  node - "$claude_raw" <<'NODE'
const fs = require("fs");
const rawPath = process.argv[2];
const lines = fs.readFileSync(rawPath, "utf8").trim().split(/\n/).filter(Boolean);
if (lines.length === 0) {
  process.stderr.write("Claude returned no JSON output.\n");
  process.exit(2);
}

let payload;
try {
  payload = JSON.parse(lines[lines.length - 1]);
} catch (error) {
  process.stderr.write(`Claude JSON parse failed: ${error.message}\n`);
  process.exit(2);
}

const result = typeof payload.result === "string" ? payload.result.trim() : "";
if (payload.is_error === true) {
  process.stderr.write(`${result || "Claude reported an error."}\n`);
  if (/authentication|invalid authentication|please run \/login|credential/i.test(result)) {
    process.exit(42);
  }
  process.exit(3);
}

if (!result) {
  process.stderr.write("Claude result was empty or missing.\n");
  process.exit(2);
}

process.stdout.write(`${result}\n`);
NODE
}

claude_login_hint() {
  if [[ "$claude_mode" == "native" ]]; then
    printf '%s/.local/bin/claude' "$HOME"
  elif [[ "$claude_mode" == "node22-shim" ]]; then
    printf 'npx -y node@22 /opt/homebrew/bin/claude'
  else
    printf 'claude'
  fi
}

run_codex() {
  if ! command -v codex >/dev/null 2>&1; then
    printf 'missing\n' >"$codex_status"
    printf 'Codex CLI not found on PATH.\n' >"$codex_opinion"
    return 0
  fi

  # The Codex desktop app rewrites ~/.codex/config.toml with service_tier
  # values the CLI rejects at config-parse time. Override to "fast" so reviews
  # keep working when the shared config has been rewritten.
  local exit_code=0
  perl -e 'alarm shift; exec @ARGV' \
    "$timeout_seconds" \
    codex exec \
    -C "$repo_root" \
    -c 'service_tier="fast"' \
    -c 'model_reasoning_effort="xhigh"' \
    --model gpt-5.6-sol \
    --sandbox read-only \
    --ephemeral \
    --json \
    --output-last-message "$codex_opinion" \
    - <"$agent_prompt" >"$codex_raw" 2>"$codex_log" || exit_code=$?

  if [[ $exit_code -eq 0 && -s "$codex_opinion" ]]; then
    printf 'ok\n' >"$codex_status"
  elif [[ $exit_code -eq 0 ]]; then
    printf 'empty-output\n' >"$codex_status"
    printf 'GPT-5.6 Sol completed without a final message. See %s and %s\n' \
      "$codex_raw" "$codex_log" >"$codex_opinion"
  elif [[ $exit_code -eq $timeout_exit_code ]]; then
    printf 'timed-out\n' >"$codex_status"
    printf 'GPT-5.6 Sol timed out after %ss. See %s and %s\n' \
      "$timeout_seconds" "$codex_raw" "$codex_log" >"$codex_opinion"
  else
    local failure
    failure="$(classify_codex_failure)"
    printf '%s\n' "$failure" >"$codex_status"
    printf 'GPT-5.6 Sol %s (exit %s). See %s and %s\n' \
      "$failure" "$exit_code" "$codex_raw" "$codex_log" >"$codex_opinion"
  fi
}

run_claude() {
  claude_mode=""
  claude_cmd=()
  if [[ -x "$HOME/.local/bin/claude" ]]; then
    claude_mode="native"
    claude_cmd=("$HOME/.local/bin/claude")
  elif command -v npx >/dev/null 2>&1 && [[ -x /opt/homebrew/bin/claude ]]; then
    claude_mode="node22-shim"
    claude_cmd=(npx -y node@22 /opt/homebrew/bin/claude)
  elif command -v claude >/dev/null 2>&1; then
    claude_mode="path"
    claude_cmd=(claude)
  fi

  if [[ ${#claude_cmd[@]} -eq 0 ]]; then
    printf 'missing\n' >"$claude_status"
    printf 'none\n' >"$claude_mode_file"
    printf 'Claude CLI not found on PATH.\n' >"$claude_opinion"
    return 0
  fi

  printf '%s\n' "$claude_mode" >"$claude_mode_file"

  local exit_code=0
  perl -e 'alarm shift; exec @ARGV' \
    "$timeout_seconds" \
    "${claude_cmd[@]}" \
    -p \
    --model claude-fable-5 \
    --effort max \
    --permission-mode plan \
    --tools "Read,Glob,Grep,Bash" \
    --output-format json \
    --no-session-persistence \
    <"$agent_prompt" >"$claude_raw" 2>"$claude_log" || exit_code=$?

  if [[ $exit_code -eq 0 ]]; then
    local parse_status=0
    if extract_claude_result >"$claude_opinion" 2>"$claude_parse_log"; then
      printf 'ok\n' >"$claude_status"
    else
      parse_status=$?
      if [[ $parse_status -eq 42 ]]; then
        printf 'auth-failed\n' >"$claude_status"
        printf 'Claude Fable 5 auth-failed. Run `%s`, then type `/login` or `/status`.\nSee %s, %s, and %s\n' \
          "$(claude_login_hint)" \
          "$claude_raw" "$claude_log" "$claude_parse_log" >"$claude_opinion"
      else
        printf 'parse-failed\n' >"$claude_status"
        printf 'Claude Fable 5 output was not parseable. See %s, %s, and %s\n' \
          "$claude_raw" "$claude_log" "$claude_parse_log" >"$claude_opinion"
      fi
    fi
  elif [[ $exit_code -eq $timeout_exit_code ]]; then
    printf 'timed-out\n' >"$claude_status"
    printf 'Claude Fable 5 timed out after %ss. See %s and %s\n' \
      "$timeout_seconds" "$claude_raw" "$claude_log" >"$claude_opinion"
  else
    local failure
    failure="$(classify_claude_failure)"
    printf '%s\n' "$failure" >"$claude_status"
    if [[ "$failure" == "auth-failed" ]]; then
      printf 'Claude Fable 5 auth-failed. Run `%s`, then type `/login` or `/status`.\nSee %s and %s\n' \
        "$(claude_login_hint)" "$claude_raw" "$claude_log" >"$claude_opinion"
    else
      printf 'Claude Fable 5 %s (exit %s). See %s and %s\n' \
        "$failure" "$exit_code" "$claude_raw" "$claude_log" >"$claude_opinion"
    fi
  fi
}

run_codex &
codex_pid=$!
run_claude &
claude_pid=$!

wait "$codex_pid" || true
wait "$claude_pid" || true

printf 'gpt_status=%s\n' "$(<"$codex_status")" >>"$status_file"
printf 'claude_status=%s\n' "$(<"$claude_status")" >>"$status_file"
if [[ -f "$claude_mode_file" ]]; then
  printf 'claude_mode=%s\n' "$(<"$claude_mode_file")" >>"$status_file"
else
  printf 'claude_mode=unknown\n' >>"$status_file"
fi

cat <<EOF
Second-opinion run finished.
Output dir: $output_dir
Status: $status_file
GPT-5.6 Sol: $codex_opinion
Claude Fable 5: $claude_opinion
EOF
