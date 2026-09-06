#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
skill_dir=$(CDPATH= cd -- "$script_dir/.." && pwd -P)
template="$skill_dir/assets/Plans-template.md"
repo_root=$(git -C "$script_dir" rev-parse --show-toplevel 2>/dev/null || true)

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "usage: $(basename -- "$0") <plan-path> [--allow-placeholders]" >&2
  exit 64
fi
if [ ! -f "$template" ] || [ ! -f "$1" ]; then
  echo "template or plan file is missing" >&2
  exit 66
fi
if [ "${2:-}" != "" ] && [ "${2:-}" != "--allow-placeholders" ]; then
  echo "unknown option: $2" >&2
  exit 64
fi

progress_model='**Progress model:** Phase task checkboxes only'
implementation_count=$(grep -c -F -x -- '## Implementation Phases' "$1" || true)
if [ "$implementation_count" -ne 1 ]; then
  echo "plan must contain exactly one ## Implementation Phases section" >&2
  exit 1
fi

if grep -F -x -- "$progress_model" "$1" >/dev/null; then
  equivalent_implementation_count=$(awk '
    /^[[:space:]]*(```|~~~)/ { in_fence = !in_fence; next }
    !in_fence && /^[[:space:]]*##[[:space:]]+Implementation Phases([[:space:]]+#+)?[[:space:]]*$/ { count++ }
    END { print count + 0 }
  ' "$1")
  if [ "$equivalent_implementation_count" -ne 1 ]; then
    echo "checkbox-only plans require one canonical ## Implementation Phases section" >&2
    exit 1
  fi
  if [ "${2:-}" != "--allow-placeholders" ]; then
    status_count=$(grep -c -E '^\*\*Status:\*\* ' "$1" || true)
    canonical_status_count=$(grep -c -E '^\*\*Status:\*\* (Planned|In-Progress|Complete)$' "$1" || true)
    if [ "$status_count" -ne 1 ] || [ "$canonical_status_count" -ne 1 ]; then
      echo "checkbox-only plans require one lifecycle-only Status value" >&2
      exit 1
    fi
  fi
  awk '
    /^[[:space:]]*(```|~~~)/ { in_fence = !in_fence; next }
    in_fence { next }

    /^[[:space:]>]*#{2,6}[[:space:]]+/ {
      line = $0
      sub(/^[[:space:]>]+/, "", line)
      sub(/[[:space:]]+#+[[:space:]]*$/, "", line)
      lower = tolower(line)
      if (lower ~ /^#{2,6}[[:space:]]+(progress( tracker| updates?)?|debug notes|execution (diary|log|notes|history)|verification log|receipt ledger)([[:space:]]*:.*)?$/) {
        printf "checkbox-only plans cannot contain duplicate progress or execution-log sections: %s\n", $0 > "/dev/stderr"
        invalid = 1
      }
    }

    /^[[:space:]>]*\*\*/ {
      if ($0 == "**Progress model:** Phase task checkboxes only") next
      line = $0
      sub(/^[[:space:]>]+/, "", line)
      lower = tolower(line)
      if (lower ~ /^\*\*(progress|verified|partial|code complete|current phase .* state)([ :—-]|$)/) {
        printf "checkbox-only plans cannot contain inline execution checkpoints: %s\n", $0 > "/dev/stderr"
        invalid = 1
      }
    }

    /^## Implementation Phases$/ { in_impl = 1; next }
    in_impl && /^[[:space:]]*##[[:space:]]+/ { in_impl = 0 }
    /^[[:space:]]*###[[:space:]]+Phase[[:space:]]+[0-9]+/ {
      if ($0 !~ /^### Phase [0-9]+:/) {
        printf "noncanonical phase heading; use ### Phase N: Name: %s\n", $0 > "/dev/stderr"
        invalid = 1
      }
    }
    /^[[:space:]>]*([-+*]|[0-9]+[.)])[[:space:]]+\[[^]]*\][[:space:]]/ {
      if (!in_impl) {
        printf "task checkbox outside Implementation Phases: %s\n", $0 > "/dev/stderr"
        invalid = 1
      } else if ($0 !~ /^- \[( |x|X)\] /) {
        printf "nonstandard task checkbox; use unindented [ ] or [x]: %s\n", $0 > "/dev/stderr"
        invalid = 1
      }
    }
    END { if (invalid) exit 1 }
  ' "$1"
else
  plan_name=$(basename -- "$1")
  case "$plan_name" in
    [0-9][0-9][0-9]-*.md)
      plan_number=${plan_name%%-*}
      if [ "$plan_number" -ge 13 ]; then
        echo "missing progress model declaration" >&2
        exit 1
      fi
      ;;
  esac
  tracked_path=$1
  case "$tracked_path" in
    "$repo_root"/*) tracked_path=${tracked_path#"$repo_root"/} ;;
  esac
  if [ -z "$repo_root" ] \
    || ! git -C "$repo_root" cat-file -e "HEAD:$tracked_path" 2>/dev/null \
    || git -C "$repo_root" show "HEAD:$tracked_path" 2>/dev/null | grep -F -x -- "$progress_model" >/dev/null; then
    echo "missing progress model declaration" >&2
    exit 1
  fi
fi

scratch_dir=$(mktemp -d "${TMPDIR:-/tmp}/create-plan-validate.XXXXXX")
trap 'rm -rf "$scratch_dir"' EXIT HUP INT TERM
expected="$scratch_dir/expected"
actual="$scratch_dir/actual"

extract_required_headings() {
  awk '
    /^## Modules$/ { in_modules = 1; next }
    in_modules && /^## Critical Reminder$/ { in_modules = 0 }
    in_modules { next }
    /^## Implementation Phases$/ { in_phases = 1; print; next }
    in_phases && /^## / { in_phases = 0 }
    /^### Phase / { next }
    /^## |^### / { print }
  ' "$1"
}

extract_required_headings "$template" > "$expected"
extract_required_headings "$1" > "$actual"

last_line=0
while IFS= read -r heading; do
  line=$(grep -n -F -x -- "$heading" "$actual" | head -n 1 | cut -d: -f1 || true)
  if [ -z "$line" ]; then
    echo "missing required heading: $heading" >&2
    exit 1
  fi
  if [ "$line" -le "$last_line" ]; then
    echo "required heading is out of order: $heading" >&2
    exit 1
  fi
  last_line=$line
done < "$expected"

if [ "${2:-}" != "--allow-placeholders" ]; then
  if grep -n -E '^# \[Feature Name\] Plan$|^\*\*Date:\*\* YYYY-MM-DD$|^\*\*Status:\*\* Planned \| In-Progress \| Complete$' "$1" >/dev/null; then
    echo "unfilled top-level template placeholders remain" >&2
    exit 1
  fi
fi

awk -v allow_placeholders="${2:-}" '
  /^## Implementation Phases$/ { in_impl = 1; next }
  in_impl && /^[[:space:]]*##[[:space:]]+/ {
    if (phase != "" && skills == 0) {
      printf "missing **Skills:** in %s\n", phase > "/dev/stderr"
      missing = 1
    }
    in_impl = 0
    phase = ""
    next
  }
  in_impl && /^### Phase [0-9]+:/ {
    if (phase != "" && skills == 0) {
      printf "missing **Skills:** in %s\n", phase > "/dev/stderr"
      missing = 1
    }
    phase = $0
    skills = 0
    next
  }
  in_impl && /^\*\*Skills:\*\*.*`[a-z0-9-]+`/ { skills = 1 }
  in_impl && allow_placeholders == "--allow-placeholders" && /^\*\*Skills:\*\* \[Skills to use in this phase\]$/ { skills = 1 }
  END {
    if (phase != "" && skills == 0) {
      printf "missing **Skills:** in %s\n", phase > "/dev/stderr"
      missing = 1
    }
    if (missing) exit 1
  }
' "$1"

if [ "${2:-}" != "--allow-placeholders" ]; then
  skills_file="$scratch_dir/skills"
  awk '
    /^\*\*Skills:\*\*/ {
      while (match($0, /`[a-z0-9-]+`/)) {
        print substr($0, RSTART + 1, RLENGTH - 2)
        $0 = substr($0, RSTART + RLENGTH)
      }
    }
  ' "$1" > "$skills_file"
  while IFS= read -r skill_name; do
    case "$skill_name" in
      ''|*[!a-z0-9-]*)
        echo "invalid skill name: $skill_name" >&2
        exit 1
        ;;
    esac
    if [ ! -f "$skill_dir/../$skill_name/SKILL.md" ] \
      && { [ -z "$repo_root" ] || [ ! -f "$repo_root/.cursor/skills/$skill_name/SKILL.md" ]; } \
      && [ ! -f "${HOME:-}/.cursor/skills/$skill_name/SKILL.md" ]; then
      echo "unknown skill: $skill_name" >&2
      exit 1
    fi
  done < "$skills_file"
fi

echo "template contract valid: $1"
