#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
skill_dir=$(CDPATH= cd -- "$script_dir/.." && pwd -P)
template="$skill_dir/assets/Plans-template.md"

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

echo "template contract valid: $1"
