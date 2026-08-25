#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
skill_dir=$(CDPATH= cd -- "$script_dir/.." && pwd -P)
repo_root=$(CDPATH= cd -- "$skill_dir/../../.." && pwd -P)
template="$skill_dir/assets/Plans-template.md"

if [ "$#" -ne 1 ]; then
  echo "usage: $(basename -- "$0") <new-plan-path>" >&2
  exit 64
fi
if [ ! -f "$template" ]; then
  echo "missing canonical template: $template" >&2
  exit 66
fi

case "$1" in
  /*) requested_path=$1 ;;
  *) requested_path="$PWD/$1" ;;
esac

requested_dir=$(dirname -- "$requested_path")
mkdir -p "$requested_dir"
resolved_dir=$(CDPATH= cd -- "$requested_dir" && pwd -P)
case "$resolved_dir/" in
  "$repo_root/"*) ;;
  *)
    echo "plan path must be inside repository: $repo_root" >&2
    exit 64
    ;;
esac

destination="$resolved_dir/$(basename -- "$requested_path")"
if [ -e "$destination" ] || [ -L "$destination" ]; then
  echo "refusing to overwrite existing plan: $destination" >&2
  exit 73
fi

cp "$template" "$destination"
echo "$destination"
