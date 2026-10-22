#!/usr/bin/env bash
# Build the paper with tectonic and package the arXiv submission tarball.
#
# Output: paper/main.pdf and paper/arxiv-submission.tar.gz containing
# main.tex, main.bbl, refs.bib and figures/*.pdf. arXiv compiles main.tex with
# pdfLaTeX and uses main.bbl as-is, so the .bbl must come from this build.
#
# Usage: paper/build_arxiv.sh   (set TECTONIC=/path/to/tectonic if not on PATH)
set -euo pipefail

cd "$(dirname "$0")"
TECTONIC="${TECTONIC:-tectonic}"

"$TECTONIC" -X compile main.tex --keep-logs --keep-intermediates

if grep -q "Overfull" main.log; then
  echo "warning: overfull boxes in main.log" >&2
fi

rm -f arxiv-submission.tar.gz
COPYFILE_DISABLE=1 tar -czf arxiv-submission.tar.gz main.tex main.bbl refs.bib figures/*.pdf
tar -tzf arxiv-submission.tar.gz
