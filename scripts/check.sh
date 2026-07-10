#!/usr/bin/env bash
set -euo pipefail

node scripts/memory-health.mjs --write-coverage
node scripts/memory-health.mjs --strict
node --check scripts/memory-health.mjs
git diff --check
