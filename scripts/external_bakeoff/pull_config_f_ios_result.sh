#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEVICE_ID="${DEVICE_ID:-00008101-001134561A0A001E}"
APP_BUNDLE_ID="${APP_BUNDLE_ID:-com.kokoro.externalbakeoff.ConfigFIOSRunnerManual}"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/outputs/external_bakeoff}"
PULLED_JSON="${PULLED_JSON:-${RESULTS_DIR}/config_f_ios_result_latest.json}"
INGESTED_JSON="${INGESTED_JSON:-${RESULTS_DIR}/results_config_f_reference_ios_iphone-12-pro.json}"

mkdir -p "${RESULTS_DIR}"

xcrun devicectl device copy from \
  --device "${DEVICE_ID}" \
  --domain-type appDataContainer \
  --domain-identifier "${APP_BUNDLE_ID}" \
  --source Documents/config_f_ios_result.json \
  --destination "${PULLED_JSON}"

cd "${REPO_ROOT}"
python scripts/external_bakeoff/ingest_ios_runner_result.py \
  --input "${PULLED_JSON}" \
  --machine-id iphone-12-pro \
  --device-model "iPhone 12 Pro (iPhone13,3)" \
  --version "Config F manual iOS runner; staged compute units; exact duration packages" \
  --output "${INGESTED_JSON}"

python scripts/external_bakeoff/summarize_competitive_frontier.py \
  --output outputs/external_bakeoff/competitive_frontier.md \
  --json-output outputs/external_bakeoff/competitive_frontier.json
