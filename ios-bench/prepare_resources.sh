#!/bin/bash
# Populate ios-bench/Resources from repo-local artifacts.
#
# Resources/coreml and Resources/mlx are gitignored (hundreds of MB of model
# artifacts); this script makes the checkout buildable. Run it before
# `xcodegen generate` — xcodegen globs Resources/* at generate time, so new
# packages need a re-generate to enter the Xcode resources phase.
#
# Sources:
#   - Core ML packages: ../coreml/ (produced by export_duration.py,
#     export_f0ntrain.py, export_decoder_pre.py, export_synth/ — see
#     README/Wiki/coreml-export.md)
#   - Bench inputs: produced by scripts/prepare_swift_bench_inputs.py
#     (committed — small JSON)
#   - MLX weights (optional, only for the mlx arm): kokoro-v1_0.safetensors
#     and voices.npz from the mlalma/KokoroTestApp distribution; place them
#     in Resources/mlx manually if benchmarking the MLX arm.
set -euo pipefail
cd "$(dirname "$0")"

SRC=../coreml
DST=Resources/coreml
mkdir -p "$DST"

# Runtime buckets shipped in the bench app (10s omitted: no 10s bakeoff input).
BUCKETS=(3 7 15 30)
# Padded duration sizes covering the 44/105/219/476-token bench inputs.
PADDED=(64 128 256 512)
# Exact-native-LSTM duration sizes, one per bench input's true token count
# (--exact-duration path; 780-op graphs vs 17k-134k-op unrolled).
EXACT=(44 105 219 476)
# F0Ntrain frame sizes per bucket (PipelineConstants.tFramesForBucket).
F0N=(120 280 600 1200)

missing=0
copy() {
  local name="$1"
  if [ ! -d "$SRC/$name" ]; then
    echo "MISSING: $SRC/$name" >&2
    missing=1
    return
  fi
  rsync -a --delete "$SRC/$name" "$DST/"
}

for b in "${BUCKETS[@]}"; do
  copy "kokoro_decoder_pre_${b}s.mlpackage"
  copy "kokoro_decoder_har_post_${b}s.mlpackage"
done
for t in "${PADDED[@]}"; do copy "kokoro_duration_t${t}.mlpackage"; done
for t in "${EXACT[@]}"; do copy "kokoro_duration_exact_t${t}.mlpackage"; done
for t in "${F0N[@]}"; do copy "kokoro_f0ntrain_t${t}.mlpackage"; done

if [ "$missing" -ne 0 ]; then
  echo "Some packages are missing from $SRC — export them first." >&2
  exit 1
fi
echo "Resources/coreml populated. Re-run 'xcodegen generate' before building."
