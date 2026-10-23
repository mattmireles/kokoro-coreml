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

# Runtime buckets shipped in the bench app (the five frozen runtime buckets).
BUCKETS=(3 7 10 15 30)
# Padded duration sizes covering the 44/105/219/476-token bench inputs.
PADDED=(64 128 256 512)
# Exact-native-LSTM duration sizes, one per bench input's true token count
# (--exact-duration path; 780-op graphs vs 17k-134k-op unrolled).
EXACT=()  # HF dadebca4 ships no exact packages; the shipped default is padded.
# F0Ntrain frame sizes per bucket (PipelineConstants.tFramesForBucket).
F0N=(120 280 400 600 1200)

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
for t in "${EXACT[@]+"${EXACT[@]}"}"; do copy "kokoro_duration_exact_t${t}.mlpackage"; done
# Shipped-default flexible generator (KokoroPipeline flexibleGeneratorPackage).
copy "kokoro_decoder_har_post_range.mlpackage"
for t in "${F0N[@]}"; do copy "kokoro_f0ntrain_t${t}.mlpackage"; done

# Optional: row-folded Neural Engine generators (README/Notes/ane-folded-generator.md),
# exported into ../coreml-fold. The app uses them only with --folded-generator 1.
FOLD_SRC=../coreml-fold
for b in "${BUCKETS[@]}"; do
  for name in "kokoro_generator_fold_${b}s.mlpackage" "kokoro_generator_fold_poly_${b}s.mlpackage"; do
    if [ -d "$FOLD_SRC/$name" ]; then rsync -a --delete "$FOLD_SRC/$name" "$DST/"; fi
  done
done

# Optional: hybrid generator (ANE convs + Metal), from ../coreml-hybrid. Used only
# with --hybrid-generator. plan.json/adain.bin ride along with the bench inputs
# so they land at the bundle root next to the compiled packages.
HYB_SRC=../coreml-hybrid
if [ -d "$HYB_SRC" ]; then
  for pkg in "$HYB_SRC"/hybrid_generator_r*.mlpackage; do rsync -a --delete "$pkg" "$DST/"; done
  cp "$HYB_SRC/plan.json" "$HYB_SRC/adain.bin" Resources/bench_inputs/
fi

# Optional: two-part duration model (ANE-layout BERT front + CPU back), from
# ../coreml-dur-ane (scripts/export_duration_ane.py). Used only with --split-duration.
DUR_SRC=../coreml-dur-ane
if [ -d "$DUR_SRC" ]; then
  for pkg in "$DUR_SRC"/kokoro_duration_ane_front_t*.mlpackage "$DUR_SRC"/kokoro_duration_back_t*.mlpackage; do
    rsync -a --delete "$pkg" "$DST/"
  done
fi

# Optional: DecoderPre in the ANE layout (scripts/export_decoder_pre_ane.py),
# same contract as the shipping package, stored under the shipping names in
# ../coreml-dpre-ane. Replaces the shipping DecoderPre when present.
DPRE_SRC=../coreml-dpre-ane
if [ -d "$DPRE_SRC" ]; then
  for b in "${BUCKETS[@]}"; do
    pkg="$DPRE_SRC/kokoro_decoder_pre_${b}s.mlpackage"
    if [ -d "$pkg" ]; then rsync -a --delete "$pkg" "$DST/"; fi
  done
fi

# Optional: two-part F0Ntrain (CPU LSTM + ANE branches), from ../coreml-f0n-split
# (scripts/export_f0ntrain_split.py). Used only with --split-f0n.
F0N_SRC=../coreml-f0n-split
if [ -d "$F0N_SRC" ]; then
  for pkg in "$F0N_SRC"/kokoro_f0ntrain_front_t*.mlpackage "$F0N_SRC"/kokoro_f0ntrain_ane_back_t*.mlpackage; do
    rsync -a --delete "$pkg" "$DST/"
  done
fi

if [ "$missing" -ne 0 ]; then
  echo "Some packages are missing from $SRC — export them first." >&2
  exit 1
fi
echo "Resources/coreml populated. Re-run 'xcodegen generate' before building."
