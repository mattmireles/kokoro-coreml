#!/usr/bin/env bash
# Render Config F (Swift + Core ML) clips under the SHIPPED STAGED compute policy.
#
# The staged policy mirrors the production Swift pipeline (swift/.../main.swift):
#   duration=cpuAndGPU, f0n=cpuAndGPU, decoderPre=cpuAndNeuralEngine, generator=cpuAndGPU
# invoked via `--compute-units staged`. bakeoff_listen.py does not expose this
# flag (it defaults to .all), so this thin wrapper calls the same release binary
# directly for the CS1 audio-quality evaluation.
#
# Output: outputs/bakeoff/listen/staged/config_f_staged_{key}.wav (+ .json)
set -euo pipefail
ROOT="/Users/mm/Documents/GitHub/kokoro-coreml"
BENCH="$ROOT/swift/.build/release/kokoro-bench"
MODELS="$ROOT/coreml"
INPUTS="$ROOT/outputs/swift_bench_inputs"
HNSF="$INPUTS/hnsf_weights.json"
OUT="$ROOT/outputs/bakeoff/listen/staged"
mkdir -p "$OUT"

# Keys default to all four frozen buckets; pass a subset as args, e.g.
#   bash scripts/gen_config_f_staged.sh 3s 7s
KEYS=("$@")
if [ ${#KEYS[@]} -eq 0 ]; then
  KEYS=(3s 7s 15s 30s)
fi

for key in "${KEYS[@]}"; do
  echo "=== staged $key ==="
  "$BENCH" \
    --models-dir "$MODELS" \
    --inputs-dir "$INPUTS" \
    --hnsf-weights "$HNSF" \
    --input-key "$key" \
    --compute-units staged \
    --seed 0 \
    --output "$OUT/config_f_staged_${key}.json" \
    --wav "$OUT/config_f_staged_${key}.wav"
done
echo "=== staged renders complete ==="
