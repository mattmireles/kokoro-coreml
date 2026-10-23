#!/bin/bash
# Plan 014: peak RAM (max RSS, /usr/bin/time -l) per arm and bucket, one fresh
# process per cell: 3 preflight-free calls of kokoro-bench --input-key (Core ML
# arms) or one run_mlx_audio iteration (MLX). Run from ~/kokoro-014.
HOST=$1; cd ~/kokoro-014; OUT=outputs/frozen_rerun_014/$HOST/ram; mkdir -p $OUT; export TMPDIR=$HOME/kokoro-014/tmp
MLX_PY=$HOME/kokoro-external-bakeoff/mlx-venv/bin/python
for k in 3s 7s 10s 15s 30s; do
  for arm in folded hybrid hybrid-e5; do
    case $arm in folded) E="KOKORO_FOLDED_GENERATOR=1";; hybrid) E="KOKORO_FOLDED_GENERATOR=1 KOKORO_HYBRID_GENERATOR=1";; hybrid-e5) E="KOKORO_FOLDED_GENERATOR=1 KOKORO_HYBRID_GENERATOR=1 KOKORO_HYBRID_E5=1";; esac
    env $E /usr/bin/time -l swift/.build/release/kokoro-bench --compute-units staged --models-dir models_014 --inputs-dir outputs/swift_bench_inputs --input-key $k --seed 42 --hnsf-weights outputs/swift_bench_inputs/hnsf_weights.json > /dev/null 2> $OUT/${arm}_$k.txt
    echo "$arm $k $(grep 'maximum resident' $OUT/${arm}_$k.txt | awk '{printf "%.0f MB", $1/1048576}')"
  done
  /usr/bin/time -l $MLX_PY scripts/external_bakeoff/run_mlx_audio.py --machine-id $HOST --iterations 1 --input-key $k --output /tmp/ram_mlx_$k.json > /dev/null 2> $OUT/mlx_$k.txt
  echo "mlx $k $(grep 'maximum resident' $OUT/mlx_$k.txt | awk '{printf "%.0f MB", $1/1048576}')"
done
