#!/bin/bash
# Plan 014 Phase 3 Mac driver. usage: mac_driver_014.sh HOST_ID [ablation]
# Run from ~/kokoro-014 (frozen source at 903f1fc1, models_014 and models_014_r01).
# Arms: folded, hybrid, hybrid-e5, mlx at 3/7/10/15/30 s; with "ablation" also
# R0, R1, R2 at 3 s and 30 s. Config F: 3 preflight + 1 cold + 10 warm.
set -u
HOST=$1; ABL=${2:-}
cd ~/kokoro-014
OUT=outputs/frozen_rerun_014/$HOST; mkdir -p $OUT
SC=/Applications/Botnet.app/Contents/Resources/SupervisorCLI
MLX_PY=$HOME/kokoro-external-bakeoff/mlx-venv/bin/python
export TMPDIR=$HOME/kokoro-014/tmp; mkdir -p $TMPDIR

log() {  # arm phase
  python3 - "$1" "$2" "$(sysctl -n vm.loadavg)" "$(pmset -g therm 2>&1 | tr '\n' ' ')" >> $OUT/run_log.jsonl <<'P'
import json, sys, datetime
print(json.dumps({"t": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                  "arm": sys.argv[1], "phase": sys.argv[2], "loadavg": sys.argv[3], "therm": sys.argv[4]}))
P
}
gate() {  # wait up to GATE_TRIES x 30 s (default 20 min) for 1-min load <= 1.5
  for i in $(seq 1 ${GATE_TRIES:-40}); do
    l=$(sysctl -n vm.loadavg | awk '{print $2}')
    python3 -c "import sys; sys.exit(0 if float('$l') <= 1.5 else 1)" && { log "$1" gate_pass; return; }
    sleep 30
  done
  log "$1" gate_fail
}
botnet_mcp() {  # mode -> set_share_mode
  python3 - "$1" <<'P'
import json, subprocess, sys
SC = "/Applications/Botnet.app/Contents/Resources/SupervisorCLI"
p = subprocess.Popen([SC, "mcp"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
def send(o): p.stdin.write(json.dumps(o) + "\n"); p.stdin.flush()
send({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "cc", "version": "1"}}}); p.stdout.readline()
send({"jsonrpc": "2.0", "method": "notifications/initialized"})
send({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "botnet_set_share_mode", "arguments": {"mode": sys.argv[1]}}}); print(p.stdout.readline()[:300])
p.stdin.close(); p.terminate()
P
}

# Botnet: record, pause, drain.
if [ -x $SC ]; then
  $SC status > $OUT/botnet_before.json 2>/dev/null
  # PRIOR_OVERRIDE: the true prior mode when resuming a host this driver already paused.
  PRIOR=${PRIOR_OVERRIDE:-$(python3 -c "import json; d=json.load(open('$OUT/botnet_before.json')); o=(d.get('hostShare',{}).get('policy',{}).get('manualOverride') or {}); print(o.get('mode') or '')")}
  echo "prior override: '$PRIOR'" >> $OUT/botnet.log
  $SC pause >> $OUT/botnet.log 2>&1
  for i in $(seq 1 60); do
    n=$($SC status 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(sum(s.get('activeJobs',0) for s in d.get('engineServices',[])))")
    [ "$n" = "0" ] && break; sleep 10
  done
  sleep 60
fi

want() { [ -z "${ARMS:-}" ] || [[ " $ARMS " == *" $1 "* ]]; }   # ARMS="R0 R1": run only these
cf() {  # arm models extra-args... ; env passed by caller
  local arm=$1 models=$2; shift 2
  want $arm || return 0
  gate $arm; log $arm start
  python3 scripts/external_bakeoff/run_config_f_reference.py --machine-id $HOST --models-dir $models \
    --compute-units staged --use-padded-duration-models --iterations 10 --preflight-runs 3 \
    --spotcheck-dir $OUT/wav_$arm --output $OUT/$arm.json "$@" > $OUT/$arm.stdout 2> $OUT/$arm.stderr
  log $arm end_rc$?
}

KOKORO_FOLDED_GENERATOR=1 cf folded models_014
KOKORO_FOLDED_GENERATOR=1 KOKORO_HYBRID_GENERATOR=1 cf hybrid models_014
KOKORO_FOLDED_GENERATOR=1 KOKORO_HYBRID_GENERATOR=1 KOKORO_HYBRID_E5=1 cf hybrid-e5 models_014
if want mlx; then
gate mlx; log mlx start
$MLX_PY scripts/external_bakeoff/run_mlx_audio.py --machine-id $HOST --iterations 10 \
  --spotcheck-dir $OUT/wav_mlx --output $OUT/mlx.json > $OUT/mlx.stdout 2> $OUT/mlx.stderr
log mlx end_rc$?
fi
if [ "$ABL" = ablation ]; then
  KOKORO_SPLIT_DURATION=0 KOKORO_SPLIT_F0N=0 cf R0 models_014_r01 --input-key 3s --input-key 30s
  cf R1 models_014_r01 --input-key 3s --input-key 30s
  KOKORO_FOLDED_GENERATOR=0 cf R2 models_014 --input-key 3s --input-key 30s
fi

# Botnet: restore the prior override.
if [ -x $SC ]; then
  if [ -n "$PRIOR" ]; then botnet_mcp "$PRIOR" >> $OUT/botnet.log 2>&1; else $SC resume >> $OUT/botnet.log 2>&1; fi
  $SC status > $OUT/botnet_after.json 2>/dev/null
fi
log all done
