#!/usr/bin/env python3
"""Plan 014 Phase 2 driver: one arm x one bucket per app launch, cooled between.

usage: iphone_driver_014.py UDID OUTDIR APPDIR [--ablation]

Arms (README/Plans/014-paper-cs1-hybrid-frozen-rerun-plan.md):
  folded, hybrid, hybrid-e5, mlx on the main app at every bucket; with
  --ablation also R0, R1 (r01 app) and R2 (main app) at 3 s and 30 s.
  R3 = folded and R4 = hybrid reuse the main cells.
Each launch: 3 preflight + 1 cold + 10 warm (ios-bench ladder defaults).
"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

UDID, OUT, APPDIR = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
ABLATION = "--ablation" in sys.argv
BUNDLE = "com.mattmireles.KokoroIPhoneBench"
COOL_S = int(os.environ.get("COOL_S", "600"))  # plan 014: 180 s after the 11:16 PT validation
STEP_TIMEOUT_S = 60 * 60
KEYS = ["3s", "7s", "10s", "15s", "30s"]
COREML = ["--mode", "ladder", "--arms", "coreml", "--policy", "runtime"]
SPLIT = ["--split-duration", "--split-f0n"]
ARMS = {
    "folded": ("main", COREML + SPLIT + ["--folded-generator"], {}),
    "hybrid": ("main", COREML + SPLIT + ["--folded-generator", "--hybrid-generator"], {}),
    "hybrid-e5": ("main", COREML + SPLIT + ["--folded-generator", "--hybrid-generator"], {"KOKORO_HYBRID_E5": "1"}),
    "mlx": ("main", ["--arms", "mlx"], {}),
    "R2": ("main", COREML + SPLIT, {}),
    "R1": ("r01", COREML + SPLIT, {}),
    "R0": ("r01", COREML, {"KOKORO_SPLIT_DURATION": "0", "KOKORO_SPLIT_F0N": "0"}),
}
main_arms = ["folded", "hybrid", "hybrid-e5", "mlx"]
STEPS = []
for i, key in enumerate(KEYS):
    order = main_arms if i % 2 == 0 else main_arms[::-1]   # alternate per bucket
    STEPS += [(key, a) for a in order]
if ABLATION:
    for i, key in enumerate(["3s", "30s"]):
        order = ["R0", "R1", "R2"] if i == 0 else ["R2", "R1", "R0"]
        STEPS += [(key, a) for a in order]
# Group app installs: main-app steps first, r01 steps last.
STEPS.sort(key=lambda s: ARMS[s[1]][0] == "r01")
OUT.mkdir(parents=True, exist_ok=True)
events = open(OUT / "driver_events.log", "a")
installed = None


def ev(msg):
    line = f"{datetime.now().astimezone().isoformat(timespec='seconds')} {msg}"
    print(line, flush=True)
    events.write(line + "\n")
    events.flush()


def install(app):
    global installed
    if installed == app:
        return
    r = subprocess.run(["xcrun", "devicectl", "device", "install", "app", "--device", UDID, str(APPDIR / f"{app}.app")],
                       capture_output=True, text=True)
    ev(f"INSTALL {app} rc={r.returncode}")
    installed = app


def pull(name):
    dst = OUT / name
    r = subprocess.run(["xcrun", "devicectl", "device", "copy", "from", "--device", UDID,
                        "--domain-type", "appDataContainer", "--domain-identifier", BUNDLE,
                        "--source", f"Documents/{name}", "--destination", str(dst)],
                       capture_output=True, text=True)
    return dst if r.returncode == 0 and dst.exists() else None


def run_step(key, arm, attempt):
    app, args, env = ARMS[arm]
    install(app)
    tag = datetime.now().strftime("%H%M")
    name = f"{arm}_{key}_a{attempt}_{tag}.json"
    log_path = OUT / f"{arm}_{key}_a{attempt}_{tag}.console.log"
    cmd = ["xcrun", "devicectl", "device", "process", "launch", "--console", "--terminate-existing",
           "--device", UDID]
    if env:
        cmd += ["-e", json.dumps(env)]
    cmd += [BUNDLE] + args + ["--keys", key, "--out", name]
    ev(f"LAUNCH {arm} {key} attempt={attempt} env={env}")
    t0 = time.time()
    done = False
    with open(log_path, "w") as log:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            log.write(line)
            log.flush()
            if "BENCHDONE" in line:
                done = True
                break
            if time.time() - t0 > STEP_TIMEOUT_S:
                break
        p.terminate()
        try:
            p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            p.kill()
    ev(f"END {arm} {key} attempt={attempt} benchdone={done} elapsed_s={time.time() - t0:.0f}")
    if "BSErrorCodeDescription = Locked" in log_path.read_text():
        ev(f"LOCKED: phone refused launch of {arm} {key}; stopping driver")
        sys.exit(2)
    dst = pull(name)
    state = None
    if dst:
        recs = json.loads(dst.read_text()).get("records", [])
        for r in recs:
            ts = r.get("thermal_states") or []
            if ts:
                state = ts[0]
        ev(f"PULLED {dst.name} records={len(recs)} errors={[r.get('error') for r in recs if r.get('error')]} thermal_first={state}")
    else:
        ev(f"NO RESULT FILE for {name}")
    return done, state


# Resume: skip cells that already have a pulled result; VALIDATE=1 first
# re-runs folded 3 s (measured after a 600 s cooldown) to check the shorter one.
def _nominal_start(f):
    try:
        ts = json.loads(f.read_text())["records"][0].get("thermal_states") or []
        return bool(ts) and ts[0] == "nominal"
    except Exception:
        return False
# A cell counts as done only with a result that started at nominal thermal state.
done_cells = {tuple(f.name.split("_a")[0].rsplit("_", 1)) for f in OUT.glob("*_a*_*.json") if _nominal_start(f)}
STEPS = [s for s in STEPS if (s[1], s[0]) not in done_cells]
if os.environ.get("VALIDATE") == "1":
    STEPS.insert(0, ("3s", "folded"))
ev(f"COOL_S {COOL_S}; STEPS {len(STEPS)}: {STEPS}")
for key, arm in STEPS:
    ev(f"COOL {COOL_S}s before {arm} {key}")
    time.sleep(COOL_S)
    done, state = run_step(key, arm, 1)
    if not done:   # launch or device-connection failure (e.g. CoreDevice error 3): one retry
        ev(f"NOT DONE {arm} {key}; cooldown then one retry")
        time.sleep(COOL_S)
        done, state = run_step(key, arm, 2)
    if done and state not in (None, "nominal"):
        ev(f"NOT NOMINAL at start ({state}); extra cooldown then one retry")
        time.sleep(COOL_S)
        run_step(key, arm, 2)
ev("ALLDONE")
