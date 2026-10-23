#!/usr/bin/env python3
"""Plan 014 summary: per-cell stats, folded-vs-hybrid ties, ablation, CSV.

usage: summarize_014.py [OUTDIR]   (run from outputs/frozen_rerun_014)
Phones: newest result per (arm, bucket) in iphone-*/ (superseded/ ignored),
only cells whose first thermal state is nominal. Macs: <host>/<arm>.json from
run_config_f_reference.py / run_mlx_audio.py, load from run_log.jsonl.
Tie rules (README/Plans/014): warm medians within 5% or overlapping min-max
ranges; also a noise-floor tie when the 30 s gap is below the device's 3 s
same-pipeline gap (both arms run folded at 3 s).
"""
import csv
import json
import re
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
PHONES = {"iphone-12-pro": "A14 (iPhone 12 Pro)", "iphone-15-pro-max": "A17 Pro (iPhone 15 Pro Max)"}
MACS = {"m2-studio": "M2 Ultra", "m2-air": "M2 Air", "irvine-m1": "M1 Mini"}
KEYS = ["3s", "7s", "10s", "15s", "30s"]


def cell(times, cold, peak=None, extra=None):
    t = [x * 1000 for x in times]
    return {"n": len(t), "median_ms": round(st.median(t), 1), "min_ms": round(min(t), 1), "max_ms": round(max(t), 1),
            "cold_ms": round(cold * 1000, 1) if cold is not None else None,
            "peak_mb": round(peak, 1) if peak else None, "warm_ms": [round(x, 2) for x in t], **(extra or {})}


def phone_cells(d):
    cells = {}
    for f in sorted((HERE / d).glob("*_a*_*.json")):
        arm, key = f.name.split("_a")[0].rsplit("_", 1)
        r = json.loads(f.read_text())["records"][0]
        ts = r.get("thermal_states") or []
        if r.get("error") or not ts or ts[0] != "nominal":
            continue
        times = r.get("warm_times_s")
        if isinstance(times, str):
            times = json.loads(times)
        cells[(arm, key)] = cell(times, r.get("cold_s"), r.get("peak_footprint_mb"),
                                 {"file": f"{d}/{f.name}", "thermal_first": ts[0], "thermal_last": ts[-1]})
    return cells


def mac_cells(h):
    cells, state = {}, {}
    log = HERE / h / "run_log.jsonl"
    if log.exists():
        for line in log.read_text().splitlines():
            e = json.loads(line)
            if e["phase"] in ("start", "gate_pass", "gate_fail"):
                s = state.setdefault(e["arm"], {})
                if e["phase"] == "start":
                    s["load_at_start"] = e["loadavg"]
                    s["t_start_utc"] = e["t"]
                else:
                    s["gate"] = e["phase"]
    for f in (HERE / h).glob("*.json"):
        arm = f.stem
        if arm.startswith("botnet") or arm.startswith("results"):
            continue
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        for r in d.get("records", []):
            if r.get("error") or not r.get("warm_wall_times_s"):
                cells[(arm, r.get("input_key"))] = {"error": r.get("error")}
                continue
            cells[(arm, r["input_key"])] = cell(r["warm_wall_times_s"], r.get("cold_wall_time_s"), None,
                                                {"file": f"{h}/{f.name}", **state.get(arm, {})})
    return cells


def verdict(c, dev_noise):
    out = {}
    for key in KEYS[1:]:
        f, h = c.get(("folded", key)), c.get(("hybrid", key))
        if not f or not h or "median_ms" not in f or "median_ms" not in h:
            continue
        diff = h["median_ms"] - f["median_ms"]
        pct = 100 * diff / f["median_ms"]
        overlap = not (h["max_ms"] < f["min_ms"] or f["max_ms"] < h["min_ms"])
        tie5 = abs(pct) <= 5 or overlap
        noise_tie = dev_noise is not None and abs(pct) < dev_noise
        out[key] = {"folded_ms": f["median_ms"], "hybrid_ms": h["median_ms"], "diff_ms": round(diff, 1), "diff_pct": round(pct, 1),
                    "ranges_overlap": overlap, "tie_5pct_rule": tie5, "noise_floor_pct": dev_noise, "tie_noise_rule": noise_tie,
                    "faster": "tie" if (tie5 or noise_tie) else ("hybrid" if diff < 0 else "folded"),
                    "folded_peak_mb": f.get("peak_mb"), "hybrid_peak_mb": h.get("peak_mb")}
    return out


def main():
    summary = {"devices": {}}
    rows = []
    for d, name in {**PHONES, **MACS}.items():
        if not (HERE / d).exists():
            continue
        c = phone_cells(d) if d in PHONES else mac_cells(d)
        f3, h3 = c.get(("folded", "3s")), c.get(("hybrid", "3s"))
        noise = round(100 * abs(h3["median_ms"] - f3["median_ms"]) / f3["median_ms"], 1) if f3 and h3 and "median_ms" in f3 and "median_ms" in h3 else None
        dev = {"name": name, "cells": {f"{a}|{k}": v for (a, k), v in sorted(c.items())},
               "same_pipeline_3s_gap_pct": noise, "folded_vs_hybrid": verdict(c, noise)}
        abl = {}
        for key in ("3s", "30s"):
            ladder = [("R0", "R0"), ("R1", "R1"), ("R2", "R2"), ("R3 folded", "folded"), ("R4 hybrid", "hybrid")]
            abl[key] = {rung: c[(arm, key)]["median_ms"] for rung, arm in ladder if "median_ms" in c.get((arm, key), {})}
        if any(len(v) > 2 for v in abl.values()):
            dev["ablation"] = abl
        summary["devices"][d] = dev
        for (a, k), v in sorted(c.items()):
            for i, t in enumerate(v.get("warm_ms", [])):
                rows.append([d, a, k, i, t, v.get("cold_ms"), v.get("peak_mb"), v.get("file")])
    (OUT / "summary_014.json").write_text(json.dumps(summary, indent=1))
    with open(OUT / "per_call_timings_014.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["device", "arm", "bucket", "warm_index", "warm_ms", "cold_ms", "peak_mb", "source"])
        w.writerows(rows)
    for d, dev in summary["devices"].items():
        print(f"\n## {dev['name']} ({d}); 3 s same-pipeline gap {dev['same_pipeline_3s_gap_pct']}%")
        arms = sorted({k.split("|")[0] for k in dev["cells"]})
        print("arm".ljust(10) + "".join(k.rjust(22) for k in KEYS))
        for a in arms:
            line = a.ljust(10)
            for k in KEYS:
                v = dev["cells"].get(f"{a}|{k}")
                line += (f"{v['median_ms']} [{v['min_ms']}-{v['max_ms']}] c{v['cold_ms']}" if v and "median_ms" in v else ("err" if v else "-")).rjust(22)
            print(line)
        for k, v in dev["folded_vs_hybrid"].items():
            print(f"  {k}: folded {v['folded_ms']} hybrid {v['hybrid_ms']} ({v['diff_pct']:+}%) overlap={v['ranges_overlap']} -> {v['faster']}")
        if "ablation" in dev:
            print("  ablation:", dev["ablation"])


if __name__ == "__main__":
    main()
