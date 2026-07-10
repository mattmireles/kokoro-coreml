#!/usr/bin/env python3
"""Profile per-stage latency for ``laishere/kokoro-coreml``.

The standard external bakeoff adapter intentionally matches laishere's public
benchmark boundary: it times the seven Core ML model calls after phonemization
and feed preparation have already happened. This script keeps that same chain
boundary, but records stage timings so we can identify which graph/runtime
choice makes laishere faster than the local HAR-post path on some machines.
"""

from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.external_bakeoff.run_laishere_kokoro_coreml import (  # noqa: E402
    MODEL_NAMES,
    SR,
    LaishereCoreMLChain,
)
from scripts.external_bakeoff.schema import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    DEFAULT_VOICE,
    RUNTIME_BUCKETS,
    load_json,
    machine_info,
    sha256_bytes,
    sha256_text,
    utc_now,
    validate_manifest,
    write_json,
)

STAGES = (
    "albert",
    "post_albert",
    "alignment",
    "prosody",
    "noise",
    "vocoder",
    "tail",
)


def _time_call(fn: Callable[[], Any]) -> tuple[Any, float]:
    start = time.perf_counter()
    value = fn()
    return value, time.perf_counter() - start


def _profile_once(chain: LaishereCoreMLChain, prepared: dict[str, Any]) -> dict[str, Any]:
    """Run one laishere chain pass and return total, per-stage times, and output."""

    stage_times: dict[str, float] = {}
    total_start = time.perf_counter()

    o1, stage_times["albert"] = _time_call(
        lambda: chain.m_albert.predict(
            {
                "input_ids": prepared["input_ids"],
                "attention_mask": prepared["attention_mask"],
            }
        )
    )
    o2, stage_times["post_albert"] = _time_call(
        lambda: chain.m_post.predict(
            {
                "bert_dur": np.array(o1["bert_dur"]).astype(np.float16),
                "input_ids": prepared["input_ids"],
                "style_s": prepared["style_s"],
                "speed": np.array([1.0], dtype=np.float16),
                "attention_mask": prepared["attention_mask"],
            }
        )
    )
    duration = np.array(o2["duration"]).flatten()
    pred_dur = np.round(duration).clip(min=1).astype(np.int32).reshape(1, -1)
    t_a = int(pred_dur.sum())
    if t_a > chain.max_frames:
        raise ValueError(f"T_a={t_a} exceeds converted max_frames={chain.max_frames}")

    o3, stage_times["alignment"] = _time_call(
        lambda: chain.m_align.predict(
            {
                "pred_dur": pred_dur,
                "d": np.array(o2["d"]).astype(np.float16),
                "t_en": np.array(o2["t_en"]).astype(np.float16),
            }
        )
    )
    o4, stage_times["prosody"] = _time_call(
        lambda: chain.m_pros.predict(
            {
                "en": np.array(o3["en"]).astype(np.float16),
                "style_s": prepared["style_s"],
            }
        )
    )
    o5, stage_times["noise"] = _time_call(
        lambda: chain.m_noise.predict(
            {
                "F0_curve": np.array(o4["F0"]).astype(np.float32),
                "style_timbre": prepared["style_timbre_f32"],
            }
        )
    )
    o6, stage_times["vocoder"] = _time_call(
        lambda: chain.m_voc.predict(
            {
                "asr": np.array(o3["asr"]).astype(np.float16),
                "F0_curve": np.array(o4["F0"]).astype(np.float16),
                "N_pred": np.array(o4["N"]).astype(np.float16),
                "x_source_0": np.array(o5["x_source_0"]).astype(np.float16),
                "x_source_1": np.array(o5["x_source_1"]).astype(np.float16),
                "style_timbre": prepared["style_timbre_f16"],
            }
        )
    )
    o7, stage_times["tail"] = _time_call(
        lambda: chain.m_tail.predict({"x_pre": np.array(o6["x_pre"]).astype(np.float32)})
    )

    total_s = time.perf_counter() - total_start
    audio = np.array(o7["audio"]).flatten().astype(np.float32)
    stage_sum_s = float(sum(stage_times.values()))
    return {
        "total_s": total_s,
        "stage_times_s": stage_times,
        "stage_sum_s": stage_sum_s,
        "python_overhead_s": total_s - stage_sum_s,
        "audio": audio,
        "t_a": t_a,
    }


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _summarize_runs(runs: list[dict[str, Any]]) -> dict[str, float]:
    summary = {
        "total_s": _median([float(run["total_s"]) for run in runs]),
        "stage_sum_s": _median([float(run["stage_sum_s"]) for run in runs]),
        "python_overhead_s": _median([float(run["python_overhead_s"]) for run in runs]),
    }
    for stage in STAGES:
        summary[f"{stage}_s"] = _median([float(run["stage_times_s"][stage]) for run in runs])
    return summary


def _public_run(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "total_s": round(float(run["total_s"]), 6),
        "stage_sum_s": round(float(run["stage_sum_s"]), 6),
        "python_overhead_s": round(float(run["python_overhead_s"]), 6),
        "stage_times_s": {
            stage: round(float(run["stage_times_s"][stage]), 6)
            for stage in STAGES
        },
        "t_a": int(run["t_a"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_OUTPUT_DIR / "runtime_input_manifest.json")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--machine-id", required=True)
    parser.add_argument("--laishere-repo", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, default=None)
    parser.add_argument("--voice", default=DEFAULT_VOICE)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--input-key", action="append", default=None)
    parser.add_argument("--max-frames", type=int, default=2000)
    args = parser.parse_args()

    manifest = load_json(args.manifest)
    validate_manifest(manifest)
    keys = args.input_key or list(RUNTIME_BUCKETS)
    unknown = sorted(set(keys) - set(RUNTIME_BUCKETS))
    if unknown:
        raise SystemExit(f"unknown input keys: {', '.join(unknown)}")

    repo = args.laishere_repo.resolve()
    models_dir = (args.models_dir or (repo / "output")).resolve()
    missing = [name for name in MODEL_NAMES if not (models_dir / name).exists()]
    if missing:
        raise SystemExit(f"missing laishere Core ML packages: {', '.join(missing)}")
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()

    chain = LaishereCoreMLChain(models_dir=models_dir, voice=args.voice, max_frames=args.max_frames)
    records: list[dict[str, Any]] = []
    for key in keys:
        item = manifest["inputs"][key]
        prepare_start = time.perf_counter()
        prepared = chain.prepare(item["text"])
        prepare_wall_time_s = time.perf_counter() - prepare_start

        cold = _profile_once(chain, prepared)
        for _ in range(args.warmup):
            _profile_once(chain, prepared)
        warm_runs = [_profile_once(chain, prepared) for _ in range(args.iterations)]
        last_audio = warm_runs[-1]["audio"]
        observed_s = float(last_audio.size) / float(SR)
        medians = _summarize_runs(warm_runs)
        records.append(
            {
                "input_key": key,
                "text_sha256": sha256_text(item["text"]),
                "voice": args.voice,
                "canonical_audio_duration_s": round(float(item["canonical_duration_s"]), 6),
                "observed_audio_duration_s": round(observed_s, 6),
                "rtf_observed_median": round(medians["total_s"] / observed_s, 6) if observed_s > 0 else None,
                "output_sha256": sha256_bytes(last_audio.astype(np.float32).tobytes()),
                "prepare_wall_time_s": round(prepare_wall_time_s, 6),
                "cold": _public_run(cold),
                "warm_median_s": {k: round(v, 6) for k, v in medians.items()},
                "warm_runs": [_public_run(run) for run in warm_runs],
                "provenance": {
                    "laishere_repo": str(repo),
                    "models_dir": str(models_dir),
                    "max_frames": args.max_frames,
                    "sample_rate": SR,
                    "t_enc": prepared["t_enc"],
                    "phoneme_count": len(prepared["phonemes"]),
                    "timing_boundary": "Laishere seven Core ML stage chain; prepare_wall_time_s is reported but excluded from warm_median_s.",
                    "compute_units": {
                        "albert": "CPU_AND_NE",
                        "post_albert": "CPU_AND_NE",
                        "alignment": "CPU_AND_NE",
                        "prosody": "CPU_AND_NE",
                        "noise": "ALL",
                        "vocoder": "CPU_AND_NE",
                        "tail": "ALL",
                    },
                },
            }
        )

    payload = {
        "created_utc": utc_now(),
        "impl": "laishere-kokoro-coreml-stage-profile",
        "machine_id": args.machine_id,
        "machine": machine_info(args.machine_id),
        "records": records,
        "provenance": {
            "laishere_sha": sha,
            "models_dir": str(models_dir),
            "iterations": args.iterations,
            "warmup": args.warmup,
            "profile_stages": STAGES,
        },
    }
    output = args.output or (
        DEFAULT_OUTPUT_DIR / "placement" / f"results_laishere_stage_profile_{args.machine_id}.json"
    )
    write_json(output, payload)
    print(f"wrote {output}")
    for record in records:
        med = record["warm_median_s"]
        stage_summary = ", ".join(f"{stage}={med[f'{stage}_s'] * 1000.0:.1f}ms" for stage in STAGES)
        print(f"{record['input_key']}: total={med['total_s'] * 1000.0:.1f}ms; {stage_summary}")


if __name__ == "__main__":
    main()
