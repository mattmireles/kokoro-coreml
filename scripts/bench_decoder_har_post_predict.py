#!/usr/bin/env python3
"""Median Core ML ``predict()`` wall time for ``kokoro_decoder_har_post_*s`` (Phase 3 fallback).

Uses the same tensor prep as production (:func:`~kokoro.synthesis_backends.build_decoder_har_post_inputs_np`).
Maps to bakeoff schema field ``t_coreml_predict_s`` (this script reports milliseconds per call).

Examples::

    uv run python scripts/bench_decoder_har_post_predict.py \\
      --package coreml/kokoro_decoder_har_post_3s.mlpackage --bucket-sec 3

    uv run python scripts/bench_decoder_har_post_predict.py \\
      --preset long --warmup 1 --iterations 11

    uv run python scripts/bench_decoder_har_post_predict.py --all-presets

    uv run python scripts/bench_decoder_har_post_predict.py \\
      --package coreml/kokoro_decoder_har_post_3s.mlpackage \\
      --baseline /path/to/pre_conv1d_3s.mlpackage \\
      --bucket-sec 3 --json-out outputs/bakeoff/ane_optimization_results.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

import coremltools as ct

from kokoro.coreml_pipeline import HybridTTSPipeline
from kokoro.synthesis_backends import build_decoder_har_post_inputs_np

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Bakeoff-style lengths (README/Plans/003-kokoro-bakeoff-plan.md). Actual F0 seconds and Core ML bucket
# come from :meth:`HybridTTSPipeline._select_bucket_seconds` (smallest loaded bucket >= ceil(duration)).
# With only {3s, 10s} decoder_har_post buckets, ~3.25s audio still maps to the 10s package.
BAKEOFF_INPUT_PRESETS: dict[str, str] = {
    "tiny": "Hello world!",
    "short": "The quick brown fox jumps over the lazy dog.",
    "medium": (
        "This is a longer sentence that will test the performance of our pipeline "
        "running on the Apple GPU."
    ),
    "long": (
        "This is a longer sentence that will test the performance of our pipeline "
        "running on the Apple GPU. More text at the end pushes nine seconds."
    ),
}


def _preset_bucket_and_text(
    pipe: HybridTTSPipeline, preset: str, voice: str, speed: float
) -> tuple[str, int, float]:
    if preset not in BAKEOFF_INPUT_PRESETS:
        raise SystemExit(f"unknown --preset {preset!r}; choose one of {sorted(BAKEOFF_INPUT_PRESETS)}")
    text = BAKEOFF_INPUT_PRESETS[preset]
    vi = pipe.extract_vocoder_inputs(text, voice, speed)
    if vi is None:
        raise RuntimeError("extract_vocoder_inputs returned None")
    t_f0 = int(vi["f0_curve"].shape[-1])
    total_seconds = t_f0 / 80.0
    selected = pipe._select_bucket_seconds(total_seconds)
    if selected is None or selected not in pipe.coreml_decoder_har_post_buckets:
        raise RuntimeError("no decoder_har_post bucket for pipeline / utterance")
    return text, selected, total_seconds


def _inputs_for_package(
    pipe: HybridTTSPipeline,
    mlpackage: Path,
    text: str,
    voice: str,
    speed: float,
    bucket_sec: int,
) -> dict:
    model = ct.models.MLModel(str(mlpackage))
    spec = model.get_spec()
    shapes = {i.name: list(i.type.multiArrayType.shape) for i in spec.description.input}
    asr_len = int(shapes["x_pre"][-1])
    har_t = int(shapes["har"][-1])

    vi = pipe.extract_vocoder_inputs(text, voice, speed)
    if vi is None:
        raise RuntimeError("extract_vocoder_inputs returned None")
    T_f0 = int(vi["f0_curve"].shape[-1])
    total_seconds = T_f0 / 80.0
    selected = pipe._select_bucket_seconds(total_seconds)
    if selected is None or selected not in pipe.coreml_decoder_har_post_buckets:
        raise RuntimeError("no decoder_har_post bucket for pipeline / utterance")
    if selected != bucket_sec:
        raise RuntimeError(
            f"pipeline selects {selected}s for this text; use --bucket-sec {selected}"
        )
    dec = pipe.pytorch_model.decoder
    x_pre, ref_s, har, t_chk, _fc = build_decoder_har_post_inputs_np(
        dec, vi, bucket_sec, asr_len, har_t, warn_geometry=True
    )
    _ = t_chk
    return {"x_pre": x_pre, "ref_s": ref_s, "har": har}


def _median_predict_ms(model: ct.models.MLModel, inputs: dict, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        _ = model.predict(inputs)
    samples: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = model.predict(inputs)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return float(statistics.median(samples))


def _run_one(
    pipe: HybridTTSPipeline,
    package: Path,
    baseline: Path | None,
    bucket_sec: int,
    text: str,
    voice: str,
    speed: float,
    warmup: int,
    iterations: int,
    json_out: Path | None,
    extra_doc: dict | None = None,
) -> dict:
    inputs = _inputs_for_package(pipe, package, text, voice, speed, bucket_sec)
    cand = ct.models.MLModel(str(package))
    med_c = _median_predict_ms(cand, inputs, warmup, iterations)
    print(f"candidate_median_ms={med_c:.3f}  package={package}")

    doc: dict = {
        "benchmark_mode": "fallback_loop",
        "t_coreml_predict_median_ms": med_c,
        "warmup": warmup,
        "iterations": iterations,
        "bucket_sec": bucket_sec,
        "package": str(package.resolve()),
        "text_preview": text[:120] + ("…" if len(text) > 120 else ""),
    }
    if extra_doc:
        doc.update(extra_doc)

    if baseline is not None:
        base = ct.models.MLModel(str(baseline))
        med_b = _median_predict_ms(base, inputs, warmup, iterations)
        doc["baseline_mlpackage"] = str(baseline.resolve())
        doc["baseline_median_ms"] = med_b
        if med_b > 0:
            doc["speedup_vs_baseline_pct"] = round(100.0 * (med_b - med_c) / med_b, 2)
        print(f"baseline_median_ms={med_b:.3f}  package={baseline}")
        if "speedup_vs_baseline_pct" in doc:
            print(f"speedup_vs_baseline_pct={doc['speedup_vs_baseline_pct']}")

    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        prev = {}
        if json_out.is_file():
            try:
                prev = json.loads(json_out.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        if "presets" not in prev or not isinstance(prev.get("presets"), dict):
            prev["presets"] = {}
        preset_key = doc.get("bakeoff_preset")
        if preset_key:
            prev["presets"][preset_key] = doc
        prev.update(doc)
        json_out.write_text(json.dumps(prev, indent=2), encoding="utf-8")
        print(f"wrote {json_out}")

    return doc


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--package", type=Path, default=None, help="Candidate .mlpackage path (default: coreml/… from --bucket-sec)")
    p.add_argument("--baseline", type=Path, default=None, help="Optional baseline .mlpackage for A/B median")
    p.add_argument("--bucket-sec", type=int, default=None, choices=(3, 10))
    p.add_argument("--text", type=str, default="Hello from Kokoro.")
    p.add_argument(
        "--preset",
        type=str,
        default=None,
        choices=sorted(BAKEOFF_INPUT_PRESETS),
        help="Use bakeoff-aligned text; infers --bucket-sec and default --package",
    )
    p.add_argument(
        "--all-presets",
        action="store_true",
        help=f"Run {sorted(BAKEOFF_INPUT_PRESETS)} with warmup=1 iterations=7 unless overridden",
    )
    p.add_argument("--voice", type=str, default="af_heart")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iterations", type=int, default=21)
    p.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Merge results into this JSON file (creates parent dirs)",
    )
    args = p.parse_args()

    if args.preset and args.all_presets:
        raise SystemExit("use either --preset or --all-presets, not both")
    if args.preset and args.text != "Hello from Kokoro.":
        raise SystemExit("--preset replaces --text; omit --text when using --preset")

    torch.manual_seed(0)
    pipe = HybridTTSPipeline()

    if args.all_presets:
        w, it = args.warmup, args.iterations
        if args.warmup == 3 and args.iterations == 21:
            w, it = 1, 7
        rows: list[tuple[str, int, float, float]] = []
        for preset in ("tiny", "short", "medium", "long"):
            text, bucket_sec, approx_sec = _preset_bucket_and_text(pipe, preset, args.voice, args.speed)
            pkg = args.package or (_REPO_ROOT / f"coreml/kokoro_decoder_har_post_{bucket_sec}s.mlpackage")
            if not pkg.is_dir():
                raise SystemExit(f"missing package {pkg}")
            print(f"\n=== preset={preset} approx_f0_sec={approx_sec:.3f} bucket_sec={bucket_sec} ===")
            doc = _run_one(
                pipe,
                pkg,
                args.baseline,
                bucket_sec,
                text,
                args.voice,
                args.speed,
                w,
                it,
                None,
                extra_doc={
                    "bakeoff_preset": preset,
                    "approx_f0_seconds": round(approx_sec, 4),
                },
            )
            rows.append((preset, bucket_sec, approx_sec, float(doc["t_coreml_predict_median_ms"])))
        if args.json_out is not None:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            prev = {}
            if args.json_out.is_file():
                try:
                    prev = json.loads(args.json_out.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    pass
            prev["all_presets_run"] = [
                {
                    "preset": r[0],
                    "bucket_sec": r[1],
                    "approx_f0_seconds": round(r[2], 4),
                    "t_coreml_predict_median_ms": r[3],
                }
                for r in rows
            ]
            args.json_out.write_text(json.dumps(prev, indent=2), encoding="utf-8")
            print(f"wrote {args.json_out}")
        return 0

    if args.preset:
        text, bucket_sec, approx_sec = _preset_bucket_and_text(pipe, args.preset, args.voice, args.speed)
        package = args.package or (_REPO_ROOT / f"coreml/kokoro_decoder_har_post_{bucket_sec}s.mlpackage")
        if args.bucket_sec is not None and args.bucket_sec != bucket_sec:
            raise SystemExit(
                f"--preset {args.preset} selects {bucket_sec}s bucket (~{approx_sec:.3f}s f0 span); "
                f"omit --bucket-sec or pass {bucket_sec}"
            )
    else:
        text = args.text
        bucket_sec = args.bucket_sec
        package = args.package
        approx_sec = None
        if bucket_sec is None:
            raise SystemExit("pass --bucket-sec or use --preset / --all-presets")
        if package is None:
            package = _REPO_ROOT / f"coreml/kokoro_decoder_har_post_{bucket_sec}s.mlpackage"

    if not package.is_dir():
        raise SystemExit(f"missing package {package}")

    _run_one(
        pipe,
        package,
        args.baseline,
        bucket_sec,
        text,
        args.voice,
        args.speed,
        args.warmup,
        args.iterations,
        args.json_out,
        extra_doc=(
            {
                "bakeoff_preset": args.preset,
                "approx_f0_seconds": round(approx_sec, 4),
            }
            if args.preset and approx_sec is not None
            else None
        ),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
