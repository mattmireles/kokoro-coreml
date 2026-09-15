#!/usr/bin/env python3
"""Benchmark the individual CoreML stages of the Swift prefix rewrite pipeline.

Measures predict latency for each CoreML model in the chain, plus the Python
DecoderPre bridge time. Combined with Swift hn-nsf timing from the Swift test
suite, this gives the complete latency budget.

Usage::

    uv run python scripts/bench_swift_stages.py

Measured stages:
    1. Duration CoreML predict
    2. F0Ntrain CoreML predict (per bucket)
    3. DecoderPre PyTorch bridge (per input)
    4. GeneratorFromHar CoreML predict (per bucket)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent


def _bench_model(model, inputs: dict, name: str, n_warmup: int = 5, n_iter: int = 20) -> dict:
    """Benchmark a CoreML model predict call."""
    # Warmup
    for _ in range(n_warmup):
        model.predict(inputs)

    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        model.predict(inputs)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    times.sort()
    median = times[len(times) // 2]
    return {
        "name": name,
        "median_ms": round(median, 2),
        "mean_ms": round(sum(times) / len(times), 2),
        "min_ms": round(times[0], 2),
        "max_ms": round(times[-1], 2),
        "n_iter": n_iter,
    }


def main():
    import coremltools as ct

    coreml_dir = _ROOT / "coreml"
    results = {}

    # --- Duration CoreML ---
    dur_path = coreml_dir / "kokoro_duration.mlpackage"
    if dur_path.exists():
        print("Benchmarking Duration CoreML...")
        dur_model = ct.models.MLModel(str(dur_path), compute_units=ct.ComputeUnit.ALL)
        dur_inputs = {
            "input_ids": np.zeros((1, 128), dtype=np.int32),
            "attention_mask": np.ones((1, 128), dtype=np.int32),
            "ref_s": np.random.randn(1, 256).astype(np.float32),
            "speed": np.array([1.0], dtype=np.float32),
        }
        results["duration"] = _bench_model(dur_model, dur_inputs, "Duration CoreML")
        print(f"  Duration: {results['duration']['median_ms']:.2f} ms median")
    else:
        print(f"  Duration model not found at {dur_path}")

    # --- F0Ntrain CoreML ---
    for t_frames in [120, 400]:
        f0n_path = coreml_dir / f"kokoro_f0ntrain_t{t_frames}.mlpackage"
        if f0n_path.exists():
            print(f"\nBenchmarking F0Ntrain T={t_frames}...")
            f0n_model = ct.models.MLModel(str(f0n_path), compute_units=ct.ComputeUnit.ALL)
            f0n_inputs = {
                "en": np.random.randn(1, 640, t_frames).astype(np.float32),
                "s": np.random.randn(1, 128).astype(np.float32),
            }
            if any(item.name == "mask" for item in f0n_model.get_spec().description.input):
                f0n_inputs["mask"] = np.ones((1, 1, t_frames), dtype=np.float32)
            results[f"f0ntrain_t{t_frames}"] = _bench_model(
                f0n_model, f0n_inputs, f"F0Ntrain T={t_frames}"
            )
            print(f"  F0Ntrain T={t_frames}: {results[f'f0ntrain_t{t_frames}']['median_ms']:.2f} ms median")
        else:
            print(f"  F0Ntrain T={t_frames} not found at {f0n_path}")

    # --- GeneratorFromHar CoreML ---
    for sec in [3, 10]:
        gen_path = coreml_dir / f"kokoro_decoder_har_post_{sec}s.mlpackage"
        if gen_path.exists():
            print(f"\nBenchmarking GeneratorFromHar {sec}s...")
            gen_model = ct.models.MLModel(str(gen_path), compute_units=ct.ComputeUnit.ALL)
            spec = gen_model.get_spec()
            shapes = {i.name: [int(d) for d in i.type.multiArrayType.shape]
                      for i in spec.description.input}
            gen_inputs = {}
            for name, shape in shapes.items():
                gen_inputs[name] = (
                    np.ones(shape, dtype=np.float32)
                    if name == "mask"
                    else np.random.randn(*shape).astype(np.float32)
                )
            results[f"generator_{sec}s"] = _bench_model(
                gen_model, gen_inputs, f"GeneratorFromHar {sec}s"
            )
            print(f"  GeneratorFromHar {sec}s: {results[f'generator_{sec}s']['median_ms']:.2f} ms median")
        else:
            print(f"  GeneratorFromHar {sec}s not found at {gen_path}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("STAGE LATENCY SUMMARY (warm median, M2 Ultra)")
    print("=" * 60)
    for key, val in results.items():
        print(f"  {val['name']:30s}  {val['median_ms']:8.2f} ms")

    # Estimate total pipeline time
    dur_ms = results.get("duration", {}).get("median_ms", 0)
    f0n_3s = results.get("f0ntrain_t120", {}).get("median_ms", 0)
    f0n_10s = results.get("f0ntrain_t400", {}).get("median_ms", 0)
    gen_3s = results.get("generator_3s", {}).get("median_ms", 0)
    gen_10s = results.get("generator_10s", {}).get("median_ms", 0)

    print("\n  ESTIMATED PIPELINE (excluding DecoderPre bridge + hn-nsf Swift):")
    print(f"  3s bucket:  Duration({dur_ms:.1f}) + F0Ntrain({f0n_3s:.1f}) + Generator({gen_3s:.1f}) = {dur_ms+f0n_3s+gen_3s:.1f} ms")
    print(f"  10s bucket: Duration({dur_ms:.1f}) + F0Ntrain({f0n_10s:.1f}) + Generator({gen_10s:.1f}) = {dur_ms+f0n_10s+gen_10s:.1f} ms")
    print("  + alignment/padding (~1ms) + hn-nsf Swift (TBD) + DecoderPre bridge (~40-100ms)")

    # Save results
    out_path = _ROOT / "outputs" / "swift_prefix_stage_bench.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
