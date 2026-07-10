#!/usr/bin/env python3
"""Probe a single Core ML package for DecoderPre + GeneratorFromHar.

The production Swift path currently calls two hot Core ML packages:

1. ``kokoro_decoder_pre_{bucket}s``: ``asr + f0 + n_input + ref_s -> x_pre``
2. ``kokoro_decoder_har_post_{bucket}s``: ``x_pre + ref_s + har -> waveform``

This probe preserves the exact Swift HAR/STFT boundary and tests whether
collapsing only the DecoderPre->Generator Core ML handoff improves warmed
short-bucket latency. It is intentionally not wired into production until parity
and warmed timing justify the extra package.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from audio_parity_tensor_io import load_tensor_dump  # noqa: E402
from probe_generator_exact_geometry import _compute_units, _load_kmodel, _metrics  # noqa: E402
from probe_generator_split import _duration_label_from_dump, _precision_arg, _remove_existing_package  # noqa: E402


def _make_decoder_pre_generator_module(decoder: Any):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class _DecoderPreGeneratorFromHar(nn.Module):
        def __init__(self, dec: Any):
            super().__init__()
            self.dec = dec

        def forward(
            self,
            asr: Any,
            f0: Any,
            n_input: Any,
            ref_s: Any,
            har: Any,
        ) -> Any:
            s = ref_s[:, :128]
            dec = self.dec
            gen = dec.generator

            f0_conv = dec.F0_conv(f0)
            n_conv = dec.N_conv(n_input)
            x = torch.cat([asr, f0_conv, n_conv], dim=1)
            x = dec.encode(x, s)
            asr_res = dec.asr_res(asr)
            res = True
            for block in dec.decode:
                if res:
                    x = torch.cat([x, asr_res, f0_conv, n_conv], dim=1)
                x = block(x, s)
                if block.upsample_type != "none":
                    res = False

            for i in range(gen.num_upsamples):
                x = F.leaky_relu(x, negative_slope=0.1)
                x_source = gen.noise_convs[i](har)
                x_source = gen.noise_res[i](x_source, s)
                x = gen.ups[i](x)
                if i == gen.num_upsamples - 1:
                    x = gen.reflection_pad(x)
                tx = x.size(2)
                ts = x_source.size(2)
                if ts < tx:
                    x_source = F.pad(x_source, (0, tx - ts))
                elif ts > tx:
                    x_source = x_source[:, :, :tx]
                x = x + x_source
                xs = None
                for j in range(gen.num_kernels):
                    y = gen.resblocks[i * gen.num_kernels + j](x, s)
                    xs = y if xs is None else xs + y
                x = xs / gen.num_kernels

            x = F.leaky_relu(x)
            x = gen.conv_post(x)
            spec = torch.exp(x[:, : gen.post_n_fft // 2 + 1, :])
            phase = torch.sin(x[:, gen.post_n_fft // 2 + 1 :, :])
            return gen.stft.inverse(spec, phase)

    return _DecoderPreGeneratorFromHar(decoder).eval()


def _predict(model: Any, feed: dict[str, np.ndarray]) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    out = model.predict(feed)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    waveform = np.asarray(out.get("waveform", next(iter(out.values())))).astype(np.float32)
    return waveform, elapsed_ms


def _input_shapes(model: Any) -> dict[str, tuple[int, ...]]:
    return {
        item.name: tuple(int(v) for v in item.type.multiArrayType.shape)
        for item in model.get_spec().description.input
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import coremltools as ct
    import torch

    from export_synth.wrappers import remove_dropout

    manifest, tensors = load_tensor_dump(args.tensor_dump)
    required = ["asr_padded", "f0_padded", "n_padded", "ref_s", "har_padded", "waveform"]
    missing = [name for name in required if name not in tensors]
    if missing:
        raise SystemExit(f"tensor dump missing required tensors: {missing}")

    label = args.label or _duration_label_from_dump(args.tensor_dump, manifest)
    work_dir = args.output_dir / label
    package = work_dir / f"kokoro_decoder_pre_generator_{label}.mlpackage"
    report_path = work_dir / args.report_name

    dec_pre_model = ct.models.MLModel(
        str(args.decoder_pre_package),
        compute_units=_compute_units(ct, args.baseline_decoder_pre_compute_units),
    )
    gen_model = ct.models.MLModel(
        str(args.generator_package),
        compute_units=_compute_units(ct, args.baseline_generator_compute_units),
    )
    dec_shapes = _input_shapes(dec_pre_model)
    gen_shapes = _input_shapes(gen_model)

    asr = tensors["asr_padded"].astype(np.float32)
    f0 = tensors["f0_padded"].astype(np.float32).reshape(dec_shapes["f0"])
    n_input = tensors["n_padded"].astype(np.float32).reshape(dec_shapes["n_input"])
    ref_s = tensors["ref_s"].astype(np.float32)
    har = tensors["har_padded"].astype(np.float32)

    if tuple(asr.shape) != dec_shapes["asr"]:
        raise SystemExit(f"asr shape {asr.shape} does not match decoder-pre {dec_shapes['asr']}")
    if tuple(ref_s.shape) != dec_shapes["ref_s"]:
        raise SystemExit(f"ref_s shape {ref_s.shape} does not match decoder-pre {dec_shapes['ref_s']}")
    if tuple(har.shape) != gen_shapes["har"]:
        raise SystemExit(f"har shape {har.shape} does not match generator {gen_shapes['har']}")

    if not args.skip_export:
        kmodel = _load_kmodel()
        module = _make_decoder_pre_generator_module(kmodel.decoder)
        removed_dropouts = remove_dropout(module)
        trace_inputs = (
            torch.zeros(tuple(asr.shape), dtype=torch.float32),
            torch.zeros(tuple(f0.shape), dtype=torch.float32),
            torch.zeros(tuple(n_input.shape), dtype=torch.float32),
            torch.zeros(tuple(ref_s.shape), dtype=torch.float32),
            torch.zeros(tuple(har.shape), dtype=torch.float32),
        )
        with torch.no_grad():
            traced = torch.jit.trace(
                module,
                trace_inputs,
                strict=False,
                check_trace=False,
            )
            traced_out = traced(*trace_inputs)
        traced_samples = int(traced_out.shape[-1])
        model = ct.convert(
            traced,
            inputs=[
                ct.TensorType(name="asr", shape=tuple(asr.shape), dtype=np.float32),
                ct.TensorType(name="f0", shape=tuple(f0.shape), dtype=np.float32),
                ct.TensorType(name="n_input", shape=tuple(n_input.shape), dtype=np.float32),
                ct.TensorType(name="ref_s", shape=tuple(ref_s.shape), dtype=np.float32),
                ct.TensorType(name="har", shape=tuple(har.shape), dtype=np.float32),
            ],
            outputs=[ct.TensorType(name="waveform")],
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.macOS13,
            compute_precision=_precision_arg(ct, args.precision),
            compute_units=ct.ComputeUnit.ALL,
        )
        _remove_existing_package(package)
        package.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(package))
    else:
        removed_dropouts = None
        traced_samples = None
        if not package.is_dir():
            raise SystemExit(f"--skip-export requested but package is missing: {package}")

    candidate = ct.models.MLModel(str(package), compute_units=_compute_units(ct, args.compute_units))

    dec_feed = {"asr": asr, "f0": f0, "n_input": n_input, "ref_s": ref_s}
    candidate_feed = {**dec_feed, "har": har}

    def predict_baseline() -> tuple[np.ndarray, float, float, float]:
        start = time.perf_counter()
        dec_start = time.perf_counter()
        dec_out = dec_pre_model.predict(dec_feed)
        dec_ms = (time.perf_counter() - dec_start) * 1000.0
        x_pre = np.asarray(dec_out["x_pre"], dtype=np.float32)
        gen_feed = {"x_pre": x_pre, "ref_s": ref_s, "har": har}
        waveform, gen_ms = _predict(gen_model, gen_feed)
        total_ms = (time.perf_counter() - start) * 1000.0
        return waveform, total_ms, dec_ms, gen_ms

    baseline_first, baseline_first_ms, baseline_dec_first_ms, baseline_gen_first_ms = predict_baseline()
    candidate_first, candidate_first_ms = _predict(candidate, candidate_feed)

    for _ in range(max(0, args.warmup)):
        predict_baseline()
        _predict(candidate, candidate_feed)

    baseline_times: list[float] = []
    baseline_dec_times: list[float] = []
    baseline_gen_times: list[float] = []
    candidate_times: list[float] = []
    last_baseline = baseline_first
    last_candidate = candidate_first
    for _ in range(max(1, args.iterations)):
        last_baseline, total_ms, dec_ms, gen_ms = predict_baseline()
        last_candidate, candidate_ms = _predict(candidate, candidate_feed)
        baseline_times.append(total_ms)
        baseline_dec_times.append(dec_ms)
        baseline_gen_times.append(gen_ms)
        candidate_times.append(candidate_ms)

    trim_len = min(int(tensors["waveform"].size), int(last_baseline.size), int(last_candidate.size))
    dump = tensors["waveform"].reshape(-1)[:trim_len]
    baseline_trim = last_baseline.reshape(-1)[:trim_len]
    candidate_trim = last_candidate.reshape(-1)[:trim_len]
    med_baseline = float(statistics.median(baseline_times))
    med_candidate = float(statistics.median(candidate_times))

    report = {
        "tensor_dump": str(args.tensor_dump),
        "label": label,
        "package": str(package),
        "decoder_pre_package": str(args.decoder_pre_package),
        "generator_package": str(args.generator_package),
        "precision": args.precision,
        "compute_units": args.compute_units,
        "baseline_decoder_pre_compute_units": args.baseline_decoder_pre_compute_units,
        "baseline_generator_compute_units": args.baseline_generator_compute_units,
        "removed_dropouts": removed_dropouts,
        "traced_samples": traced_samples,
        "manifest_metadata": manifest.get("metadata", {}),
        "shapes": {
            "asr": list(asr.shape),
            "f0": list(f0.shape),
            "n_input": list(n_input.shape),
            "ref_s": list(ref_s.shape),
            "har": list(har.shape),
            "trim_len": trim_len,
        },
        "first_predict_ms": {
            "baseline_total": float(baseline_first_ms),
            "baseline_decoder_pre": float(baseline_dec_first_ms),
            "baseline_generator": float(baseline_gen_first_ms),
            "candidate_merged": float(candidate_first_ms),
        },
        "warm_predict_times_ms": {
            "baseline_total": baseline_times,
            "baseline_decoder_pre": baseline_dec_times,
            "baseline_generator": baseline_gen_times,
            "candidate_merged": candidate_times,
        },
        "warm_predict_median_ms": {
            "baseline_total": med_baseline,
            "baseline_decoder_pre": float(statistics.median(baseline_dec_times)),
            "baseline_generator": float(statistics.median(baseline_gen_times)),
            "candidate_merged": med_candidate,
        },
        "speedup_vs_two_prediction_pct": (med_baseline - med_candidate) / med_baseline * 100.0,
        "metrics_vs_dump": {
            "baseline_two_prediction": _metrics(dump, baseline_trim),
            "candidate_merged": _metrics(dump, candidate_trim),
        },
        "metrics_vs_baseline_two_prediction": _metrics(baseline_trim, candidate_trim),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor-dump", type=Path, required=True)
    parser.add_argument("--decoder-pre-package", type=Path, required=True)
    parser.add_argument("--generator-package", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/decoder_pre_generator_merge"))
    parser.add_argument("--report-name", default="report.json")
    parser.add_argument("--label", default=None)
    parser.add_argument("--precision", default="fp16", choices=["fp16", "float16", "fp32", "float32"])
    parser.add_argument("--compute-units", default="cpuAndGPU")
    parser.add_argument("--baseline-decoder-pre-compute-units", default="cpuAndNeuralEngine")
    parser.add_argument("--baseline-generator-compute-units", default="cpuAndGPU")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--skip-export", action="store_true")
    args = parser.parse_args()

    report = run(args)
    print(json.dumps(report["warm_predict_median_ms"], indent=2, sort_keys=True))
    print(f"speedup_vs_two_prediction_pct={report['speedup_vs_two_prediction_pct']:.3f}")
    print(f"report={args.output_dir / (args.label or report['label']) / args.report_name}")


if __name__ == "__main__":
    main()
