#!/usr/bin/env python3
"""Probe a split ``GeneratorFromHar`` Core ML pipeline.

The shipping HAR-post generator package fuses two different jobs:

1. The large style-conditioned residual/noise convolution stack.
2. The final waveform tail: ``exp`` + ``sin`` + export-friendly iSTFT.

This script exports those pieces as separate temporary packages under
``outputs/`` and compares sequential ``body.predict`` + ``tail.predict``
against the current fused ``kokoro_decoder_har_post_*s.mlpackage`` on the same
Swift tensor dump. It is a scheduling experiment only; it does not modify
shipping packages.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_ROOT))

from audio_parity_tensor_io import load_tensor_dump  # noqa: E402
from probe_generator_exact_geometry import _compute_units, _load_kmodel, _metrics  # noqa: E402


class GeneratorBodyFromHar:
    """Factory namespace for the PyTorch body wrapper.

    The actual ``torch.nn.Module`` class is created lazily inside
    :func:`_make_body_module` so importing this script stays cheap until export
    is requested.
    """


def _duration_label_from_dump(path: Path, manifest: dict[str, Any]) -> str:
    value = manifest.get("metadata", {}).get("input_key")
    if value:
        return str(value).replace("/", "_")
    return path.name.replace("/", "_")


def _precision_arg(ct: Any, precision: str):
    value = precision.strip().lower()
    if value in {"fp32", "float32"}:
        return ct.precision.FLOAT32
    if value in {"fp16", "float16"}:
        return ct.precision.FLOAT16
    raise ValueError(f"unsupported precision {precision!r}")


def _predict_inputs(tensors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "x_pre": tensors["x_pre_padded"].astype(np.float32),
        "ref_s": tensors["ref_s"].astype(np.float32),
        "har": tensors["har_padded"].astype(np.float32),
    }


def _make_body_module(generator: Any):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class _GeneratorBodyFromHar(nn.Module):
        """Generator conv/noise/residual body ending at pre-tail vocoder logits.

        Inputs match ``GeneratorFromHar``:
        - ``x_pre``: decoder-pre output, ``(B, 512, T_asr)``
        - ``ref_s``: full style embedding, ``(B, 256)``
        - ``har``: Swift HnSF STFT features, ``(B, 22, T_har)``

        Output:
        - ``vocoder_logits``: pre-tail tensor, ``(B, post_n_fft + 2, T_tail)``

        The output is exactly the tensor that the fused wrapper feeds into
        ``exp``/``sin``/``CustomSTFT.inverse``.
        """

        def __init__(self, gen: Any):
            super().__init__()
            self.generator = gen

        def forward(self, x_pre: torch.Tensor, ref_s: torch.Tensor, har: torch.Tensor) -> torch.Tensor:
            s = ref_s[:, :128]
            gen = self.generator
            x = x_pre
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
            return gen.conv_post(x)

    return _GeneratorBodyFromHar(generator).eval()


def _make_tail_module(generator: Any):
    import torch
    import torch.nn as nn

    class _GeneratorTail(nn.Module):
        """Final waveform tail for pre-tail vocoder logits."""

        def __init__(self, gen: Any):
            super().__init__()
            self.post_n_fft = int(gen.post_n_fft)
            self.stft = gen.stft

        def forward(self, vocoder_logits: torch.Tensor) -> torch.Tensor:
            spec = torch.exp(vocoder_logits[:, : self.post_n_fft // 2 + 1, :])
            phase = torch.sin(vocoder_logits[:, self.post_n_fft // 2 + 1 :, :])
            return self.stft.inverse(spec, phase)

    return _GeneratorTail(generator).eval()


def _remove_existing_package(path: Path) -> None:
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def _export_split_packages(
    body_package: Path,
    tail_package: Path,
    tensors: dict[str, np.ndarray],
    precision: str,
) -> dict[str, Any]:
    import coremltools as ct
    import torch

    from export_synth.wrappers import remove_dropout

    kmodel = _load_kmodel()
    gen = kmodel.decoder.generator

    x_pre_shape = tuple(int(v) for v in tensors["x_pre_padded"].shape)
    ref_s_shape = tuple(int(v) for v in tensors["ref_s"].shape)
    har_shape = tuple(int(v) for v in tensors["har_padded"].shape)
    x_pre = torch.zeros(x_pre_shape, dtype=torch.float32)
    ref_s = torch.zeros(ref_s_shape, dtype=torch.float32)
    har = torch.zeros(har_shape, dtype=torch.float32)

    body = _make_body_module(gen)
    body_removed_dropouts = remove_dropout(body)
    with torch.no_grad():
        traced_body = torch.jit.trace(
            body,
            (x_pre, ref_s, har),
            strict=False,
            check_trace=False,
        )
        logits = traced_body(x_pre, ref_s, har)
    logits_shape = tuple(int(v) for v in logits.shape)

    body_model = ct.convert(
        traced_body,
        inputs=[
            ct.TensorType(name="x_pre", shape=x_pre_shape, dtype=np.float32),
            ct.TensorType(name="ref_s", shape=ref_s_shape, dtype=np.float32),
            ct.TensorType(name="har", shape=har_shape, dtype=np.float32),
        ],
        outputs=[ct.TensorType(name="vocoder_logits")],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS13,
        compute_precision=_precision_arg(ct, precision),
        compute_units=ct.ComputeUnit.ALL,
    )
    body_package.parent.mkdir(parents=True, exist_ok=True)
    _remove_existing_package(body_package)
    body_model.save(str(body_package))

    tail = _make_tail_module(gen)
    tail_removed_dropouts = remove_dropout(tail)
    tail_input = torch.zeros(logits_shape, dtype=torch.float32)
    with torch.no_grad():
        traced_tail = torch.jit.trace(
            tail,
            (tail_input,),
            strict=False,
            check_trace=False,
        )
        traced_tail_out = traced_tail(tail_input)
    tail_samples = int(traced_tail_out.shape[-1])

    tail_model = ct.convert(
        traced_tail,
        inputs=[ct.TensorType(name="vocoder_logits", shape=logits_shape, dtype=np.float32)],
        outputs=[ct.TensorType(name="waveform")],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS13,
        compute_precision=_precision_arg(ct, precision),
        compute_units=ct.ComputeUnit.ALL,
    )
    _remove_existing_package(tail_package)
    tail_model.save(str(tail_package))

    return {
        "precision": precision,
        "body_package": str(body_package),
        "tail_package": str(tail_package),
        "x_pre_shape": list(x_pre_shape),
        "ref_s_shape": list(ref_s_shape),
        "har_shape": list(har_shape),
        "vocoder_logits_shape": list(logits_shape),
        "tail_samples": tail_samples,
        "body_removed_dropouts": body_removed_dropouts,
        "tail_removed_dropouts": tail_removed_dropouts,
    }


def _load_models(args: argparse.Namespace, body_package: Path, tail_package: Path):
    import coremltools as ct

    fused = ct.models.MLModel(
        str(args.fused_package),
        compute_units=_compute_units(ct, args.fused_compute_units),
    )
    body = ct.models.MLModel(
        str(body_package),
        compute_units=_compute_units(ct, args.body_compute_units),
    )
    tail = ct.models.MLModel(
        str(tail_package),
        compute_units=_compute_units(ct, args.tail_compute_units),
    )
    return fused, body, tail


def _predict_split(body: Any, tail: Any, inputs: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, float]]:
    start = time.perf_counter()
    body_out = body.predict(inputs)
    body_ms = (time.perf_counter() - start) * 1000.0
    logits_key = "vocoder_logits" if "vocoder_logits" in body_out else next(iter(body_out))
    logits = np.asarray(body_out[logits_key], dtype=np.float32)

    start = time.perf_counter()
    tail_out = tail.predict({"vocoder_logits": logits})
    tail_ms = (time.perf_counter() - start) * 1000.0
    waveform_key = "waveform" if "waveform" in tail_out else next(iter(tail_out))
    waveform = np.asarray(tail_out[waveform_key], dtype=np.float32)
    return waveform, {"body_ms": body_ms, "tail_ms": tail_ms, "total_ms": body_ms + tail_ms}


def _predict_fused(fused: Any, inputs: dict[str, np.ndarray]) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    out = fused.predict(inputs)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    key = "waveform" if "waveform" in out else next(iter(out))
    return np.asarray(out[key], dtype=np.float32), elapsed_ms


def _benchmark(
    args: argparse.Namespace,
    tensors: dict[str, np.ndarray],
    body_package: Path,
    tail_package: Path,
) -> dict[str, Any]:
    inputs = _predict_inputs(tensors)
    fused, body, tail = _load_models(args, body_package, tail_package)

    fused_first, fused_first_ms = _predict_fused(fused, inputs)
    split_first, split_first_times = _predict_split(body, tail, inputs)

    for _ in range(max(0, args.warmup)):
        _predict_fused(fused, inputs)
        _predict_split(body, tail, inputs)

    fused_times: list[float] = []
    split_body_times: list[float] = []
    split_tail_times: list[float] = []
    split_total_times: list[float] = []
    last_fused = fused_first
    last_split = split_first
    for _ in range(max(1, args.iterations)):
        last_fused, fused_ms = _predict_fused(fused, inputs)
        last_split, split_times = _predict_split(body, tail, inputs)
        fused_times.append(fused_ms)
        split_body_times.append(split_times["body_ms"])
        split_tail_times.append(split_times["tail_ms"])
        split_total_times.append(split_times["total_ms"])

    reference_full = tensors["waveform_full"]
    trim_len = int(tensors["waveform"].size)
    fused_trim = last_fused.reshape(-1)[:trim_len]
    split_trim = last_split.reshape(-1)[:trim_len]

    return {
        "fused_compute_units": args.fused_compute_units,
        "body_compute_units": args.body_compute_units,
        "tail_compute_units": args.tail_compute_units,
        "warmup": int(max(0, args.warmup)),
        "iterations": int(max(1, args.iterations)),
        "first_predict_ms": {
            "fused": float(fused_first_ms),
            "split_body": float(split_first_times["body_ms"]),
            "split_tail": float(split_first_times["tail_ms"]),
            "split_total": float(split_first_times["total_ms"]),
        },
        "warm_predict_times_ms": {
            "fused": fused_times,
            "split_body": split_body_times,
            "split_tail": split_tail_times,
            "split_total": split_total_times,
        },
        "warm_predict_median_ms": {
            "fused": float(statistics.median(fused_times)),
            "split_body": float(statistics.median(split_body_times)),
            "split_tail": float(statistics.median(split_tail_times)),
            "split_total": float(statistics.median(split_total_times)),
        },
        "metrics": {
            "fused_vs_dump_full": _metrics(reference_full, last_fused),
            "split_vs_dump_full": _metrics(reference_full, last_split),
            "split_vs_fused_full": _metrics(last_fused, last_split),
            "fused_vs_dump_trimmed": _metrics(tensors["waveform"], fused_trim),
            "split_vs_dump_trimmed": _metrics(tensors["waveform"], split_trim),
            "split_vs_fused_trimmed": _metrics(fused_trim, split_trim),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest, tensors = load_tensor_dump(args.tensor_dump)
    required = ["x_pre_padded", "ref_s", "har_padded", "waveform_full", "waveform"]
    missing = [name for name in required if name not in tensors]
    if missing:
        raise SystemExit(f"tensor dump missing required tensors: {missing}")

    label = args.label or _duration_label_from_dump(args.tensor_dump, manifest)
    work_dir = args.output_dir / label
    body_package = work_dir / f"kokoro_generator_body_from_har_{label}.mlpackage"
    tail_package = work_dir / f"kokoro_generator_tail_{label}.mlpackage"
    report_path = work_dir / "report.json"

    export_report: dict[str, Any] | None = None
    if args.skip_export:
        if not body_package.is_dir() or not tail_package.is_dir():
            raise SystemExit(
                "--skip-export requested but split packages are missing: "
                f"{body_package}, {tail_package}"
            )
    else:
        export_report = _export_split_packages(
            body_package,
            tail_package,
            tensors,
            args.precision,
        )

    benchmark = _benchmark(args, tensors, body_package, tail_package)
    split_metrics = benchmark["metrics"]["split_vs_fused_trimmed"]
    passes = bool(
        split_metrics["correlation"] is not None
        and split_metrics["correlation"] >= args.min_corr
        and split_metrics["snr_db"] >= args.min_snr
        and split_metrics["max_abs_error"] <= args.max_abs_error
    )

    med = benchmark["warm_predict_median_ms"]
    speedup_vs_fused_pct = None
    if med["fused"] > 0:
        speedup_vs_fused_pct = 100.0 * (med["fused"] - med["split_total"]) / med["fused"]

    report = {
        "tensor_dump": str(args.tensor_dump),
        "fused_package": str(args.fused_package),
        "body_package": str(body_package),
        "tail_package": str(tail_package),
        "report": str(report_path),
        "manifest_metadata": manifest.get("metadata", {}),
        "export": export_report,
        "benchmark": benchmark,
        "thresholds": {
            "min_corr": args.min_corr,
            "min_snr": args.min_snr,
            "max_abs_error": args.max_abs_error,
        },
        "speedup_vs_fused_pct": speedup_vs_fused_pct,
        "passes": passes,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tensor-dump",
        type=Path,
        default=Path("outputs/generator_isolation/dumps/3s"),
        help="Swift generator tensor dump.",
    )
    parser.add_argument(
        "--fused-package",
        type=Path,
        default=Path("coreml/kokoro_decoder_har_post_3s.mlpackage"),
        help="Shipping fused HAR-post package to compare against.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/generator_split"),
        help="Directory for generated split packages and reports.",
    )
    parser.add_argument("--label", default=None)
    parser.add_argument(
        "--precision",
        default="fp16",
        choices=("fp16", "float16", "fp32", "float32"),
        help="Core ML conversion precision for split packages.",
    )
    parser.add_argument("--fused-compute-units", default="cpuAndGPU")
    parser.add_argument("--body-compute-units", default="cpuAndGPU")
    parser.add_argument("--tail-compute-units", default="cpuAndGPU")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--min-corr", type=float, default=0.99)
    parser.add_argument("--min-snr", type=float, default=35.0)
    parser.add_argument("--max-abs-error", type=float, default=1e-2)
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--fail-on-difference", action="store_true")
    args = parser.parse_args()

    report = run(args)
    med = report["benchmark"]["warm_predict_median_ms"]
    split_metrics = report["benchmark"]["metrics"]["split_vs_fused_trimmed"]
    print(
        "generator_split "
        f"passes={report['passes']} "
        f"label={Path(report['body_package']).parent.name} "
        f"fused_median_ms={med['fused']:.3f} "
        f"split_median_ms={med['split_total']:.3f} "
        f"body_median_ms={med['split_body']:.3f} "
        f"tail_median_ms={med['split_tail']:.3f} "
        f"speedup_vs_fused_pct={report['speedup_vs_fused_pct']:.2f} "
        f"corr={split_metrics['correlation']} "
        f"snr_db={split_metrics['snr_db']:.2f} "
        f"max_abs={split_metrics['max_abs_error']:.6g} "
        f"report={report['report']}"
    )
    if args.fail_on_difference and not report["passes"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
