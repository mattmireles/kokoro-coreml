#!/usr/bin/env python3
"""Probe exact-duration ``GeneratorFromHar`` Core ML geometry from a tensor dump.

The production ``decoder-har`` packages are bucket shaped: a 3s package emits a
72,000-sample waveform and the runtime trims it to the observed utterance
length. This script tests the narrower hypothesis that the generator package
itself can be exported at the observed waveform length and still match the
current trimmed reference.

Generated dumps, packages, and reports are written under ``outputs/`` by
default. Shipping packages in ``coreml/`` are not modified.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_ROOT))

from audio_parity_tensor_io import TensorDumpWriter, load_tensor_dump  # noqa: E402


@dataclass(frozen=True)
class ExactGeometry:
    target_samples: int
    target_internal_samples: int
    target_duration_s: float
    source_output_samples: int
    source_internal_samples: int
    source_x_pre_time: int
    source_har_input_time: int
    source_har_source_time: int
    x_pre_time: int
    har_input_time: int
    har_source_time: int
    f0_upsample_samples: int
    har_hop_samples: int


def _int_div(numer: int, denom: int, label: str) -> int:
    if denom <= 0:
        raise ValueError(f"{label}: denominator must be positive, got {denom}")
    if numer % denom:
        raise ValueError(f"{label}: {numer} is not divisible by {denom}")
    return numer // denom


def _duration_label(samples: int, sample_rate: int = 24000) -> str:
    seconds = samples / float(sample_rate)
    text = f"{seconds:.4f}".rstrip("0").rstrip(".")
    return text.replace(".", "p") + "s"


def _derive_geometry(
    manifest: dict[str, Any],
    tensors: dict[str, np.ndarray],
    target_samples: int,
) -> ExactGeometry:
    required = ["x_pre_padded", "har", "har_padded", "waveform_full", "waveform"]
    missing = [name for name in required if name not in tensors]
    if missing:
        raise ValueError(f"source dump missing required tensors: {missing}")

    if target_samples <= 0:
        raise ValueError(f"target samples must be positive, got {target_samples}")

    source_output_samples = int(tensors["waveform_full"].shape[-1])
    if target_samples > source_output_samples:
        raise ValueError(
            f"target samples {target_samples} exceeds source waveform_full "
            f"{source_output_samples}"
        )

    source_internal_samples = source_output_samples * 2
    source_x_pre_time = int(tensors["x_pre_padded"].shape[-1])
    source_har_input_time = int(tensors["har_padded"].shape[-1])
    source_har_source_time = int(tensors["har"].shape[-1])

    f0_upsample_samples = _int_div(
        source_internal_samples,
        source_x_pre_time,
        "source internal samples / x_pre time",
    )
    har_hop_from_internal = _int_div(
        source_internal_samples,
        source_har_input_time - 1,
        "source internal samples / padded HAR hops",
    )
    har_hop_from_audio = _int_div(
        source_output_samples,
        source_har_source_time - 1,
        "source output samples / unpadded HAR hops",
    )
    if har_hop_from_internal != har_hop_from_audio:
        raise ValueError(
            "inconsistent HAR hop sizes: "
            f"{har_hop_from_internal} vs {har_hop_from_audio}"
        )

    target_internal_samples = target_samples * 2
    x_pre_time = _int_div(
        target_internal_samples,
        f0_upsample_samples,
        "target internal samples / f0 upsample",
    )
    har_input_time = (
        _int_div(
            target_internal_samples,
            har_hop_from_internal,
            "target internal samples / HAR hop",
        )
        + 1
    )
    har_source_time = (
        _int_div(
            target_samples,
            har_hop_from_audio,
            "target output samples / HAR hop",
        )
        + 1
    )

    sample_rate = int(
        manifest.get("metadata", {}).get("sample_rate")
        or manifest.get("metadata", {}).get("audio_sample_rate")
        or 24000
    )
    return ExactGeometry(
        target_samples=target_samples,
        target_internal_samples=target_internal_samples,
        target_duration_s=target_samples / float(sample_rate),
        source_output_samples=source_output_samples,
        source_internal_samples=source_internal_samples,
        source_x_pre_time=source_x_pre_time,
        source_har_input_time=source_har_input_time,
        source_har_source_time=source_har_source_time,
        x_pre_time=x_pre_time,
        har_input_time=har_input_time,
        har_source_time=har_source_time,
        f0_upsample_samples=f0_upsample_samples,
        har_hop_samples=har_hop_from_internal,
    )


def _copy_with_last_dim(array: np.ndarray, length: int) -> np.ndarray:
    if array.shape[-1] < length:
        raise ValueError(f"cannot crop last dim {array.shape[-1]} to {length}")
    return np.ascontiguousarray(array[..., :length])


def _prepare_exact_dump(
    source_dump: Path,
    output_dump: Path,
    geometry: ExactGeometry,
) -> dict[str, Any]:
    manifest, tensors = load_tensor_dump(source_dump)
    metadata = dict(manifest.get("metadata", {}))
    metadata.update(
        {
            "producer": "probe_generator_exact_geometry.py",
            "source_tensor_dump": str(source_dump),
            "exact_generator_geometry": True,
            "bucket_seconds": None,
            "canonical_duration_s": geometry.target_duration_s,
            "observed_audio_duration_s": geometry.target_duration_s,
            "trim_len": geometry.target_samples,
            "full_f0_len": geometry.x_pre_time,
            "x_pre_expected_time": geometry.x_pre_time,
            "har_expected_time": geometry.har_input_time,
            "source_bucket_output_samples": geometry.source_output_samples,
            "source_bucket_internal_samples": geometry.source_internal_samples,
            "target_internal_samples": geometry.target_internal_samples,
            "f0_upsample_samples": geometry.f0_upsample_samples,
            "har_hop_samples": geometry.har_hop_samples,
        }
    )

    writer = TensorDumpWriter(output_dump, metadata=metadata)
    tensor_names = [record["name"] for record in manifest.get("tensors", [])]

    for name in tensor_names:
        arr = tensors[name]
        if name in {"x_pre", "x_pre_padded"}:
            out = _copy_with_last_dim(arr, geometry.x_pre_time)
        elif name in {"f0", "n", "f0_padded", "n_padded"}:
            out = _copy_with_last_dim(arr, geometry.x_pre_time)
        elif name in {"har_magnitude", "har_phase", "har"}:
            out = _copy_with_last_dim(arr, geometry.har_source_time)
        elif name == "har_padded":
            source_har = _copy_with_last_dim(tensors["har"], geometry.har_source_time)
            out = np.zeros(
                (
                    int(source_har.shape[0]),
                    int(source_har.shape[1]),
                    geometry.har_input_time,
                ),
                dtype=np.float32,
            )
            out[..., : geometry.har_source_time] = source_har
        elif name == "har_source":
            out = _copy_with_last_dim(arr, geometry.target_samples)
        elif name == "waveform_full":
            out = np.ascontiguousarray(
                tensors["waveform"].reshape(1, 1, -1)[..., : geometry.target_samples]
            )
        elif name in {"waveform_raw_trimmed", "waveform"}:
            out = np.ascontiguousarray(arr.reshape(-1)[: geometry.target_samples])
        else:
            out = arr
        writer.write(name, out)

    manifest_path = writer.close()
    return json.loads(manifest_path.read_text())


def _load_kmodel():
    from export_synth.wrappers import KModel

    config_path = Path("checkpoints/config.json")
    checkpoint_path = Path("checkpoints/kokoro-v1_0.pth")

    def _is_readable_file(path: Path) -> bool:
        try:
            return path.is_file()
        except OSError:
            return False

    if _is_readable_file(config_path) and _is_readable_file(checkpoint_path):
        return KModel(
            config=str(config_path),
            model=str(checkpoint_path),
            disable_complex=True,
        )
    if _is_readable_file(config_path):
        return KModel(config=str(config_path), disable_complex=True)
    return KModel(disable_complex=True)


def _precision_arg(ct: Any, precision: str):
    value = precision.strip().lower()
    if value in {"fp32", "float32"}:
        return ct.precision.FLOAT32
    if value in {"fp16", "float16"}:
        return ct.precision.FLOAT16
    raise ValueError(f"unsupported precision {precision!r}")


def _compute_units(ct: Any, value: str):
    normalized = value.strip().lower().replace("_", "").replace("-", "")
    if normalized == "all":
        return ct.ComputeUnit.ALL
    if normalized in {"cpuandgpu", "cpugpu"}:
        return ct.ComputeUnit.CPU_AND_GPU
    if normalized in {"cpuandne", "cpuandneuralengine", "cpune"}:
        return ct.ComputeUnit.CPU_AND_NE
    if normalized in {"cpuonly", "cpu"}:
        return ct.ComputeUnit.CPU_ONLY
    raise ValueError(f"unsupported compute units {value!r}")


def _export_exact_generator(
    package: Path,
    tensors: dict[str, np.ndarray],
    geometry: ExactGeometry,
    precision: str,
) -> dict[str, Any]:
    import coremltools as ct
    import torch

    from export_synth.wrappers import GeneratorFromHar, remove_dropout

    ref_s_shape = tuple(int(v) for v in tensors["ref_s"].shape)
    x_pre_shape = (
        int(tensors["x_pre_padded"].shape[0]),
        int(tensors["x_pre_padded"].shape[1]),
        geometry.x_pre_time,
    )
    har_shape = (
        int(tensors["har_padded"].shape[0]),
        int(tensors["har_padded"].shape[1]),
        geometry.har_input_time,
    )

    kmodel = _load_kmodel()
    gen_from_har = GeneratorFromHar(kmodel.decoder.generator).eval()
    removed_dropouts = remove_dropout(gen_from_har)

    x_pre = torch.zeros(x_pre_shape, dtype=torch.float32)
    ref_s = torch.zeros(ref_s_shape, dtype=torch.float32)
    har = torch.zeros(har_shape, dtype=torch.float32)
    with torch.no_grad():
        traced = torch.jit.trace(
            gen_from_har,
            (x_pre, ref_s, har),
            strict=False,
            check_trace=False,
        )
        traced_out = traced(x_pre, ref_s, har)
    traced_samples = int(traced_out.shape[-1])
    if traced_samples != geometry.target_samples:
        raise RuntimeError(
            f"traced exact generator emitted {traced_samples} samples, "
            f"expected {geometry.target_samples}"
        )

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="x_pre", shape=x_pre_shape, dtype=np.float32),
            ct.TensorType(name="ref_s", shape=ref_s_shape, dtype=np.float32),
            ct.TensorType(name="har", shape=har_shape, dtype=np.float32),
        ],
        outputs=[ct.TensorType(name="waveform")],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS13,
        compute_precision=_precision_arg(ct, precision),
        compute_units=ct.ComputeUnit.ALL,
    )
    if package.exists():
        import shutil

        shutil.rmtree(package)
    package.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(package))
    return {
        "package": str(package),
        "precision": precision,
        "removed_dropouts": removed_dropouts,
        "traced_samples": traced_samples,
        "x_pre_shape": list(x_pre_shape),
        "ref_s_shape": list(ref_s_shape),
        "har_shape": list(har_shape),
    }


def _metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    ref = reference.astype(np.float64).reshape(-1)
    cand = candidate.astype(np.float64).reshape(-1)
    n = min(ref.size, cand.size)
    ref = ref[:n]
    cand = cand[:n]
    diff = ref - cand
    corr = None
    if n > 1 and float(ref.std()) > 0.0 and float(cand.std()) > 0.0:
        corr = float(np.corrcoef(ref, cand)[0, 1])
    denom = float(np.linalg.norm(ref) * np.linalg.norm(cand))
    cosine = float(np.dot(ref, cand) / denom) if denom > 0 else None
    snr = 10.0 * np.log10(
        (float(np.sum(ref * ref)) + 1e-12)
        / (float(np.sum(diff * diff)) + 1e-12)
    )
    return {
        "reference_samples": int(reference.size),
        "candidate_samples": int(candidate.size),
        "samples_compared": int(n),
        "max_abs_error": float(np.max(np.abs(diff))) if n else 0.0,
        "mean_abs_error": float(np.mean(np.abs(diff))) if n else 0.0,
        "rmse": float(np.sqrt(np.mean(diff * diff))) if n else 0.0,
        "snr_db": float(snr),
        "correlation": corr,
        "cosine_similarity": cosine,
    }


def _run_predict_report(
    package: Path,
    tensors: dict[str, np.ndarray],
    compute_units: str,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    import coremltools as ct

    model = ct.models.MLModel(
        str(package),
        compute_units=_compute_units(ct, compute_units),
    )
    inputs = {
        "x_pre": tensors["x_pre_padded"].astype(np.float32),
        "ref_s": tensors["ref_s"].astype(np.float32),
        "har": tensors["har_padded"].astype(np.float32),
    }
    first_start = time.perf_counter()
    prediction = model.predict(inputs)
    first_predict_ms = (time.perf_counter() - first_start) * 1000.0
    key = "waveform" if "waveform" in prediction else next(iter(prediction))
    waveform_full = np.asarray(prediction[key], dtype=np.float32)
    trim_len = int(tensors["waveform"].size)
    waveform = waveform_full.reshape(-1)[:trim_len]

    for _ in range(max(0, warmup)):
        _ = model.predict(inputs)

    samples_ms: list[float] = []
    for _ in range(max(1, iterations)):
        start = time.perf_counter()
        _ = model.predict(inputs)
        samples_ms.append((time.perf_counter() - start) * 1000.0)

    return {
        "compute_units": compute_units,
        "prediction_key": key,
        "first_predict_ms": float(first_predict_ms),
        "warmup": int(max(0, warmup)),
        "iterations": int(max(1, iterations)),
        "warm_predict_times_ms": samples_ms,
        "warm_predict_median_ms": float(statistics.median(samples_ms)),
        "waveform_full_metrics": _metrics(tensors["waveform_full"], waveform_full),
        "waveform_trimmed_metrics": _metrics(tensors["waveform"], waveform),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_manifest, source_tensors = load_tensor_dump(args.tensor_dump)
    target_samples = int(
        args.target_samples
        if args.target_samples is not None
        else source_manifest.get("metadata", {}).get("trim_len")
        or source_tensors["waveform"].size
    )
    geometry = _derive_geometry(source_manifest, source_tensors, target_samples)
    label = args.label or _duration_label(target_samples)
    work_dir = args.output_dir / label
    exact_dump = work_dir / "tensor_dump"
    package = work_dir / f"kokoro_decoder_har_post_exact_{label}.mlpackage"
    report_path = work_dir / "report.json"

    exact_manifest = _prepare_exact_dump(args.tensor_dump, exact_dump, geometry)
    _, exact_tensors = load_tensor_dump(exact_dump)

    export_report: dict[str, Any] | None = None
    if args.skip_export:
        if not package.is_dir():
            raise SystemExit(f"--skip-export requested but package is missing: {package}")
    else:
        export_report = _export_exact_generator(
            package,
            exact_tensors,
            geometry,
            args.precision,
        )

    predict_report = _run_predict_report(
        package,
        exact_tensors,
        args.compute_units,
        args.warmup,
        args.iterations,
    )
    trimmed = predict_report["waveform_trimmed_metrics"]
    passes = bool(
        trimmed["correlation"] is not None
        and trimmed["correlation"] >= args.min_corr
        and trimmed["snr_db"] >= args.min_snr
        and trimmed["max_abs_error"] <= args.max_abs_error
    )

    report = {
        "source_tensor_dump": str(args.tensor_dump),
        "exact_tensor_dump": str(exact_dump),
        "package": str(package),
        "report": str(report_path),
        "geometry": asdict(geometry),
        "exact_manifest_metadata": exact_manifest.get("metadata", {}),
        "export": export_report,
        "predict": predict_report,
        "thresholds": {
            "min_corr": args.min_corr,
            "min_snr": args.min_snr,
            "max_abs_error": args.max_abs_error,
        },
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
        help="Source generator tensor dump.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/generator_exact_geometry"),
        help="Directory for generated exact dumps, packages, and reports.",
    )
    parser.add_argument("--label", default=None, help="Output label, e.g. 2p8s.")
    parser.add_argument(
        "--target-samples",
        type=int,
        default=None,
        help="Exact output sample count. Defaults to source manifest trim_len.",
    )
    parser.add_argument(
        "--precision",
        default="fp16",
        choices=("fp16", "float16", "fp32", "float32"),
        help="Core ML conversion precision.",
    )
    parser.add_argument(
        "--compute-units",
        default="cpuAndGPU",
        help="Core ML compute units for prediction timing.",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--min-corr", type=float, default=0.99)
    parser.add_argument("--min-snr", type=float, default=35.0)
    parser.add_argument("--max-abs-error", type=float, default=1e-2)
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Reuse an existing package under the output label.",
    )
    parser.add_argument("--fail-on-difference", action="store_true")
    args = parser.parse_args()

    report = run(args)
    trimmed = report["predict"]["waveform_trimmed_metrics"]
    print(
        "generator_exact_geometry "
        f"passes={report['passes']} "
        f"label={Path(report['package']).parent.name} "
        f"median_ms={report['predict']['warm_predict_median_ms']:.3f} "
        f"corr={trimmed['correlation']} "
        f"snr_db={trimmed['snr_db']:.2f} "
        f"max_abs={trimmed['max_abs_error']:.6g} "
        f"report={report['report']}"
    )
    if args.fail_on_difference and not report["passes"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
