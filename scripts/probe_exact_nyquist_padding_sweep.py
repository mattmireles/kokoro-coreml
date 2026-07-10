#!/usr/bin/env python3
"""Sweep HAR padding lengths with exact Swift Nyquist phase repair.

This is a focused follow-up to ``probe_nyquist_phase_contribution.py``. It
tests whether the now-solved strict Nyquist formula can recover speed by using
less than the full padded shipping HAR length while preserving waveform parity.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_ROOT))

from audio_parity_tensor_io import load_tensor_dump  # noqa: E402
from probe_generator_exact_geometry import _load_kmodel, _metrics  # noqa: E402
from probe_nyquist_phase_contribution import (  # noqa: E402
    NYQUIST_HAR_CHANNEL,
    _manual_stft_har,
    _pad_or_trim_har,
    _run_generator_from_har,
    _swift_basis_nyquist_atan2_phase,
)


def _strict_gate(metrics: dict[str, Any]) -> bool:
    return (
        float(metrics.get("correlation") or 0.0) >= 0.99998
        and float(metrics.get("snr_db") or 0.0) >= 45.0
        and float(metrics.get("max_abs_error") or float("inf")) <= 0.01
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    manifest, tensors = load_tensor_dump(args.tensor_dump)
    required = ["har_source", "x_pre_padded", "ref_s", "waveform_raw_trimmed"]
    missing = [name for name in required if name not in tensors]
    if missing:
        raise SystemExit(f"tensor dump missing required tensors: {missing}")

    generator = _load_kmodel().decoder.generator.eval()
    recomputed_har = _manual_stft_har(generator, tensors["har_source"])
    exact_nyquist = torch.from_numpy(
        _swift_basis_nyquist_atan2_phase(tensors["har_source"], int(recomputed_har.size(2)))
    )
    recomputed_har[:, NYQUIST_HAR_CHANNEL, :] = exact_nyquist

    reference_waveform = tensors["waveform_raw_trimmed"].reshape(-1)
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for pad_har_to in args.pad_har_to:
            har = _pad_or_trim_har(recomputed_har, pad_har_to)
            waveform = _run_generator_from_har(generator, tensors, har).detach().cpu().numpy().astype(np.float32)
            metrics = _metrics(reference_waveform, waveform.reshape(-1))
            rows.append(
                {
                    "pad_har_to": int(pad_har_to),
                    "snr_db": float(metrics["snr_db"]),
                    "correlation": float(metrics["correlation"]),
                    "max_abs_error": float(metrics["max_abs_error"]),
                    "passes_strict_gate": _strict_gate(metrics),
                }
            )

    passing = [row for row in rows if row["passes_strict_gate"]]
    report = {
        "tensor_dump": str(args.tensor_dump),
        "manifest_metadata": manifest.get("metadata", {}),
        "natural_har_time": int(recomputed_har.size(2)),
        "full_padded_har_time": int(tensors["har_padded"].shape[-1]) if "har_padded" in tensors else None,
        "rows": rows,
        "first_strict_pad_har_to": passing[0]["pad_har_to"] if passing else None,
        "strict_gate": {
            "correlation_min": 0.99998,
            "snr_db_min": 45.0,
            "max_abs_max": 0.01,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tensor_dump", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pad-har-to", type=int, action="append", required=True)
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "natural_har_time": report["natural_har_time"],
                "full_padded_har_time": report["full_padded_har_time"],
                "first_strict_pad_har_to": report["first_strict_pad_har_to"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
