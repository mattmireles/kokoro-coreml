#!/usr/bin/env python3
"""Analyze whether candidate WAV drift is explainable by lag or gain.

This script is intentionally lightweight: it consumes the listening-pack
``index.json`` written by ``scripts/create_f0_source_listening_pack.py`` and
compares each candidate WAV against the baseline WAV from the same record. It
does not approve candidates. It answers a narrower debugging question: would a
simple sample shift and affine amplitude correction make the waveform pass the
strict metric gate?
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

from audio_quality_probe import PCM_MAX, read_wav_pcm  # noqa: E402
from probe_generator_exact_geometry import _metrics  # noqa: E402


def _repo_path(path: str) -> Path:
    """Resolve a repo-relative path string from a listening-pack index."""

    p = Path(path)
    return p if p.is_absolute() else _ROOT / p


def _read_wav_float(path: Path) -> np.ndarray:
    """Read a 16-bit PCM WAV as a contiguous float32 mono vector."""

    _, _, _, pcm = read_wav_pcm(path)
    return np.ascontiguousarray(pcm.astype(np.float32) / PCM_MAX)


def _aligned_slices(reference: np.ndarray, candidate: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Return overlapping slices for a candidate lag.

    Positive ``lag`` means the candidate is shifted right relative to the
    reference, so comparison uses ``reference[:-lag]`` and ``candidate[lag:]``.
    Negative ``lag`` means the candidate is shifted left.
    """

    if lag > 0:
        n = min(reference.size - lag, candidate.size - lag)
        return reference[:n], candidate[lag : lag + n]
    if lag < 0:
        shift = -lag
        n = min(reference.size - shift, candidate.size - shift)
        return reference[shift : shift + n], candidate[:n]
    n = min(reference.size, candidate.size)
    return reference[:n], candidate[:n]


def _fit_gain_offset(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
    """Return least-squares ``gain, offset`` mapping candidate to reference."""

    ref = reference.astype(np.float64).reshape(-1)
    cand = candidate.astype(np.float64).reshape(-1)
    if ref.size == 0 or cand.size == 0:
        return 1.0, 0.0
    design = np.column_stack([cand, np.ones_like(cand)])
    gain, offset = np.linalg.lstsq(design, ref, rcond=None)[0]
    return float(gain), float(offset)


def _estimate_lag(reference: np.ndarray, candidate: np.ndarray, max_lag: int, downsample: int) -> int:
    """Estimate the best lag by coarse downsampled correlation then local search."""

    if max_lag <= 0:
        return 0
    factor = max(1, int(downsample))
    ref_ds = reference[::factor]
    cand_ds = candidate[::factor]
    max_lag_ds = max(1, max_lag // factor)

    best_lag_ds = 0
    best_score = -np.inf
    for lag in range(-max_lag_ds, max_lag_ds + 1):
        ref_slice, cand_slice = _aligned_slices(ref_ds, cand_ds, lag)
        if ref_slice.size < 2:
            continue
        ref_centered = ref_slice - float(ref_slice.mean())
        cand_centered = cand_slice - float(cand_slice.mean())
        denom = float(np.linalg.norm(ref_centered) * np.linalg.norm(cand_centered))
        score = float(np.dot(ref_centered, cand_centered) / denom) if denom > 0 else -np.inf
        if score > best_score:
            best_score = score
            best_lag_ds = lag

    center = best_lag_ds * factor
    window = max(factor * 2, 16)
    lo = max(-max_lag, center - window)
    hi = min(max_lag, center + window)
    best_lag = 0
    best_score = -np.inf
    for lag in range(lo, hi + 1):
        ref_slice, cand_slice = _aligned_slices(reference, candidate, lag)
        if ref_slice.size < 2:
            continue
        ref_centered = ref_slice - float(ref_slice.mean())
        cand_centered = cand_slice - float(cand_slice.mean())
        denom = float(np.linalg.norm(ref_centered) * np.linalg.norm(cand_centered))
        score = float(np.dot(ref_centered, cand_centered) / denom) if denom > 0 else -np.inf
        if score > best_score:
            best_score = score
            best_lag = lag
    return best_lag


def analyze_pair(reference: np.ndarray, candidate: np.ndarray, max_lag: int, downsample: int) -> dict[str, Any]:
    """Return raw, affine, lagged, and lagged-affine metrics for one pair."""

    raw = _metrics(reference, candidate)
    gain0, offset0 = _fit_gain_offset(reference[: raw["samples_compared"]], candidate[: raw["samples_compared"]])
    affine_candidate = candidate[: raw["samples_compared"]] * gain0 + offset0
    affine = _metrics(reference[: raw["samples_compared"]], affine_candidate)

    lag = _estimate_lag(reference, candidate, max_lag=max_lag, downsample=downsample)
    ref_lag, cand_lag = _aligned_slices(reference, candidate, lag)
    lagged = _metrics(ref_lag, cand_lag)
    gain_lag, offset_lag = _fit_gain_offset(ref_lag, cand_lag)
    lagged_affine = _metrics(ref_lag, cand_lag * gain_lag + offset_lag)

    return {
        "raw": raw,
        "affine": {"gain": gain0, "offset": offset0, "metrics": affine},
        "lagged": {"lag_samples": lag, "metrics": lagged},
        "lagged_affine": {
            "lag_samples": lag,
            "gain": gain_lag,
            "offset": offset_lag,
            "metrics": lagged_affine,
        },
    }


def analyze_index(index_path: Path, max_lag: int, downsample: int) -> dict[str, Any]:
    """Analyze all records in one F0-source listening-pack index."""

    index = json.loads(index_path.read_text())
    records = []
    for record in index.get("records", []):
        wavs = record.get("wavs") or {}
        baseline_path = _repo_path(str(wavs.get("baseline") or ""))
        candidate_path = _repo_path(str(wavs.get("candidate") or ""))
        if not baseline_path.exists() or not candidate_path.exists():
            raise FileNotFoundError(f"missing WAV for {record.get('label')}: {baseline_path}, {candidate_path}")
        baseline = _read_wav_float(baseline_path)
        candidate = _read_wav_float(candidate_path)
        analysis = analyze_pair(baseline, candidate, max_lag=max_lag, downsample=downsample)
        records.append(
            {
                "label": record.get("label"),
                "baseline_wav": str(baseline_path.relative_to(_ROOT)),
                "candidate_wav": str(candidate_path.relative_to(_ROOT)),
                "analysis": analysis,
            }
        )
    return {
        "index": str(index_path),
        "max_lag": int(max_lag),
        "downsample": int(downsample),
        "records": records,
    }


def _metric_summary(metrics: dict[str, Any]) -> str:
    corr = metrics.get("correlation")
    corr_s = "n/a" if corr is None else f"{corr:.6f}"
    return f"corr={corr_s} snr={metrics['snr_db']:.2f} max={metrics['max_abs_error']:.5f}"


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    """Write a compact markdown alignment report."""

    lines = [
        "# Waveform Alignment Analysis",
        "",
        f"Index: `{result['index']}`",
        f"Max lag searched: `{result['max_lag']}` samples.",
        "",
        "| Label | Raw | Gain/offset | Best lag | Lagged | Lag+gain/offset |",
        "| --- | --- | --- | ---: | --- | --- |",
    ]
    for record in result["records"]:
        analysis = record["analysis"]
        lines.append(
            "| "
            f"`{record['label']}` | "
            f"{_metric_summary(analysis['raw'])} | "
            f"{_metric_summary(analysis['affine']['metrics'])} | "
            f"{analysis['lagged']['lag_samples']} | "
            f"{_metric_summary(analysis['lagged']['metrics'])} | "
            f"{_metric_summary(analysis['lagged_affine']['metrics'])} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True, help="Listening-pack index.json to analyze.")
    parser.add_argument("--max-lag", type=int, default=2400, help="Maximum sample lag to test in either direction.")
    parser.add_argument("--downsample", type=int, default=8, help="Downsample factor for coarse lag search.")
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = analyze_index(args.index, max_lag=args.max_lag, downsample=args.downsample)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    write_markdown(args.output, result)
    print(
        json.dumps(
            {
                "records": len(result["records"]),
                "output": str(args.output),
                "json_output": str(args.json_output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
