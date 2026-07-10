#!/usr/bin/env python3
"""Create the five-bucket external bakeoff runtime input manifest."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.bakeoff_harness import (  # noqa: E402
    BAKEOFF_INPUTS,
    HAR_POST_BUCKETS,
    SPEED,
    VOICE,
    _select_bucket_seconds_standalone,
    _sha256_str,
)
from scripts.external_bakeoff.schema import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    RUNTIME_BUCKETS,
    validate_manifest,
    write_json,
)

TEN_SECOND_CANDIDATES = [
    (
        "The research team gathered before sunrise to test the portable speech "
        "system. Every sentence had to sound natural, stable, and fast enough "
        "for an impatient listener."
    ),
    (
        "A careful benchmark is only useful when every implementation receives "
        "the same text, the same voice, and the same timing boundary from the "
        "first sample to the final in-memory waveform."
    ),
    (
        "At the edge of the harbor, the engineer watched the status display "
        "while the model warmed, measured, and returned clean audio before the "
        "next request arrived."
    ),
]


def _extract_duration(text: str) -> tuple[float, int]:
    import torch
    from kokoro.coreml_pipeline import HybridTTSPipeline
    from kokoro.model import KModel
    from kokoro.pipeline import KPipeline

    class _CPUBenchPipeline:
        def __init__(self) -> None:
            self.pytorch_model = KModel().to("cpu")
            self.pytorch_model.eval()
            self.pipeline = KPipeline(lang_code="a", model=False)
            self.coreml_synth_buckets = {}
            self.coreml_decoder_har_buckets = {}
            self.coreml_decoder_har_post_buckets = {sec: None for sec in HAR_POST_BUCKETS}

        extract_vocoder_inputs = HybridTTSPipeline.extract_vocoder_inputs

    torch.manual_seed(0)
    pipe = _CPUBenchPipeline()
    vi = pipe.extract_vocoder_inputs(text, VOICE, SPEED)
    if vi is None:
        raise RuntimeError("extract_vocoder_inputs returned None")
    t_f0 = int(vi["f0_curve"].shape[-1])
    return t_f0 / 80.0, t_f0


def _entry(key: str, text: str, canonical_duration_s: float, t_f0: int) -> dict[str, Any]:
    expected_bucket = _select_bucket_seconds_standalone(
        canonical_duration_s, sorted(HAR_POST_BUCKETS)
    )
    return {
        "text": text,
        "voice": VOICE,
        "speed": SPEED,
        "canonical_duration_s": round(canonical_duration_s, 6),
        "expected_bucket_s": expected_bucket,
        "T_f0": t_f0,
        "text_sha256": _sha256_str(text),
    }


def build_manifest(bakeoff_manifest: Path | None) -> dict[str, Any]:
    inputs: dict[str, dict[str, Any]] = {}
    source_inputs: dict[str, Any] = {}
    if bakeoff_manifest and bakeoff_manifest.exists():
        source_inputs = json.loads(bakeoff_manifest.read_text()).get("inputs", {})

    for key in ("3s", "7s", "15s", "30s"):
        source = source_inputs.get(key)
        if source is not None:
            inputs[key] = {
                "text": source["text"],
                "voice": source.get("voice", VOICE),
                "speed": source.get("speed", SPEED),
                "canonical_duration_s": source["canonical_duration_s"],
                "expected_bucket_s": source["expected_bucket_s"],
                "T_f0": source.get("T_f0"),
                "text_sha256": source.get("text_sha256") or _sha256_str(source["text"]),
            }
        else:
            duration, t_f0 = _extract_duration(BAKEOFF_INPUTS[key])
            inputs[key] = _entry(key, BAKEOFF_INPUTS[key], duration, t_f0)

    for candidate in TEN_SECOND_CANDIDATES:
        duration, t_f0 = _extract_duration(candidate)
        bucket = _select_bucket_seconds_standalone(duration, sorted(HAR_POST_BUCKETS))
        if bucket == 10:
            inputs["10s"] = _entry("10s", candidate, duration, t_f0)
            break
    else:
        raise RuntimeError("no configured 10s candidate routed to the 10s bucket")

    manifest = {
        "schema_version": 1,
        "voice": VOICE,
        "speed": SPEED,
        "runtime_buckets": list(RUNTIME_BUCKETS),
        "inputs": {key: inputs[key] for key in RUNTIME_BUCKETS},
    }
    validate_manifest(manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bakeoff-manifest",
        type=Path,
        default=Path("outputs/bakeoff/input_manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "runtime_input_manifest.json",
    )
    args = parser.parse_args()

    manifest = build_manifest(args.bakeoff_manifest)
    write_json(args.output, manifest)
    print(f"Wrote {args.output}")
    for key, item in manifest["inputs"].items():
        print(f"{key}: {item['canonical_duration_s']:.3f}s -> {item['expected_bucket_s']}s")


if __name__ == "__main__":
    main()
