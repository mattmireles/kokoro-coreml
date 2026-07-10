#!/usr/bin/env python3
"""Ingest JSON rendered by the Soniqo Kokoro iOS runner."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.external_bakeoff.schema import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    RUNTIME_BUCKETS,
    load_json,
    utc_now,
    validate_manifest,
    validate_result_payload,
    write_json,
)


DEFAULT_INPUT = DEFAULT_OUTPUT_DIR / "ios_runner_payload_latest.json"
DEFAULT_MANIFEST = DEFAULT_OUTPUT_DIR / "runtime_input_manifest.json"
DEFAULT_MACHINE_ID = "iphone-12-pro"
DEFAULT_DEVICE_MODEL = "iPhone 12 Pro (iPhone13,3)"
DEFAULT_VERSION = "soniqo/speech-swift pinned clone; see runner provenance"


def _impl_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _require_number(record: dict[str, Any], field: str) -> float:
    value = record.get(field)
    if not isinstance(value, (int, float)):
        raise ValueError(f"{record.get('input_key', '<unknown>')} missing numeric field {field}")
    return float(value)


def _require_list(record: dict[str, Any], field: str) -> list[float]:
    values = record.get(field)
    if not isinstance(values, list) or not values:
        raise ValueError(f"{record.get('input_key', '<unknown>')} missing non-empty list field {field}")
    return [float(value) for value in values]


def _ingest_records(
    *,
    payload: dict[str, Any],
    manifest: dict[str, Any],
    machine_id: str,
    version: str,
    source_path: Path,
) -> list[dict[str, Any]]:
    inputs = manifest["inputs"]
    raw_records = payload.get("records")
    if not isinstance(raw_records, list):
        raise ValueError("iOS payload must contain records list")

    impl = str(payload.get("impl") or "soniqo-speech-swift-kokoro-ios")
    framework = str(payload.get("framework") or "Swift + Core ML")
    hardware_target = str(payload.get("hardware_target") or "ANE/Core ML")
    compute_units = str(payload.get("compute_units") or "all")
    warm_iterations = int(payload.get("warm_iterations") or 0)

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in raw_records:
        if not isinstance(raw, dict):
            raise ValueError("each iOS record must be an object")
        key = str(raw.get("input_key") or "")
        if key not in RUNTIME_BUCKETS:
            raise ValueError(f"unexpected input_key {key!r}")
        if key in seen:
            raise ValueError(f"duplicate input_key {key!r}")
        seen.add(key)

        manifest_input = inputs[key]
        if raw.get("text_sha256") != manifest_input["text_sha256"]:
            raise ValueError(f"{key} text_sha256 does not match runtime manifest")
        expected_bucket = int(raw.get("expected_bucket_s") or 0)
        if expected_bucket != int(key.rstrip("s")):
            raise ValueError(f"{key} expected_bucket_s={expected_bucket}")
        voice = str(raw.get("voice") or "")
        if voice != manifest_input["voice"]:
            raise ValueError(f"{key} voice={voice!r}; expected {manifest_input['voice']!r}")
        canonical_duration = _require_number(raw, "canonical_audio_duration_s")
        manifest_duration = float(manifest_input["canonical_duration_s"])
        if abs(canonical_duration - manifest_duration) > 0.001:
            raise ValueError(
                f"{key} canonical_audio_duration_s={canonical_duration}; "
                f"expected {manifest_duration}"
            )

        warm_times = _require_list(raw, "warm_wall_times_s")
        if warm_iterations and len(warm_times) != warm_iterations:
            raise ValueError(
                f"{key} warm iteration count {len(warm_times)} != payload warm_iterations {warm_iterations}"
            )
        observed_duration = _require_number(raw, "observed_audio_duration_s")
        if observed_duration <= 0:
            raise ValueError(f"{key} observed_audio_duration_s must be positive")
        rtf_observed = [round(value / observed_duration, 6) for value in warm_times]

        records.append(
            {
                "impl": impl,
                "framework": framework,
                "hardware_target": hardware_target,
                "version": version,
                "machine_id": machine_id,
                "input_key": key,
                "text_sha256": str(raw["text_sha256"]),
                "voice": voice,
                "cold_wall_time_s": round(_require_number(raw, "cold_wall_time_s"), 6),
                "warm_wall_times_s": [round(value, 6) for value in warm_times],
                "canonical_audio_duration_s": round(canonical_duration, 6),
                "observed_audio_duration_s": round(observed_duration, 6),
                "rtf_observed": rtf_observed,
                "output_sha256": "",
                "status": "ok",
                "error": None,
                "provenance": {
                    "source": "SoniqoKokoroIOSRunner rendered JSON",
                    "source_path": str(source_path),
                    "compute_units": compute_units,
                    "expected_bucket_s": expected_bucket,
                    "sample_count": int(raw.get("sample_count") or 0),
                    "sample_rate": int(raw.get("sample_rate") or 0),
                    "output_sha256_unavailable": True,
                    "spotcheck_wav_unavailable": True,
                },
            }
        )

    missing = [key for key in RUNTIME_BUCKETS if key not in seen]
    if missing:
        raise ValueError(f"iOS payload missing runtime buckets: {', '.join(missing)}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--machine-id", default=DEFAULT_MACHINE_ID)
    parser.add_argument("--device-model", default=DEFAULT_DEVICE_MODEL)
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    ios_payload = load_json(args.input)
    manifest = load_json(args.manifest)
    validate_manifest(manifest)
    records = _ingest_records(
        payload=ios_payload,
        manifest=manifest,
        machine_id=args.machine_id,
        version=args.version,
        source_path=args.input,
    )
    impl = records[0]["impl"]
    output = args.output or DEFAULT_OUTPUT_DIR / f"results_{_impl_slug(impl)}_{args.machine_id}.json"
    payload = {
        "created_utc": utc_now(),
        "impl": impl,
        "machine_id": args.machine_id,
        "machine": {
            "machine_id": args.machine_id,
            "device_model": args.device_model,
            "source": "SoniqoKokoroIOSRunner rendered JSON",
        },
        "records": records,
        "provenance": {
            "source": "SoniqoKokoroIOSRunner rendered JSON",
            "source_path": str(args.input),
            "manifest": str(args.manifest),
            "version": args.version,
            "device_model": args.device_model,
            "output_sha256_unavailable": True,
            "spotcheck_wav_unavailable": True,
        },
    }
    validate_result_payload(payload)
    write_json(output, payload)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
