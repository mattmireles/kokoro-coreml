#!/usr/bin/env python3
"""Shared schema helpers for external Kokoro bakeoff adapters."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
import wave
from pathlib import Path
from typing import Any

RUNTIME_BUCKETS = ("3s", "7s", "10s", "15s", "30s")
DEFAULT_VOICE = "af_heart"
DEFAULT_SPEED = 1.0
DEFAULT_OUTPUT_DIR = Path("outputs/external_bakeoff")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_wav_mono16(path: Path, samples: Any, sample_rate: int) -> None:
    """Write mono float samples as peak-normalized PCM16 WAV for spot checks."""
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= 1e-7:
        pcm = np.zeros(audio.shape, dtype=np.int16)
    else:
        pcm = np.clip(audio / peak, -1.0, 1.0)
        pcm = np.rint(pcm * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(int(sample_rate))
        f.writeframes(pcm.tobytes())


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def machine_info(machine_id: str) -> dict[str, Any]:
    info: dict[str, Any] = {
        "machine_id": machine_id,
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
    }
    for key, cmd in {
        "sw_vers": ["sw_vers"],
        "cpu_brand": ["sysctl", "-n", "machdep.cpu.brand_string"],
        "hw_model": ["sysctl", "-n", "hw.model"],
        "memory_bytes": ["sysctl", "-n", "hw.memsize"],
    }.items():
        try:
            value = subprocess.check_output(cmd, text=True).strip()
            info[key] = int(value) if key == "memory_bytes" else value
        except Exception:
            info[key] = "unknown"
    return info


def validate_manifest(manifest: dict[str, Any]) -> None:
    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("manifest must contain object field 'inputs'")
    missing = [key for key in RUNTIME_BUCKETS if key not in inputs]
    if missing:
        raise ValueError(f"manifest missing runtime buckets: {', '.join(missing)}")
    for key in RUNTIME_BUCKETS:
        item = inputs[key]
        if item.get("expected_bucket_s") != int(key.rstrip("s")):
            raise ValueError(
                f"{key} expected_bucket_s={item.get('expected_bucket_s')!r}; "
                f"expected {key.rstrip('s')}"
            )
        for field in ("text", "voice", "speed", "canonical_duration_s", "text_sha256"):
            if field not in item:
                raise ValueError(f"{key} missing required field {field!r}")


def result_record(
    *,
    impl: str,
    framework: str,
    hardware_target: str,
    version: str,
    machine_id: str,
    input_key: str,
    text: str,
    voice: str,
    cold_wall_time_s: float,
    warm_wall_times_s: list[float],
    canonical_audio_duration_s: float,
    observed_audio_duration_s: float,
    output_sha256: str,
    provenance: dict[str, Any],
    status: str = "ok",
    error: str | None = None,
) -> dict[str, Any]:
    rtf_observed = (
        [round(t / observed_audio_duration_s, 6) for t in warm_wall_times_s]
        if observed_audio_duration_s > 0
        else []
    )
    return {
        "impl": impl,
        "framework": framework,
        "hardware_target": hardware_target,
        "version": version,
        "machine_id": machine_id,
        "input_key": input_key,
        "text_sha256": sha256_text(text),
        "voice": voice,
        "cold_wall_time_s": round(cold_wall_time_s, 6),
        "warm_wall_times_s": [round(t, 6) for t in warm_wall_times_s],
        "canonical_audio_duration_s": round(canonical_audio_duration_s, 6),
        "observed_audio_duration_s": round(observed_audio_duration_s, 6),
        "rtf_observed": rtf_observed,
        "output_sha256": output_sha256,
        "status": status,
        "error": error,
        "provenance": provenance,
    }


def error_record(
    *,
    impl: str,
    framework: str,
    hardware_target: str,
    version: str,
    machine_id: str,
    input_key: str,
    text: str,
    voice: str,
    canonical_audio_duration_s: float,
    provenance: dict[str, Any],
    error: str,
) -> dict[str, Any]:
    return {
        "impl": impl,
        "framework": framework,
        "hardware_target": hardware_target,
        "version": version,
        "machine_id": machine_id,
        "input_key": input_key,
        "text_sha256": sha256_text(text),
        "voice": voice,
        "cold_wall_time_s": None,
        "warm_wall_times_s": [],
        "canonical_audio_duration_s": round(canonical_audio_duration_s, 6),
        "observed_audio_duration_s": None,
        "rtf_observed": [],
        "output_sha256": "",
        "status": "error",
        "error": error,
        "provenance": provenance,
    }


def result_file_payload(
    *,
    impl: str,
    machine_id: str,
    records: list[dict[str, Any]],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    return {
        "created_utc": utc_now(),
        "impl": impl,
        "machine_id": machine_id,
        "machine": machine_info(machine_id),
        "records": records,
        "provenance": provenance,
    }


def validate_result_payload(payload: dict[str, Any]) -> None:
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("result payload must contain non-empty records list")
    required = {
        "impl",
        "machine_id",
        "framework",
        "hardware_target",
        "version",
        "input_key",
        "text_sha256",
        "voice",
        "cold_wall_time_s",
        "warm_wall_times_s",
        "canonical_audio_duration_s",
        "observed_audio_duration_s",
        "rtf_observed",
        "output_sha256",
        "provenance",
    }
    for idx, record in enumerate(records):
        missing = sorted(required - set(record))
        if missing:
            raise ValueError(f"record {idx} missing fields: {', '.join(missing)}")
        if record["input_key"] not in RUNTIME_BUCKETS:
            raise ValueError(f"record {idx} has invalid input_key {record['input_key']!r}")
        if not isinstance(record["warm_wall_times_s"], list):
            raise ValueError(f"record {idx} warm_wall_times_s must be a list")
