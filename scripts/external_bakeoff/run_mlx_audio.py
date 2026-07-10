#!/usr/bin/env python3
"""Benchmark Blaizzy/mlx-audio Kokoro against the external bakeoff schema."""
from __future__ import annotations

import argparse
import importlib.metadata
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.external_bakeoff.schema import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    DEFAULT_VOICE,
    error_record,
    result_file_payload,
    result_record,
    sha256_bytes,
    validate_manifest,
    validate_result_payload,
    load_json,
    write_wav_mono16,
    write_json,
)


def _generate_once(model, text: str, voice: str) -> tuple[float, np.ndarray, int]:
    start = time.perf_counter()
    chunks = list(model.generate(text=text, voice=voice, speed=1.0, lang_code="a"))
    elapsed = time.perf_counter() - start
    arrays = [np.array(chunk.audio, dtype=np.float32).reshape(-1) for chunk in chunks]
    audio = np.concatenate(arrays) if arrays else np.array([], dtype=np.float32)
    sample_rate = int(chunks[0].sample_rate) if chunks else 24000
    return elapsed, audio, sample_rate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_OUTPUT_DIR / "runtime_input_manifest.json")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--machine-id", required=True)
    parser.add_argument("--model-id", default="mlx-community/Kokoro-82M-bf16")
    parser.add_argument("--voice", default=DEFAULT_VOICE)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--input-key", action="append", default=None)
    parser.add_argument("--spotcheck-dir", type=Path, default=None)
    args = parser.parse_args()

    from mlx_audio.tts.utils import load_model

    manifest = load_json(args.manifest)
    validate_manifest(manifest)
    keys = args.input_key or list(manifest["inputs"].keys())
    version = importlib.metadata.version("mlx-audio")
    model = load_model(args.model_id)
    spotcheck_dir = args.spotcheck_dir or (
        DEFAULT_OUTPUT_DIR / "spotcheck_wavs" / f"mlx_audio_{args.machine_id}"
    )

    records = []
    for key in keys:
        item = manifest["inputs"][key]
        try:
            cold, audio, sample_rate = _generate_once(model, item["text"], args.voice)
            warm_times = []
            last_audio = audio
            for _ in range(args.iterations):
                elapsed, last_audio, sample_rate = _generate_once(model, item["text"], args.voice)
                warm_times.append(elapsed)
            observed = float(last_audio.size) / float(sample_rate)
            spotcheck_wav = spotcheck_dir / f"{key}.wav"
            write_wav_mono16(spotcheck_wav, last_audio, sample_rate)
            records.append(
                result_record(
                    impl="mlx-audio",
                    framework="MLX Python",
                    hardware_target="GPU/Metal",
                    version=version,
                    machine_id=args.machine_id,
                    input_key=key,
                    text=item["text"],
                    voice=args.voice,
                    cold_wall_time_s=cold,
                    warm_wall_times_s=warm_times,
                    canonical_audio_duration_s=float(item["canonical_duration_s"]),
                    observed_audio_duration_s=observed,
                    output_sha256=sha256_bytes(last_audio.astype(np.float32).tobytes()),
                    provenance={
                        "model_id": args.model_id,
                        "sample_rate": sample_rate,
                        "spotcheck_wav": str(spotcheck_wav),
                        "adapter_note": "Use np.array(result.audio).size; GenerationResult.samples is not reliable for Kokoro.",
                    },
                )
            )
        except Exception as exc:
            records.append(
                error_record(
                    impl="mlx-audio",
                    framework="MLX Python",
                    hardware_target="GPU/Metal",
                    version=version,
                    machine_id=args.machine_id,
                    input_key=key,
                    text=item["text"],
                    voice=args.voice,
                    canonical_audio_duration_s=float(item["canonical_duration_s"]),
                    provenance={"model_id": args.model_id, "spotcheck_dir": str(spotcheck_dir)},
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    payload = result_file_payload(
        impl="mlx-audio",
        machine_id=args.machine_id,
        records=records,
        provenance={
            "model_id": args.model_id,
            "mlx_audio_version": version,
            "spotcheck_dir": str(spotcheck_dir),
        },
    )
    validate_result_payload(payload)
    output = args.output or (DEFAULT_OUTPUT_DIR / f"results_mlx_audio_{args.machine_id}.json")
    write_json(output, payload)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
