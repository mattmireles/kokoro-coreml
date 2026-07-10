#!/usr/bin/env python3
"""Run same-window Config F through the existing Swift benchmark CLI."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.external_bakeoff.schema import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    error_record,
    result_file_payload,
    result_record,
    validate_manifest,
    validate_result_payload,
    load_json,
    write_json,
)


class ConfigFBatchRunner:
    """Persistent kokoro-bench batch subprocess with one cache lifetime."""

    def __init__(
        self,
        binary: Path,
        compute_units: str,
        models_dir: Path,
        inputs_dir: Path,
        hnsf_weights: Path,
        use_exact_duration_models: bool,
    ) -> None:
        env = os.environ.copy()
        if use_exact_duration_models:
            env["KOKORO_USE_EXACT_DURATION_MODELS"] = "1"
        self.proc = subprocess.Popen(
            [
                str(binary),
                "--models-dir", str(models_dir),
                "--inputs-dir", str(inputs_dir),
                "--hnsf-weights", str(hnsf_weights),
                "--batch",
                "--compute-units", compute_units,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            env=env,
        )
        self._read_until("READY")

    def close(self) -> None:
        if self.proc.stdin:
            self.proc.stdin.close()
        return_code = self.proc.wait(timeout=30)
        if return_code != 0:
            raise RuntimeError(f"kokoro-bench batch exited with status {return_code}")

    def _read_status_line(self) -> str:
        if not self.proc.stdout:
            raise RuntimeError("kokoro-bench stdout pipe is unavailable")
        line = self.proc.stdout.readline()
        if not line:
            return_code = self.proc.poll()
            raise RuntimeError(f"kokoro-bench batch ended before status line, return={return_code}")
        return line.strip()

    def _read_until(self, token: str) -> str:
        while True:
            line = self._read_status_line()
            if token in line:
                return line
            if "ERROR" in line:
                raise RuntimeError(f"kokoro-bench batch returned error status: {line}")

    def run_once(
        self,
        input_key: str,
        seed: int = 42,
        warmup: bool = False,
        wav_path: Optional[Path] = None,
    ) -> dict[str, Any]:
        if not self.proc.stdin:
            raise RuntimeError("kokoro-bench stdin pipe is unavailable")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            out = Path(f.name)
        if wav_path is None:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                wav = Path(f.name)
        else:
            wav = wav_path
            wav.parent.mkdir(parents=True, exist_ok=True)

        command = {
            "input_key": input_key,
            "seed": seed,
            "output": str(out),
            "wav": str(wav),
            "warmup": warmup,
        }
        self.proc.stdin.write(json.dumps(command, separators=(",", ":")) + "\n")
        self.proc.stdin.flush()
        self._read_until("DONE")

        data = json.loads(out.read_text())
        if wav.exists():
            data["output_sha256"] = hashlib.sha256(wav.read_bytes()).hexdigest()
        out.unlink(missing_ok=True)
        if wav_path is None:
            wav.unlink(missing_ok=True)
        if data.get("status") != "ok":
            raise RuntimeError(f"kokoro-bench batch record failed: {data}")
        return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_OUTPUT_DIR / "runtime_input_manifest.json")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--machine-id", required=True)
    parser.add_argument("--binary", type=Path, default=Path("swift/.build/release/kokoro-bench"))
    parser.add_argument("--models-dir", type=Path, default=Path("coreml"))
    parser.add_argument("--inputs-dir", type=Path, default=Path("outputs/swift_bench_inputs"))
    parser.add_argument(
        "--hnsf-weights",
        type=Path,
        default=Path("outputs/swift_bench_inputs/hnsf_weights.json"),
    )
    parser.add_argument(
        "--compute-units",
        default="staged",
        help=(
            "Core ML compute policy passed to kokoro-bench. Default 'staged' "
            "matches the production Swift pipeline: duration/F0/generator on "
            "CPU+GPU, decoder-pre on CPU+ANE. Use 'all' to reproduce the old "
            "single-policy bakeoff cells."
        ),
    )
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument(
        "--preflight-runs",
        type=int,
        default=3,
        help="Run and discard this many calls per input before recording cold/warm timings.",
    )
    parser.add_argument(
        "--use-exact-duration-models",
        action="store_true",
        default=True,
        help="Set KOKORO_USE_EXACT_DURATION_MODELS=1 for the Swift benchmark subprocess.",
    )
    parser.add_argument(
        "--use-padded-duration-models",
        dest="use_exact_duration_models",
        action="store_false",
        help="Disable exact-duration model discovery and reproduce the old padded-duration cells.",
    )
    parser.add_argument("--input-key", action="append", default=None)
    parser.add_argument("--spotcheck-dir", type=Path, default=None)
    args = parser.parse_args()

    if not args.binary.exists():
        raise SystemExit(f"missing Swift benchmark binary: {args.binary}")

    manifest = load_json(args.manifest)
    validate_manifest(manifest)
    keys = args.input_key or list(manifest["inputs"].keys())

    records = []
    spotcheck_dir = args.spotcheck_dir or (
        DEFAULT_OUTPUT_DIR / "spotcheck_wavs" / f"config_f_reference_{args.machine_id}"
    )
    runner: Optional[ConfigFBatchRunner] = None
    try:
        runner = ConfigFBatchRunner(
            args.binary,
            args.compute_units,
            args.models_dir,
            args.inputs_dir,
            args.hnsf_weights,
            args.use_exact_duration_models,
        )
        for key in keys:
            item = manifest["inputs"][key]
            try:
                preflight = [
                    runner.run_once(key, warmup=True)
                    for _ in range(args.preflight_runs)
                ]
                cold_result = runner.run_once(key, warmup=False)
                warm = []
                for idx in range(args.iterations):
                    wav_path = spotcheck_dir / f"{key}.wav" if idx == args.iterations - 1 else None
                    warm.append(runner.run_once(key, warmup=True, wav_path=wav_path))
                records.append(
                    result_record(
                        impl="config-f-reference",
                        framework="Swift + Core ML",
                        hardware_target="ANE/Core ML",
                        version="local",
                        machine_id=args.machine_id,
                        input_key=key,
                        text=item["text"],
                        voice=item["voice"],
                        cold_wall_time_s=float(cold_result["wall_time_s"]),
                        warm_wall_times_s=[float(row["wall_time_s"]) for row in warm],
                        canonical_audio_duration_s=float(item["canonical_duration_s"]),
                        observed_audio_duration_s=float(warm[-1]["observed_audio_duration_s"]),
                        output_sha256=str(warm[-1].get("output_sha256") or ""),
                        provenance={
                            "binary": str(args.binary),
                            "models_dir": str(args.models_dir),
                            "inputs_dir": str(args.inputs_dir),
                            "hnsf_weights": str(args.hnsf_weights),
                            "compute_units": args.compute_units,
                            "batch_mode": True,
                            "preflight_discarded_runs": args.preflight_runs,
                            "use_exact_duration_models": args.use_exact_duration_models,
                            "raw_preflight_results": preflight,
                            "bucket_used": warm[-1].get("bucket_used"),
                            "spotcheck_wav": str(spotcheck_dir / f"{key}.wav"),
                            "raw_last_result": warm[-1],
                            "raw_warm_results": warm,
                        },
                    )
                )
            except Exception as exc:
                records.append(
                    error_record(
                        impl="config-f-reference",
                        framework="Swift + Core ML",
                        hardware_target="ANE/Core ML",
                        version="local",
                        machine_id=args.machine_id,
                        input_key=key,
                        text=item["text"],
                        voice=item["voice"],
                        canonical_audio_duration_s=float(item["canonical_duration_s"]),
                        provenance={
                            "binary": str(args.binary),
                            "models_dir": str(args.models_dir),
                            "inputs_dir": str(args.inputs_dir),
                            "hnsf_weights": str(args.hnsf_weights),
                            "compute_units": args.compute_units,
                            "batch_mode": True,
                            "preflight_discarded_runs": args.preflight_runs,
                            "use_exact_duration_models": args.use_exact_duration_models,
                            "spotcheck_dir": str(spotcheck_dir),
                        },
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
    finally:
        if runner is not None:
            runner.close()

    payload = result_file_payload(
        impl="config-f-reference",
        machine_id=args.machine_id,
        records=records,
        provenance={
            "binary": str(args.binary),
            "models_dir": str(args.models_dir),
            "inputs_dir": str(args.inputs_dir),
            "hnsf_weights": str(args.hnsf_weights),
            "compute_units": args.compute_units,
            "batch_mode": True,
            "preflight_discarded_runs": args.preflight_runs,
            "use_exact_duration_models": args.use_exact_duration_models,
            "spotcheck_dir": str(spotcheck_dir),
        },
    )
    validate_result_payload(payload)
    output = args.output or (DEFAULT_OUTPUT_DIR / f"results_config_f_reference_{args.machine_id}.json")
    write_json(output, payload)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
