#!/usr/bin/env python3
"""Kokoro TTS Bakeoff Harness -- unified benchmark with four modes.

Modes
-----
  prepare-inputs   Measure canonical audio durations with PyTorch CPU; write input manifest.
  run              Timed benchmark iterations for selected configs with preloaded models.
  telemetry-loop   Sustained inference loop for powermetrics observation.
  summarize        Read results files and emit tables plus gate answers.

Configs
-------
  A   Shipping hybrid HAR-post path (3s/10s buckets, explicit path load)
  B   Naive decoder-only 10s artifact, compute_units=ALL
  C   Naive decoder-only 10s artifact, compute_units=CPU_AND_GPU
  D   PyTorch end-to-end on MPS (requires PYTORCH_ENABLE_MPS_FALLBACK=1)
  E   PyTorch end-to-end on CPU
  F   Swift + CoreML pipeline, compute_units=ALL
  G   Swift + CoreML pipeline, compute_units=CPU_AND_GPU

Diagnostic-only (telemetry-loop only, not for ``run --configs``):
  bcpu  Naive decoder-only 10s artifact, compute_units=CPU_ONLY

Usage::

    python scripts/bakeoff_harness.py prepare-inputs
    python scripts/bakeoff_harness.py run --configs a,b,c,d,e,f,g --iterations 5 --order-seed 0
    python scripts/bakeoff_harness.py telemetry-loop --config f --input 30s --seconds 60
    python scripts/bakeoff_harness.py summarize --results outputs/bakeoff/results_m2_ultra.json
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
import uuid
from math import ceil
from pathlib import Path
from typing import Any

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
_BAKEOFF_DIR = _REPO_ROOT / "outputs" / "bakeoff"
_BAKEOFF_MODELS_DIR = _BAKEOFF_DIR / "models"
_INPUT_MANIFEST_PATH = _BAKEOFF_DIR / "input_manifest.json"

# The decoder-only control artifact exported by Phase 2.
DECODER_ONLY_ARTIFACT = str(_BAKEOFF_MODELS_DIR / "kokoro_decoder_only_10s.mlpackage")

# Shipping HAR-post bucket paths — all 5 buckets must be loaded for fair
# comparison with Config F (Swift), which uses all 5.  Previous versions only
# loaded 3s and 10s, causing _select_bucket_seconds to fall back to 10s for
# 7s/15s/30s inputs — a silent workload mismatch.
HAR_POST_BUCKETS: dict[int, str] = {
    3:  str(_REPO_ROOT / "coreml" / "kokoro_decoder_har_post_3s.mlpackage"),
    7:  str(_REPO_ROOT / "coreml" / "kokoro_decoder_har_post_7s.mlpackage"),
    10: str(_REPO_ROOT / "coreml" / "kokoro_decoder_har_post_10s.mlpackage"),
    15: str(_REPO_ROOT / "coreml" / "kokoro_decoder_har_post_15s.mlpackage"),
    30: str(_REPO_ROOT / "coreml" / "kokoro_decoder_har_post_30s.mlpackage"),
}

# ---------------------------------------------------------------------------
# Benchmark Inputs -- frozen text, voice, and speed for every config.
# Texts chosen to hit duration targets within the 10s HAR-post ceiling.
# Starting point: PRESETS from scripts/bench_pipeline_stages.py:38.
# ---------------------------------------------------------------------------
VOICE = "af_heart"
SPEED = 1.0

BAKEOFF_INPUTS: dict[str, str] = {
    "3s": "The quick brown fox jumps over the dog.",
    "7s": (
        "The morning sun cast long shadows across the garden as birds began "
        "their chorus in the ancient oak tree."
    ),
    "15s": (
        "The ancient lighthouse stood alone on the rocky cliff, its beam sweeping "
        "across dark waters with the patience of centuries. Ships had come and gone, "
        "storms had battered its walls, yet still it turned, guiding sailors home."
    ),
    "30s": (
        "When the last train departed that evening, the platform fell silent. "
        "The old stationmaster locked the ticket office with trembling hands, "
        "running his fingers along the worn counter where countless journeys "
        "had begun. Outside, autumn wind scattered golden leaves across the "
        "empty tracks. He had spent forty years here, watching the world rush "
        "past in a blur of faces and farewells. The station would stand a while "
        "longer, its clock still ticking, its roof sheltering the pigeons."
    ),
}

# Maximum canonical audio duration (seconds).  prepare-inputs hard-fails above this.
MAX_CANONICAL_DURATION_S = 30.0
CONFIG_F_DURATION_TOLERANCE_FRACTION = 0.15

# Headline config IDs valid for ``run --configs``.
HEADLINE_CONFIGS = {"a", "b", "c", "d", "e", "f", "g"}
# Diagnostic config valid only for ``telemetry-loop --config``.
DIAGNOSTIC_CONFIGS = {"bcpu"}
ALL_CONFIGS = HEADLINE_CONFIGS | DIAGNOSTIC_CONFIGS

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _sha256_file(path: str) -> str:
    """Return hex SHA-256 of a file (or key internal file for .mlpackage dirs)."""
    p = Path(path)
    if p.is_dir():
        # For .mlpackage directories, hash the weights file if present.
        weights = p / "Data" / "com.apple.CoreML" / "weights" / "weight.bin"
        if weights.exists():
            return _sha256_file(str(weights))
        spec = p / "Data" / "com.apple.CoreML" / "model.mlmodel"
        if spec.exists():
            return _sha256_file(str(spec))
        return "directory_no_hashable_file"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def _sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(_REPO_ROOT), text=True
        ).strip()
    except Exception:
        return "unknown"

def _git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=str(_REPO_ROOT), text=True
        ).strip()
        return bool(out)
    except Exception:
        return True

def _machine_info() -> dict:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
    }
    try:
        info["sw_vers"] = subprocess.check_output(["sw_vers"], text=True).strip()
    except Exception:
        info["sw_vers"] = "unknown"
    try:
        info["cpu_brand"] = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    except Exception:
        info["cpu_brand"] = "unknown"
    try:
        mem = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
        info["memory_gb"] = round(mem / (1024 ** 3), 1)
    except Exception:
        info["memory_gb"] = "unknown"
    return info

def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for pkg in ["torch", "torchaudio", "coremltools", "numpy", "transformers"]:
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except Exception:
            versions[pkg] = "not_installed"
    return versions

def _select_bucket_seconds_standalone(
    total_seconds: float, available_buckets: list[int]
) -> int | None:
    """Replicate _select_bucket_seconds logic without a pipeline instance.

    Picks the smallest available bucket >= ceil(total_seconds).
    Matches coreml_pipeline.py:_select_bucket_seconds exactly.
    """
    available = sorted(available_buckets)
    threshold = int(ceil(total_seconds))
    for sec in available:
        if sec >= threshold:
            return sec
    return available[-1] if available else None

def _hnsf_weights_sha256(path: Path) -> str | None:
    try:
        data = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if "linear_weights" not in data or "linear_bias" not in data:
        return None
    payload = json.dumps(
        {"linear_weights": data["linear_weights"], "linear_bias": data["linear_bias"]},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def _swift_input_staleness(input_json: Path, hnsf_weights: Path, manifest_entry: dict[str, Any]) -> str | None:
    """Return a mismatch reason if a prepared Swift input does not match the manifest."""
    try:
        prepared = json.loads(input_json.read_text())
    except json.JSONDecodeError as exc:
        return f"invalid JSON in {input_json}: {exc}"

    for field in ("text", "voice"):
        expected = manifest_entry.get(field)
        if expected is not None and prepared.get(field) != expected:
            return f"{input_json.name} {field} does not match manifest"

    expected_speed = manifest_entry.get("speed")
    if expected_speed is not None and abs(float(prepared.get("speed", -1)) - float(expected_speed)) > 1e-6:
        return f"{input_json.name} speed does not match manifest"

    expected_duration = manifest_entry.get("canonical_duration_s")
    prepared_duration = prepared.get("canonical_duration_s")
    if (
        isinstance(expected_duration, (int, float))
        and isinstance(prepared_duration, (int, float))
        and abs(float(prepared_duration) - float(expected_duration)) > 1e-3
    ):
        return f"{input_json.name} canonical duration does not match manifest"

    prepared_hnsf_hash = prepared.get("hnsf_weights_sha256")
    current_hnsf_hash = _hnsf_weights_sha256(hnsf_weights)
    if not prepared_hnsf_hash or prepared_hnsf_hash != current_hnsf_hash:
        return f"{input_json.name} does not match current hn-nsf weights"

    return None

# ---------------------------------------------------------------------------
# prepare-inputs mode
# ---------------------------------------------------------------------------

def cmd_prepare_inputs(args: argparse.Namespace) -> None:
    """Measure canonical audio durations with PyTorch CPU and write input manifest."""
    _BAKEOFF_DIR.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(_REPO_ROOT))
    from kokoro.coreml_pipeline import HybridTTSPipeline
    from kokoro.model import KModel
    from kokoro.pipeline import KPipeline

    print("=" * 60)
    print("BAKEOFF: prepare-inputs")
    print("=" * 60)

    print("\n Loading PyTorch model on CPU...")
    torch.manual_seed(0)
    kmodel = KModel().to("cpu")
    kmodel.eval()
    kpipeline = KPipeline(lang_code="a", model=False)

    # Minimal stand-in that only runs extract_vocoder_inputs on CPU.
    # Avoids loading CoreML models, which may not exist yet.
    class _CPUBenchPipeline:
        def __init__(self, km, kp):
            self.pytorch_model = km
            self.pipeline = kp
            self.coreml_synth_buckets = {}
            self.coreml_decoder_har_buckets = {}
            self.coreml_decoder_har_post_buckets = {sec: None for sec in HAR_POST_BUCKETS}

        extract_vocoder_inputs = HybridTTSPipeline.extract_vocoder_inputs
        _select_bucket_seconds = HybridTTSPipeline._select_bucket_seconds

    pipe = _CPUBenchPipeline(kmodel, kpipeline)

    entries: dict[str, dict] = {}
    any_failure = False

    for input_key, text in BAKEOFF_INPUTS.items():
        snippet = repr(text)[:60]
        ellipsis = "..." if len(text) > 60 else ""
        print(f"\n--- {input_key}: {snippet}{ellipsis}")
        torch.manual_seed(0)
        vi = pipe.extract_vocoder_inputs(text, VOICE, SPEED)
        if vi is None:
            print(f"  extract_vocoder_inputs returned None for {input_key!r}")
            any_failure = True
            continue

        T_f0 = int(vi["f0_curve"].shape[-1])
        canonical_duration_s = T_f0 / 80.0
        expected_bucket = _select_bucket_seconds_standalone(canonical_duration_s, sorted(HAR_POST_BUCKETS))

        print(f"  T_f0={T_f0}, canonical_duration={canonical_duration_s:.3f}s, bucket={expected_bucket}s")

        # Hard-fail if duration exceeds ceiling.
        if canonical_duration_s > MAX_CANONICAL_DURATION_S:
            print(
                f"  FAIL: canonical duration {canonical_duration_s:.3f}s "
                f"exceeds {MAX_CANONICAL_DURATION_S}s ceiling."
            )
            sys.exit(1)

        # Bucket-boundary assertion: input key matches expected bucket.
        if input_key in ("3s", "7s", "15s", "30s"):
            target_bucket = int(input_key.rstrip("s"))
            if expected_bucket != target_bucket:
                print(
                    f"  WARNING: '{input_key}' routes to {expected_bucket}s bucket "
                    f"instead of target {target_bucket}s (canonical={canonical_duration_s:.3f}s)."
                )

        # Smoke assertion: bucket must be in the available set.
        available_buckets = set(HAR_POST_BUCKETS)
        if expected_bucket not in available_buckets:
            print(
                f"  FAIL: expected bucket {expected_bucket}s is not in {sorted(available_buckets)} -- "
                f"no model available for this duration."
            )
            sys.exit(1)

        entries[input_key] = {
            "text": text,
            "voice": VOICE,
            "speed": SPEED,
            "canonical_duration_s": round(canonical_duration_s, 6),
            "expected_bucket_s": expected_bucket,
            "T_f0": T_f0,
            "text_sha256": _sha256_str(text),
        }

    if any_failure:
        print("\nABORT: one or more inputs failed extract_vocoder_inputs.")
        sys.exit(1)

    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "voice": VOICE,
        "speed": SPEED,
        "git_commit": _git_commit(),
        "python_executable": sys.executable,
        "inputs": entries,
    }

    _INPUT_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\n Input manifest written: {_INPUT_MANIFEST_PATH}")
    print(f"   {len(entries)} inputs frozen.")
    for k, v in entries.items():
        print(f"   {k}: {v['canonical_duration_s']:.3f}s -> {v['expected_bucket_s']}s bucket")

# ---------------------------------------------------------------------------
# Benchmark Contexts — preload everything before timing.
# ---------------------------------------------------------------------------

class ConfigAContext:
    """Shipping hybrid HAR-post path with explicit artifact loading.

    Loads HybridTTSPipeline for PyTorch text-processing, then overrides
    coreml_decoder_har_post_buckets with explicit MLModel loads by path
    so the benchmark uses known artifacts (with recorded SHA256).
    """

    config_id = "a"

    def __init__(self) -> None:
        import coremltools as ct
        from kokoro.coreml_pipeline import HybridTTSPipeline

        self.available = True
        self.unavailable_reason = ""
        self.artifacts: dict[str, dict] = {}

        # Check that all 5 HAR-post bucket artifacts exist.
        for sec, path in HAR_POST_BUCKETS.items():
            if not Path(path).exists():
                self.available = False
                self.unavailable_reason = f"HAR-post {sec}s artifact not found: {path}"
                return

        # Config A only needs the PyTorch prefix plus the HAR-post buckets
        # below.  Do not let HybridTTSPipeline auto-discover every Core ML
        # package in coreml/: stale diagnostic packages can spend minutes in
        # Core ML AOT compilation before the benchmark even reaches its
        # availability summary.
        self.pipe = HybridTTSPipeline(force_engine="pytorch")

        # Override with explicit loads of all 5 buckets so Config A uses the
        # same bucket set as Config F (fair comparison).
        self.pipe.coreml_decoder_har_post_buckets = {
            sec: ct.models.MLModel(path, compute_units=ct.ComputeUnit.ALL)
            for sec, path in HAR_POST_BUCKETS.items()
        }
        self.pipe.use_coreml = True

        # Record artifact metadata.
        for sec, path in HAR_POST_BUCKETS.items():
            label = f"config_a_har_post_{sec}s"
            model = self.pipe.coreml_decoder_har_post_buckets[sec]
            spec = model.get_spec()
            shapes = {
                i.name: list(i.type.multiArrayType.shape)
                for i in spec.description.input
            }
            self.artifacts[label] = {
                "path": path,
                "sha256": _sha256_file(path),
                "compute_units": "ALL",
                "input_shapes": shapes,
            }

    def warmup(self, text: str) -> None:
        """Run one untimed iteration for JIT warmup."""
        from kokoro.synthesis_backends import decoder_har_post_bucket_impl
        torch.manual_seed(0)
        decoder_har_post_bucket_impl(self.pipe, text, VOICE, SPEED)

class DecoderOnlyContext:
    """Naive decoder-only 10s artifact loaded with a specific compute_units policy.

    Shares the HybridTTSPipeline for extract_vocoder_inputs() (shared prefix).
    """

    def __init__(self, compute_units_str: str, config_id: str, shared_pipe=None) -> None:
        import coremltools as ct

        self.config_id = config_id
        self.available = True
        self.unavailable_reason = ""
        self.artifacts: dict[str, dict] = {}
        self.compute_units_str = compute_units_str

        CU_MAP = {
            "ALL": ct.ComputeUnit.ALL,
            "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
            "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
        }
        cu = CU_MAP.get(compute_units_str)
        if cu is None:
            self.available = False
            self.unavailable_reason = f"Unknown compute_units: {compute_units_str}"
            return

        if not Path(DECODER_ONLY_ARTIFACT).exists():
            self.available = False
            self.unavailable_reason = (
                f"Decoder-only artifact not found: {DECODER_ONLY_ARTIFACT}. "
                f"Run Phase 2 export first."
            )
            return

        # Reuse shared pipeline for extract_vocoder_inputs.
        if shared_pipe is not None:
            self.pipe = shared_pipe
        else:
            from kokoro.coreml_pipeline import HybridTTSPipeline
            self.pipe = HybridTTSPipeline()

        try:
            self.model = ct.models.MLModel(DECODER_ONLY_ARTIFACT, compute_units=cu)
        except Exception as exc:
            self.available = False
            self.unavailable_reason = f"Failed to load decoder-only artifact: {exc}"
            return

        spec = self.model.get_spec()
        shapes = {
            i.name: list(i.type.multiArrayType.shape)
            for i in spec.description.input
        }
        artifact_key = f"config_{config_id}_decoder_{compute_units_str.lower()}"
        self.artifacts[artifact_key] = {
            "path": DECODER_ONLY_ARTIFACT,
            "sha256": _sha256_file(DECODER_ONLY_ARTIFACT),
            "compute_units": compute_units_str,
            "input_shapes": shapes,
        }

    def warmup(self, text: str) -> None:
        torch.manual_seed(0)
        _run_decoder_only(self, text)

class PyTorchContext:
    """PyTorch end-to-end inference on a specific device (cpu or mps)."""

    def __init__(self, device: str, config_id: str) -> None:
        from kokoro.model import KModel
        from kokoro.pipeline import KPipeline

        self.config_id = config_id
        self.available = True
        self.unavailable_reason = ""
        self.artifacts: dict[str, dict] = {}
        self.device = device

        if device == "mps":
            if not torch.backends.mps.is_available():
                self.available = False
                self.unavailable_reason = "MPS not available on this machine."
                return
            if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "1":
                self.available = False
                self.unavailable_reason = (
                    "PYTORCH_ENABLE_MPS_FALLBACK=1 must be set for Config D."
                )
                return

        self.kmodel = KModel().to(device)
        self.kmodel.eval()
        self.kpipeline = KPipeline(lang_code="a", model=False)

    def warmup(self, text: str) -> None:
        torch.manual_seed(0)
        _run_pytorch(self, text)

class SwiftPipelineContext:
    """Swift + CoreML pipeline via persistent kokoro-bench subprocess.

    Uses ``--batch`` mode so CoreML models are compiled once and cached across
    iterations. The subprocess stays alive for the entire bakeoff run. Commands
    are sent as JSON on stdin; the binary writes results to temp files and
    prints "DONE\\n" on stdout when each command completes.
    """

    def __init__(self, compute_units: str = "all", config_id: str = "f") -> None:
        self.config_id = config_id
        self.available = True
        self.unavailable_reason = ""
        self.artifacts: dict[str, dict] = {}
        self._proc: subprocess.Popen | None = None
        self._compute_units = compute_units
        self._timed_timeout_s = float(os.environ.get("KOKORO_SWIFT_TIMED_TIMEOUT_S", "600"))
        self._warmup_timeout_s = float(os.environ.get("KOKORO_SWIFT_WARMUP_TIMEOUT_S", "1800"))

        # Locate the built Swift binary
        self._swift_binary = _REPO_ROOT / "swift" / ".build" / "release" / "kokoro-bench"
        self._models_dir = _REPO_ROOT / "coreml"
        self._inputs_dir = _REPO_ROOT / "outputs" / "swift_bench_inputs"
        self._hnsf_weights = self._inputs_dir / "hnsf_weights.json"

        if not self._swift_binary.exists():
            self.available = False
            self.unavailable_reason = (
                f"Swift binary not found at {self._swift_binary}. "
                "Build with: cd swift && swift build -c release --product kokoro-bench"
            )
            return

        if not self._hnsf_weights.exists():
            self.available = False
            self.unavailable_reason = (
                f"hn-nsf weights not found at {self._hnsf_weights}. "
                "Run: uv run python scripts/prepare_swift_bench_inputs.py"
            )
            return

        self.artifacts[f"config_{config_id}_swift_runtime"] = {
            "swift_binary": str(self._swift_binary),
            "swift_binary_sha256": _sha256_file(str(self._swift_binary)),
            "models_dir": str(self._models_dir),
            "hnsf_weights": str(self._hnsf_weights),
            "hnsf_weights_sha256": _sha256_file(str(self._hnsf_weights)),
            "compute_units": compute_units,
        }

        # Read manifest to get the first input key for warmup
        manifest_path = _REPO_ROOT / "outputs" / "bakeoff" / "input_manifest.json"
        if manifest_path.exists():
            import json as _json
            with open(manifest_path) as f:
                _manifest = _json.load(f)
            self._first_input_key = next(iter(_manifest["inputs"]))
        else:
            self._first_input_key = "3s"  # fallback

    def _ensure_subprocess(self) -> subprocess.Popen:
        """Start the persistent batch subprocess if not already running."""
        if self._proc is not None and self._proc.poll() is None:
            return self._proc

        import subprocess as _sp
        # stderr must NOT be PIPE — the Swift binary writes verbose compilation
        # logs to stderr, and if the pipe buffer fills (64KB) it deadlocks.
        # Let stderr flow to the parent's stderr so the user sees progress.
        self._proc = _sp.Popen(
            [
                str(self._swift_binary),
                "--models-dir", str(self._models_dir),
                "--inputs-dir", str(self._inputs_dir),
                "--hnsf-weights", str(self._hnsf_weights),
                "--batch",
                "--compute-units", self._compute_units,
            ],
            stdin=_sp.PIPE,
            stdout=_sp.PIPE,
            stderr=None,  # inherit parent stderr
        )
        # Wait for "READY" signal from the Swift binary
        ready_line = self._proc.stdout.readline().decode().strip()
        if ready_line != "READY":
            raise RuntimeError(
                f"Swift batch subprocess did not send READY, got: {ready_line!r}"
            )
        print("  Swift batch subprocess started (pid={})".format(self._proc.pid))
        return self._proc

    def _send_command(self, cmd: dict, timeout: float = 300) -> str:
        """Send a JSON command to the batch subprocess and wait for DONE.

        Returns the path to the output file (caller reads it).
        """
        proc = self._ensure_subprocess()
        cmd_json = json.dumps(cmd) + "\n"
        proc.stdin.write(cmd_json.encode())
        proc.stdin.flush()

        # Wait for "DONE" response with timeout
        import select
        import time
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._terminate_subprocess()
                raise TimeoutError(f"Swift batch subprocess timed out after {timeout}s")
            # Use select to wait for stdout with timeout
            ready, _, _ = select.select([proc.stdout], [], [], min(remaining, 1.0))
            if ready:
                line = proc.stdout.readline().decode().strip()
                if line in ("DONE", "ERROR") or line.endswith("DONE") or line.endswith("ERROR"):
                    return cmd["output"]
            # Check if process died
            if proc.poll() is not None:
                raise RuntimeError(
                    f"Swift batch subprocess died (rc={proc.returncode})"
                )

    def _terminate_subprocess(self) -> None:
        """Terminate a stuck Swift subprocess so the next command starts fresh."""
        proc = self._proc
        self._proc = None
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)

    def warmup(self, text: str) -> None:
        """Warmup by running each input key through the persistent subprocess.

        This triggers model compilation and warmup for all buckets, so timed
        runs don't pay the compilation cost.
        """
        import tempfile
        manifest_path = _REPO_ROOT / "outputs" / "bakeoff" / "input_manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
            input_keys = list(manifest["inputs"].keys())
        else:
            input_keys = [self._first_input_key]

        failures: list[str] = []
        for ik in input_keys:
            out_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
            out_file.close()
            try:
                print(f"  Swift warmup: {ik} (compiling models for this bucket)...")
                self._send_command({
                    "input_key": ik,
                    "seed": 0,
                    "output": out_file.name,
                    "warmup": True,
                }, timeout=self._warmup_timeout_s)
                try:
                    record = json.loads(Path(out_file.name).read_text())
                except Exception as exc:
                    raise RuntimeError(f"warmup did not write a valid result JSON: {exc}") from exc
                if record.get("status") != "ok":
                    raise RuntimeError(
                        f"warmup result status={record.get('status')!r}: {record.get('error')}"
                    )
            except Exception as exc:
                print(f"  Swift warmup failed for {ik}: {exc}")
                failures.append(f"{ik}: {exc}")
            finally:
                try:
                    os.unlink(out_file.name)
                except OSError:
                    pass
        if failures:
            raise RuntimeError("Swift warmup failed: " + "; ".join(failures))

    def close(self) -> None:
        """Shut down the persistent subprocess."""
        if self._proc is not None and self._proc.poll() is None:
            self._proc.stdin.close()
            self._proc.wait(timeout=10)
            print(f"  Swift batch subprocess exited (rc={self._proc.returncode})")
            self._proc = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _run_swift_timed(
    ctx: SwiftPipelineContext, input_key: str, manifest_entry: dict
) -> dict[str, Any]:
    """Run a Swift CoreML config via the persistent Swift batch subprocess.

    Models are already compiled and cached in the subprocess — this only
    measures inference time.
    """
    import tempfile

    record: dict[str, Any] = {
        "config": ctx.config_id,
        "input_key": input_key,
        "status": "ok",
        "error": None,
    }

    # Check that the pre-tokenized input exists
    input_json = ctx._inputs_dir / f"{input_key}.json"
    if not input_json.exists():
        record.update(status="input_missing", error=f"No pre-tokenized input for {input_key}")
        return record
    if staleness := _swift_input_staleness(input_json, ctx._hnsf_weights, manifest_entry):
        record.update(status="input_stale", error=staleness)
        return record

    out_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    out_file.close()

    try:
        ctx._send_command({
            "input_key": input_key,
            "seed": 0,
            "output": out_file.name,
            "warmup": False,
        }, timeout=ctx._timed_timeout_s)

        swift_result = json.loads(Path(out_file.name).read_text())
        record.update(swift_result)
        record["config"] = ctx.config_id
        record["input_key"] = input_key
        if record.get("status") == "ok" and not _duration_agreement_ok(
            record.get("observed_audio_duration_s"),
            manifest_entry.get("canonical_duration_s"),
        ):
            record.update(
                status="duration_mismatch",
                error=(
                    f"observed={record.get('observed_audio_duration_s')}s vs "
                    f"canonical={manifest_entry.get('canonical_duration_s')}s"
                ),
            )
        return record

    except TimeoutError:
        record.update(status="timeout", error="Swift batch subprocess timed out")
        return record
    except Exception as exc:
        record.update(status="exception", error=str(exc))
        return record
    finally:
        try:
            os.unlink(out_file.name)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Per-config run functions
# ---------------------------------------------------------------------------

def _run_config_a_timed(
    ctx: ConfigAContext, text: str, manifest_entry: dict
) -> dict[str, Any]:
    """Instrumented replica of decoder_har_post_bucket_impl() with stage timers.

    Wraps the exact logic from synthesis_backends.py:167-214 with
    time.perf_counter() pairs around each stage. Returns a per-iteration record.
    """
    from kokoro.synthesis_backends import build_decoder_har_post_inputs_np

    record: dict[str, Any] = {
        "config": "a",
        "input_key": manifest_entry.get("_input_key"),
        "status": "ok",
        "error": None,
    }

    torch.manual_seed(0)
    wall_start = time.perf_counter()

    # Stage 1: extract_vocoder_inputs (shared prefix).
    t0 = time.perf_counter()
    vi = ctx.pipe.extract_vocoder_inputs(text, VOICE, SPEED)
    t_prefix_extract = time.perf_counter() - t0

    if vi is None:
        wall_end = time.perf_counter()
        record.update(
            status="prefix_failed",
            wall_time_s=wall_end - wall_start,
            t_prefix_extract_s=t_prefix_extract,
        )
        return record

    T_f0 = int(vi["f0_curve"].shape[-1])
    total_seconds = T_f0 / 80.0
    sec = ctx.pipe._select_bucket_seconds(total_seconds)
    if sec is None or sec not in ctx.pipe.coreml_decoder_har_post_buckets:
        wall_end = time.perf_counter()
        record.update(
            status="no_bucket",
            wall_time_s=wall_end - wall_start,
            t_prefix_extract_s=t_prefix_extract,
        )
        return record

    model = ctx.pipe.coreml_decoder_har_post_buckets[sec]
    spec = model.get_spec()
    shapes = {i.name: list(i.type.multiArrayType.shape) for i in spec.description.input}
    asr_len = int(shapes["x_pre"][-1])
    har_t = int(shapes["har"][-1])

    # Stage 2: CPU tensor prep (decoder pre-stack + hn-nsf).
    dec = ctx.pipe.pytorch_model.decoder
    t0 = time.perf_counter()
    x_pre_np, ref_s, har_np, _t_check, _fc, mask_np = build_decoder_har_post_inputs_np(
        dec, vi, sec, asr_len, har_t, warn_geometry=False
    )
    t_har_builder_cpu = time.perf_counter() - t0
    # Must match production function (synthesis_backends.py:197).
    assert _t_check == T_f0, f"geometry mismatch: _t_check={_t_check} vs T_f0={T_f0}"

    # Stage 2b: CPU dict assembly (negligible but counted for completeness).
    t0 = time.perf_counter()
    inputs = {"x_pre": x_pre_np, "ref_s": ref_s, "har": har_np}
    if "mask" in shapes:
        inputs["mask"] = mask_np
    t_decoder_pre_cpu = time.perf_counter() - t0

    # Stage 3: Core ML predict.
    t0 = time.perf_counter()
    res = model.predict(inputs)
    t_coreml_predict = time.perf_counter() - t0

    key = "waveform" if "waveform" in res else list(res.keys())[0]
    audio = np.asarray(res[key], dtype=np.float32).squeeze()

    # Stage 4: Trim to natural length.
    t0 = time.perf_counter()
    target_len = int(round((T_f0 / 80.0) * 24000.0))
    audio = audio[: min(int(audio.shape[-1]), target_len)]
    t_trim = time.perf_counter() - t0

    wall_end = time.perf_counter()
    wall_time = wall_end - wall_start
    canonical_dur = manifest_entry.get("canonical_duration_s", total_seconds)
    observed_dur = len(audio) / 24000.0

    # Orchestration = wall - sum of measured stages (clamped to 0; can go slightly
    # negative due to perf_counter nesting granularity).
    t_orchestration = max(0.0, wall_time - (
        t_prefix_extract + t_har_builder_cpu + t_decoder_pre_cpu + t_coreml_predict + t_trim
    ))

    record.update(
        wall_time_s=round(wall_time, 6),
        canonical_audio_duration_s=round(canonical_dur, 6),
        observed_audio_duration_s=round(observed_dur, 6),
        rtf_canonical=round(wall_time / canonical_dur, 6) if canonical_dur > 0 else None,
        rtf_observed=round(wall_time / observed_dur, 6) if observed_dur > 0 else None,
        speed_vs_realtime_canonical=round(canonical_dur / wall_time, 2) if wall_time > 0 else None,
        bucket_used=f"{sec}s",
        t_prefix_extract_s=round(t_prefix_extract, 6),
        t_decoder_pre_cpu_s=round(t_decoder_pre_cpu, 6),
        t_har_builder_cpu_s=round(t_har_builder_cpu, 6),
        t_coreml_predict_s=round(t_coreml_predict, 6),
        t_trim_s=round(t_trim, 6),
        t_orchestration_s=round(t_orchestration, 6),
    )
    return record

def _run_decoder_only(ctx: DecoderOnlyContext, text: str) -> np.ndarray | None:
    """Run decoder-only inference through shared prefix + decoder-only model."""
    torch.manual_seed(0)
    vi = ctx.pipe.extract_vocoder_inputs(text, VOICE, SPEED)
    if vi is None:
        return None

    # Build decoder-only input bundle from vocoder inputs.
    # The decoder-only model expects: asr, F0_pred, N_pred, ref_s.
    spec = ctx.model.get_spec()
    shapes = {i.name: list(i.type.multiArrayType.shape) for i in spec.description.input}

    asr_shape = shapes.get("asr", shapes.get(list(shapes.keys())[0]))
    frame_count = int(asr_shape[-1])
    f0_len = int(shapes.get("F0_pred", shapes.get(list(shapes.keys())[1]))[-1])

    asr = vi["asr"].astype(np.float32)
    f0 = vi["f0_curve"].astype(np.float32)
    n = vi["n"].astype(np.float32)
    ref_s = vi["ref_s"].astype(np.float32)

    # Pad to model dimensions.
    asr_pad = np.zeros((1, asr.shape[1], frame_count), dtype=np.float32)
    t_asr = min(frame_count, asr.shape[-1])
    asr_pad[:, :, :t_asr] = asr[:, :, :t_asr]

    f0_pad = np.zeros((1, f0_len), dtype=np.float32)
    n_pad = np.zeros((1, f0_len), dtype=np.float32)
    t_f0 = min(f0_len, f0.shape[-1])
    f0_pad[:, :t_f0] = f0[:, :t_f0]
    n_pad[:, :t_f0] = n[:, :t_f0]

    inputs = {"asr": asr_pad, "F0_pred": f0_pad, "N_pred": n_pad, "ref_s": ref_s}
    res = ctx.model.predict(inputs)
    key = "waveform" if "waveform" in res else list(res.keys())[0]
    return np.asarray(res[key], dtype=np.float32).squeeze()

def _run_decoder_only_timed(
    ctx: DecoderOnlyContext, text: str, manifest_entry: dict
) -> dict[str, Any]:
    """Timed wrapper for decoder-only configs (B, C, Bcpu)."""
    record: dict[str, Any] = {
        "config": ctx.config_id,
        "input_key": manifest_entry.get("_input_key"),
        "status": "ok",
        "error": None,
    }
    torch.manual_seed(0)
    wall_start = time.perf_counter()
    audio = _run_decoder_only(ctx, text)
    wall_end = time.perf_counter()

    if audio is None:
        record.update(status="prefix_failed", wall_time_s=round(wall_end - wall_start, 6))
        return record

    wall_time = wall_end - wall_start
    canonical_dur = manifest_entry.get("canonical_duration_s", 0)
    observed_dur = len(audio) / 24000.0

    record.update(
        wall_time_s=round(wall_time, 6),
        canonical_audio_duration_s=round(canonical_dur, 6),
        observed_audio_duration_s=round(observed_dur, 6),
        rtf_canonical=round(wall_time / canonical_dur, 6) if canonical_dur > 0 else None,
        rtf_observed=round(wall_time / observed_dur, 6) if observed_dur > 0 else None,
        speed_vs_realtime_canonical=round(canonical_dur / wall_time, 2) if wall_time > 0 else None,
        bucket_used=None,
        t_prefix_extract_s=None,
        t_decoder_pre_cpu_s=None,
        t_har_builder_cpu_s=None,
        t_coreml_predict_s=None,
        t_trim_s=None,
        t_orchestration_s=None,
    )
    return record

def _run_pytorch(ctx: PyTorchContext, text: str) -> np.ndarray | None:
    """PyTorch end-to-end inference on ctx.device."""
    from kokoro.pipeline import voice_embedding_for_phoneme_string

    torch.manual_seed(0)
    voice_pack = ctx.kpipeline.load_voice(VOICE)
    phonemes = None
    for _, ps, _ in ctx.kpipeline(text, VOICE, SPEED):
        phonemes = ps
        break
    if not phonemes:
        return None

    ref_s = voice_embedding_for_phoneme_string(voice_pack, phonemes)
    ref_s = ref_s.to(ctx.device)

    audio_np = ctx.kmodel(phonemes, ref_s, speed=SPEED)
    if audio_np is None:
        return None

    if ctx.device == "mps":
        torch.mps.synchronize()

    if isinstance(audio_np, torch.Tensor):
        audio_np = audio_np.cpu().numpy()

    return np.asarray(audio_np, dtype=np.float32).squeeze()

def _run_pytorch_timed(
    ctx: PyTorchContext, text: str, manifest_entry: dict
) -> dict[str, Any]:
    """Timed wrapper for PyTorch configs (D, E)."""
    record: dict[str, Any] = {
        "config": ctx.config_id,
        "input_key": manifest_entry.get("_input_key"),
        "status": "ok",
        "error": None,
    }

    torch.manual_seed(0)
    wall_start = time.perf_counter()
    audio = _run_pytorch(ctx, text)
    # _run_pytorch already calls torch.mps.synchronize() for MPS before returning.
    wall_end = time.perf_counter()

    if audio is None:
        record.update(status="inference_failed", wall_time_s=round(wall_end - wall_start, 6))
        return record

    wall_time = wall_end - wall_start
    canonical_dur = manifest_entry.get("canonical_duration_s", 0)
    observed_dur = len(audio) / 24000.0

    record.update(
        wall_time_s=round(wall_time, 6),
        canonical_audio_duration_s=round(canonical_dur, 6),
        observed_audio_duration_s=round(observed_dur, 6),
        rtf_canonical=round(wall_time / canonical_dur, 6) if canonical_dur > 0 else None,
        rtf_observed=round(wall_time / observed_dur, 6) if observed_dur > 0 else None,
        speed_vs_realtime_canonical=round(canonical_dur / wall_time, 2) if wall_time > 0 else None,
        bucket_used=None,
        t_prefix_extract_s=None,
        t_decoder_pre_cpu_s=None,
        t_har_builder_cpu_s=None,
        t_coreml_predict_s=None,
        t_trim_s=None,
        t_orchestration_s=None,
    )
    return record

# ---------------------------------------------------------------------------
# Smoke checks
# ---------------------------------------------------------------------------

def _duration_agreement_ok(observed: Any, canonical: Any) -> bool:
    if not isinstance(observed, (int, float)) or not isinstance(canonical, (int, float)):
        return False
    if observed <= 0 or canonical <= 0:
        return False
    return abs(float(observed) - float(canonical)) / float(canonical) <= CONFIG_F_DURATION_TOLERANCE_FRACTION

def _smoke_check_config_a(ctx: ConfigAContext, text: str) -> None:
    """Verify timed runner agrees with production function.

    1. Produces a finite waveform and same bucket.
    2. Emits a warning when wall time drifts from calling
       decoder_har_post_bucket_impl directly.
    """
    from kokoro.synthesis_backends import decoder_har_post_bucket_impl

    # Run production function.
    torch.manual_seed(0)
    t0 = time.perf_counter()
    prod_audio = decoder_har_post_bucket_impl(ctx.pipe, text, VOICE, SPEED)
    prod_time = time.perf_counter() - t0

    assert prod_audio is not None, "Production function returned None during smoke check"
    assert np.isfinite(prod_audio).all(), "Production function returned non-finite audio"

    # Run timed replica.
    manifest_entry = {"_input_key": "smoke", "canonical_duration_s": len(prod_audio) / 24000.0}
    torch.manual_seed(0)
    record = _run_config_a_timed(ctx, text, manifest_entry)

    assert record["status"] == "ok", f"Timed runner failed: {record.get('error', record['status'])}"
    assert record["bucket_used"] is not None, "Timed runner did not select a bucket"

    # Wall-time agreement is a smoke signal, not a correctness gate. Core ML
    # residual compile/cache state can make either side faster on a single pair.
    timed_wall = record["wall_time_s"]
    ratio = timed_wall / prod_time if prod_time > 0 else float("inf")
    if not 0.25 <= ratio <= 4.0:
        print(
            f"  WARNING: Config A wall-time drift: timed={timed_wall:.4f}s vs "
            f"prod={prod_time:.4f}s (ratio={ratio:.3f})"
        )
    print(f"  Config A smoke check passed: timed={timed_wall:.4f}s, prod={prod_time:.4f}s, ratio={ratio:.3f}")

def _smoke_check_swift_config(ctx: SwiftPipelineContext, manifest: dict) -> None:
    """Verify a Swift config does not mark invalid-duration audio as successful."""
    config_label = ctx.config_id.upper()
    for input_key, entry in manifest["inputs"].items():
        record = _run_swift_timed(ctx, input_key, {**entry, "_input_key": input_key})
        if record["status"] != "ok":
            raise RuntimeError(
                f"Config {config_label} smoke failed for {input_key}: "
                f"{record.get('status')} {record.get('error')}"
            )
        if not _duration_agreement_ok(
            record.get("observed_audio_duration_s"),
            entry.get("canonical_duration_s"),
        ):
            raise RuntimeError(
                f"Config {config_label} smoke duration mismatch for {input_key}: "
                f"observed={record.get('observed_audio_duration_s')}s, "
                f"canonical={entry.get('canonical_duration_s')}s"
            )
        if record.get("wall_time_s", 0) <= 0:
            raise RuntimeError(f"Config {config_label} smoke has no wall time for {input_key}")
    print(f"  Config {config_label} smoke check passed: {len(manifest['inputs'])} input(s)")

# ---------------------------------------------------------------------------
# run mode
# ---------------------------------------------------------------------------

def _load_manifest() -> dict:
    """Load and validate the input manifest."""
    if not _INPUT_MANIFEST_PATH.exists():
        print(f"Input manifest not found: {_INPUT_MANIFEST_PATH}")
        print("Run 'prepare-inputs' first.")
        sys.exit(1)
    return json.loads(_INPUT_MANIFEST_PATH.read_text())

def cmd_run(args: argparse.Namespace) -> None:
    """Run timed benchmark iterations for selected configs."""
    _BAKEOFF_DIR.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(_REPO_ROOT))

    manifest = _load_manifest()
    requested = [c.strip().lower() for c in args.configs.split(",")]

    # Validate config IDs.
    for c in requested:
        if c not in HEADLINE_CONFIGS:
            if c in DIAGNOSTIC_CONFIGS:
                print(f"Config '{c}' is diagnostic-only (use telemetry-loop instead).")
            else:
                print(f"Unknown config: '{c}'. Valid: {sorted(HEADLINE_CONFIGS)}")
            sys.exit(1)

    iterations = args.iterations
    order_seed = args.order_seed

    print("=" * 60)
    print("BAKEOFF: run")
    print(f"  configs: {requested}")
    print(f"  iterations: {iterations}")
    print(f"  order_seed: {order_seed}")
    print("=" * 60)

    # --- Build contexts ---
    contexts: dict[str, Any] = {}
    shared_pipe = None  # Reuse pipeline across decoder-only loads.

    if "a" in requested:
        print("\nLoading Config A (HAR-post)...")
        ctx_a = ConfigAContext()
        contexts["a"] = ctx_a
        shared_pipe = ctx_a.pipe if ctx_a.available else None

    if "b" in requested:
        print("\nLoading Config B (decoder-only, ALL)...")
        contexts["b"] = DecoderOnlyContext("ALL", "b", shared_pipe=shared_pipe)

    if "c" in requested:
        print("\nLoading Config C (decoder-only, CPU_AND_GPU)...")
        contexts["c"] = DecoderOnlyContext("CPU_AND_GPU", "c", shared_pipe=shared_pipe)

    if "d" in requested:
        print("\nLoading Config D (PyTorch MPS)...")
        contexts["d"] = PyTorchContext("mps", "d")

    if "e" in requested:
        print("\nLoading Config E (PyTorch CPU)...")
        contexts["e"] = PyTorchContext("cpu", "e")

    if "f" in requested:
        print("\nLoading Config F (Swift + CoreML pipeline, ALL)...")
        contexts["f"] = SwiftPipelineContext(compute_units="all", config_id="f")

    if "g" in requested:
        print("\nLoading Config G (Swift + CoreML pipeline, CPU_AND_GPU)...")
        contexts["g"] = SwiftPipelineContext(compute_units="cpuAndGPU", config_id="g")

    # --- Print availability summary ---
    print("\n" + "-" * 40)
    print("Config availability:")
    for c in requested:
        ctx = contexts.get(c)
        if ctx is None:
            print(f"  {c}: NOT LOADED")
        elif ctx.available:
            print(f"  {c}: READY")
        else:
            print(f"  {c}: UNAVAILABLE -- {ctx.unavailable_reason}")
    print("-" * 40)

    # --- Warmup (must happen before smoke check so both paths are warm) ---
    print("\nWarming up contexts...")
    warmup_text = list(manifest["inputs"].values())[0]["text"]
    for c in requested:
        ctx = contexts.get(c)
        if ctx and ctx.available:
            torch.manual_seed(0)
            print(f"  Warming up {c}...")
            try:
                ctx.warmup(warmup_text)
            except Exception as exc:
                print(f"  Warmup failed for {c}: {exc}")
                ctx.available = False
                ctx.unavailable_reason = f"Warmup failed: {exc}"

    # --- Pre-run Config A drift check (after warmup so both paths are warm) ---
    if "a" in contexts and contexts["a"].available:
        if os.environ.get("BAKEOFF_SKIP_SMOKE") == "1":
            print("\nSkipping Config A smoke check (BAKEOFF_SKIP_SMOKE=1)")
        else:
            print("\nRunning Config A smoke check...")
            _smoke_check_config_a(contexts["a"], warmup_text)

    for swift_config in ("f", "g"):
        if swift_config not in contexts or not contexts[swift_config].available:
            continue
        if os.environ.get("BAKEOFF_SKIP_SMOKE") == "1":
            print(f"\nSkipping Config {swift_config.upper()} smoke check (BAKEOFF_SKIP_SMOKE=1)")
        else:
            print(f"\nRunning Config {swift_config.upper()} smoke check...")
            _smoke_check_swift_config(contexts[swift_config], manifest)

    # --- Timed iterations with independently shuffled config and input order ---
    # Note: configs and inputs are shuffled separately per plan spec. All inputs
    # for one config run before moving to the next config within a repetition.
    input_keys = list(manifest["inputs"].keys())
    results: list[dict] = []
    execution_order: list[dict] = []

    for rep in range(iterations):
        rng = random.Random(order_seed + rep)
        config_order = list(requested)
        input_order = list(input_keys)
        rng.shuffle(config_order)
        rng.shuffle(input_order)
        execution_order.append({"repetition": rep, "config_order": config_order, "input_order": input_order})

        for cfg_id in config_order:
            ctx = contexts.get(cfg_id)
            if not ctx or not ctx.available:
                # Record unavailable sentinel.
                for ik in input_order:
                    results.append({
                        "config": cfg_id,
                        "input_key": ik,
                        "iteration": rep,
                        "status": "config_unavailable",
                        "error": ctx.unavailable_reason if ctx else "not loaded",
                        "wall_time_s": None,
                        "canonical_audio_duration_s": manifest["inputs"][ik]["canonical_duration_s"],
                        "observed_audio_duration_s": None,
                        "rtf_canonical": None, "rtf_observed": None,
                        "speed_vs_realtime_canonical": None,
                        "bucket_used": None,
                        "t_prefix_extract_s": None, "t_decoder_pre_cpu_s": None,
                        "t_har_builder_cpu_s": None, "t_coreml_predict_s": None,
                        "t_trim_s": None, "t_orchestration_s": None,
                    })
                continue

            for ik in input_order:
                entry = manifest["inputs"][ik]
                entry_with_key = {**entry, "_input_key": ik}
                text = entry["text"]

                try:
                    if cfg_id == "a":
                        rec = _run_config_a_timed(ctx, text, entry_with_key)
                    elif cfg_id in ("b", "c"):
                        rec = _run_decoder_only_timed(ctx, text, entry_with_key)
                    elif cfg_id in ("d", "e"):
                        rec = _run_pytorch_timed(ctx, text, entry_with_key)
                    elif cfg_id in ("f", "g"):
                        rec = _run_swift_timed(ctx, ik, entry_with_key)
                    else:
                        rec = {"config": cfg_id, "status": "unknown_config", "error": cfg_id}
                except Exception as exc:
                    rec = {
                        "config": cfg_id,
                        "input_key": ik,
                        "status": "exception",
                        "error": str(exc),
                        "wall_time_s": None,
                    }

                rec["iteration"] = rep
                results.append(rec)
                status_char = "." if rec.get("status") == "ok" else "X"
                print(status_char, end="", flush=True)

            # GC between configs.
            gc.collect()
            if cfg_id == "d" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()

        print(f" [rep {rep}]")

    # --- Clean up persistent subprocesses ---
    for ctx in contexts.values():
        if ctx and hasattr(ctx, "close"):
            try:
                ctx.close()
            except Exception:
                pass

    # --- Collect artifacts from all contexts ---
    all_artifacts: dict[str, dict] = {}
    for ctx in contexts.values():
        if ctx and hasattr(ctx, "artifacts"):
            all_artifacts.update(ctx.artifacts)

    # --- Build results JSON ---
    machine_id = args.machine_id
    if machine_id is None:
        mi = _machine_info()
        machine_id = mi.get("cpu_brand", "unknown").replace(" ", "_").lower()[:30]

    output = {
        "run_id": str(uuid.uuid4()),
        "order_seed": order_seed,
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "python_executable": sys.executable,
        "machine": _machine_info(),
        "package_versions": _package_versions(),
        "artifacts": all_artifacts,
        "inputs": manifest["inputs"],
        "execution_order": execution_order,
        "results": results,
    }

    out_path = _BAKEOFF_DIR / f"results_{machine_id}.json"
    out_path.write_text(json.dumps(output, indent=2, default=str) + "\n")
    print(f"\nResults written: {out_path}")
    print(f"  {len(results)} iteration records across {len(requested)} configs x {len(input_keys)} inputs x {iterations} reps")

# ---------------------------------------------------------------------------
# telemetry-loop mode
# ---------------------------------------------------------------------------

def cmd_telemetry_loop(args: argparse.Namespace) -> None:
    """Sustained inference loop for powermetrics observation."""
    sys.path.insert(0, str(_REPO_ROOT))
    manifest = _load_manifest()

    config_id = args.config.strip().lower()
    input_key = getattr(args, "input").strip()
    seconds = args.seconds

    if config_id not in ALL_CONFIGS:
        print(f"Unknown config: '{config_id}'. Valid for telemetry-loop: {sorted(ALL_CONFIGS)}")
        sys.exit(1)

    if input_key not in manifest["inputs"]:
        print(f"Unknown input key: '{input_key}'. Valid: {list(manifest['inputs'].keys())}")
        sys.exit(1)

    text = manifest["inputs"][input_key]["text"]

    print("=" * 60)
    print(f"BAKEOFF: telemetry-loop (config={config_id}, input={input_key}, seconds={seconds})")
    print("=" * 60)

    # Build context.
    if config_id == "a":
        ctx = ConfigAContext()
    elif config_id == "b":
        ctx = DecoderOnlyContext("ALL", "b")
    elif config_id == "c":
        ctx = DecoderOnlyContext("CPU_AND_GPU", "c")
    elif config_id == "bcpu":
        ctx = DecoderOnlyContext("CPU_ONLY", "bcpu")
    elif config_id == "d":
        ctx = PyTorchContext("mps", "d")
    elif config_id == "e":
        ctx = PyTorchContext("cpu", "e")
    elif config_id == "f":
        ctx = SwiftPipelineContext(compute_units="all", config_id="f")
    elif config_id == "g":
        ctx = SwiftPipelineContext(compute_units="cpuAndGPU", config_id="g")
    else:
        print(f"Unimplemented config: {config_id}")
        sys.exit(1)

    if not ctx.available:
        print(f"Config {config_id} unavailable: {ctx.unavailable_reason}")
        sys.exit(1)

    # Warmup.
    print(f"Warming up config {config_id}...")
    ctx.warmup(text)

    # Sustained loop.
    print(f"\nStarting sustained loop for {seconds}s...")
    print("(Start powermetrics in another terminal now if not already running.)")
    iteration = 0
    t_end = time.time() + seconds
    while time.time() < t_end:
        torch.manual_seed(0)
        if config_id == "a":
            from kokoro.synthesis_backends import decoder_har_post_bucket_impl
            decoder_har_post_bucket_impl(ctx.pipe, text, VOICE, SPEED)
        elif config_id in ("b", "c", "bcpu"):
            _run_decoder_only(ctx, text)
        elif config_id in ("d", "e"):
            _run_pytorch(ctx, text)
        elif config_id in ("f", "g"):
            _run_swift_timed(ctx, input_key, {**manifest["inputs"][input_key], "_input_key": input_key})
        iteration += 1
        if iteration % 5 == 0:
            print(f"  iteration {iteration}, elapsed={time.time() - (t_end - seconds):.1f}s")

    if hasattr(ctx, "close"):
        ctx.close()

    print(f"\nTelemetry loop complete: {iteration} iterations in {seconds}s")

# ---------------------------------------------------------------------------
# summarize mode (delegated to bakeoff_summarize.py per LOC guard)
# ---------------------------------------------------------------------------

def cmd_summarize(args: argparse.Namespace) -> None:
    """Delegate to bakeoff_summarize.py (zero coupling to benchmark contexts)."""
    from bakeoff_summarize import cmd_summarize as _summarize
    _summarize(args)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Kokoro TTS Bakeoff Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # prepare-inputs
    sub.add_parser(
        "prepare-inputs",
        help="Measure canonical durations and write input manifest",
    )

    # run
    p_run = sub.add_parser("run", help="Run timed benchmark iterations")
    p_run.add_argument(
        "--configs", required=True, help="Comma-separated config IDs (a,b,c,d,e,f,g)"
    )
    p_run.add_argument(
        "--iterations", type=int, default=5, help="Iterations per config/input pair"
    )
    p_run.add_argument(
        "--order-seed", type=int, default=0, help="Deterministic order seed"
    )
    p_run.add_argument(
        "--machine-id", default=None, help="Machine identifier for output filename"
    )

    # telemetry-loop
    p_tl = sub.add_parser(
        "telemetry-loop", help="Sustained inference loop for powermetrics"
    )
    p_tl.add_argument(
        "--config", required=True, help="Single config ID (a,b,c,d,e,f,g,bcpu)"
    )
    p_tl.add_argument(
        "--input", required=True,
        help="Input key from manifest (3s, 7s, 15s, 30s)",
    )
    p_tl.add_argument(
        "--seconds", type=int, default=60, help="Duration of the loop in seconds"
    )

    # summarize
    p_sum = sub.add_parser(
        "summarize", help="Read results files and emit tables + gate answers"
    )
    p_sum.add_argument(
        "--results", required=True, nargs="+", help="Path(s) to results JSON files"
    )

    args = parser.parse_args()

    if args.mode == "prepare-inputs":
        cmd_prepare_inputs(args)
    elif args.mode == "run":
        cmd_run(args)
    elif args.mode == "telemetry-loop":
        cmd_telemetry_loop(args)
    elif args.mode == "summarize":
        cmd_summarize(args)

if __name__ == "__main__":
    main()
