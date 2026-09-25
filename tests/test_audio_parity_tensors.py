"""Tests for audio parity tensor dump and comparison helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(name: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_io = _load_script_module("audio_parity_tensor_io")
_compare = _load_script_module("compare_audio_parity_tensors")


class _FakeModel:
    def __init__(self, inputs: dict[str, tuple[int, ...]]):
        descriptions = [
            SimpleNamespace(
                name=name,
                type=SimpleNamespace(multiArrayType=SimpleNamespace(shape=shape)),
            )
            for name, shape in inputs.items()
        ]
        self._spec = SimpleNamespace(description=SimpleNamespace(input=descriptions))

    def get_spec(self):
        return self._spec


def test_tensor_dump_round_trips_float_and_int(tmp_path: Path) -> None:
    writer = _io.TensorDumpWriter(tmp_path, metadata={"producer": "test"})
    writer.write("tokens", np.array([[1, 2, 3]], dtype=np.int64))
    writer.write("waveform", np.array([0.0, 0.25, -0.5], dtype=np.float64))
    manifest_path = writer.close()

    manifest, tensors = _io.load_tensor_dump(tmp_path)

    assert manifest_path.name == "tensor_manifest.json"
    assert manifest["metadata"]["producer"] == "test"
    assert tensors["tokens"].dtype == np.dtype("<i4")
    assert tensors["tokens"].tolist() == [[1, 2, 3]]
    assert tensors["waveform"].dtype == np.dtype("<f4")
    np.testing.assert_allclose(tensors["waveform"], [0.0, 0.25, -0.5])


def test_mask_aware_inputs_adds_binary_generator_prefix() -> None:
    model = _FakeModel({"x_pre": (1, 512, 8), "mask": (1, 1, 8)})
    tensors = {"pred_dur_valid": np.array([[1, 2]], dtype=np.int32)}

    inputs = _io.mask_aware_inputs(model, {"x_pre": np.zeros((1, 512, 8))}, tensors)

    assert inputs["mask"].flatten().tolist() == [1, 1, 1, 1, 1, 1, 0, 0]


def test_mask_aware_inputs_preserves_legacy_maskless_artifact() -> None:
    model = _FakeModel({"x_pre": (1, 512, 8)})
    original = {"x_pre": np.zeros((1, 512, 8))}

    assert _io.mask_aware_inputs(model, original, {}) is original


def test_compare_tensor_reports_shape_mismatch() -> None:
    result = _compare.compare_tensor(
        "x_pre",
        np.zeros((1, 2, 3), dtype=np.float32),
        np.zeros((1, 2, 4), dtype=np.float32),
        max_abs=1e-3,
        min_corr=0.999,
    )

    assert result["status"] == "shape_mismatch"
    assert result["reference_shape"] == [1, 2, 3]
    assert result["candidate_shape"] == [1, 2, 4]


def test_compare_tensor_accepts_correlated_float_arrays() -> None:
    reference = np.linspace(-1.0, 1.0, 64, dtype=np.float32)
    candidate = reference + 1e-2

    result = _compare.compare_tensor(
        "har",
        reference,
        candidate,
        max_abs=1e-4,
        min_corr=0.999,
    )

    assert result["status"] == "pass"
    assert result["max_abs_error"] > 1e-4
    assert result["correlation"] > 0.999


def test_generator_reference_uses_same_mask_as_coreml_candidate() -> None:
    """The checker's PyTorch reference must see the mask the package gets.

    Regression: the reference was the dump's unmasked waveform while the
    package received a mask, a false 6-8 dB failure on every bucket.
    """
    checker = _load_script_module("check_coreml_generator_from_dump")
    model = _FakeModel({"x_pre": (1, 512, 8), "ref_s": (1, 256), "har": (1, 22, 8), "mask": (1, 1, 8)})
    tensors = {"pred_dur_valid": np.array([[1, 2]], dtype=np.int32)}
    base = {
        "x_pre": np.zeros((1, 512, 8), dtype=np.float32),
        "ref_s": np.zeros((1, 256), dtype=np.float32),
        "har": np.zeros((1, 22, 8), dtype=np.float32),
    }
    candidate_inputs = _io.mask_aware_inputs(model, base, tensors)
    seen: dict[str, np.ndarray] = {}

    def generator(**kwargs):
        seen.update({name: value.numpy() for name, value in kwargs.items()})
        return kwargs["x_pre"][:, :1, :]

    checker.pytorch_reference(generator, candidate_inputs)

    assert set(seen) == set(candidate_inputs)
    np.testing.assert_array_equal(seen["mask"], candidate_inputs["mask"])


def test_capture_reference_stages_are_masked_like_the_runtime() -> None:
    """The dump's F0/N and ``x_pre`` must not depend on bucket padding.

    Regression: the capture ran F0Ntrain and decoder-pre unmasked while the
    runtime masks both stages, so the Python dump carried padding-contaminated
    statistics that a correct Swift/Core ML run could never match.
    """
    torch = pytest.importorskip("torch")
    from kokoro import KModel

    capture = _load_script_module("capture_audio_parity_tensors")
    try:
        kmodel = KModel(disable_complex=True).eval()
    except Exception as exc:
        pytest.skip(f"KModel load failed: {exc}")

    valid = 24
    rng = np.random.default_rng(0)
    en = rng.standard_normal((1, 640, valid)).astype(np.float32)
    asr = rng.standard_normal((1, 512, valid)).astype(np.float32)
    s = rng.standard_normal((1, 128)).astype(np.float32)
    ref_s = rng.standard_normal((1, 256)).astype(np.float32)

    def run(t_frames: int):
        with torch.no_grad():
            return capture.masked_f0n_and_x_pre(
                kmodel,
                capture._pad_time(en, t_frames),
                s,
                capture._pad_time(asr, t_frames),
                ref_s,
                valid,
                2 * t_frames,
            )

    f0_a, n_a, _, _, x_pre_a = run(32)
    f0_b, n_b, _, _, x_pre_b = run(96)

    np.testing.assert_allclose(f0_a[..., : 2 * valid], f0_b[..., : 2 * valid], atol=1e-3)
    np.testing.assert_allclose(n_a[..., : 2 * valid], n_b[..., : 2 * valid], atol=1e-3)
    np.testing.assert_allclose(x_pre_a[..., : 2 * valid], x_pre_b[..., : 2 * valid], atol=1e-3)
