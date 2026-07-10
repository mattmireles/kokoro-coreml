"""Tests for ``scripts/gemini_audio_judge_direct.py`` (no network)."""

from __future__ import annotations

import importlib.util
import json
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "gemini_audio_judge_direct",
    _ROOT / "scripts" / "gemini_audio_judge_direct.py",
)
assert _spec and _spec.loader
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)


def _write_wav(path: Path, pcm: np.ndarray, sample_rate: int = 24_000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.astype("<i2").tobytes())


def test_parse_clip_args_rejects_duplicate_labels() -> None:
    with pytest.raises(SystemExit, match="duplicate"):
        _mod.parse_clip_args(["a=/tmp/a.wav", "a=/tmp/b.wav"])


def test_validate_clip_path_rejects_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.wav"
    with pytest.raises(SystemExit, match="not found"):
        _mod.validate_clip_path("coreml", missing)


def test_validate_clip_path_rejects_non_audio_suffix(tmp_path: Path) -> None:
    secret = tmp_path / "secret.env"
    secret.write_text("GEMINI_API_KEY=abc\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="unsupported suffix"):
        _mod.validate_clip_path("leak", secret)


def test_resolve_output_path_must_stay_in_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_mod, "_REPO_ROOT", tmp_path)
    outside = tmp_path.parent / "outside.json"
    with pytest.raises(SystemExit, match="inside repo root"):
        _mod.resolve_output_path(outside)


def test_build_prompt_preserves_user_braces() -> None:
    prompt = _mod.build_prompt(
        prompt='Say {"hello"}',
        expected_style="speech",
        baseline="pytorch",
        context="",
    )
    assert '{"hello"}' in prompt
    assert "$prompt" not in prompt


def test_extract_gemini_text_handles_blocked_response() -> None:
    with pytest.raises(SystemExit, match="blocked"):
        _mod.extract_gemini_text({"promptFeedback": {"blockReason": "SAFETY"}})


def test_extract_gemini_text_reads_first_text_part() -> None:
    text = _mod.extract_gemini_text(
        {
            "candidates": [
                {"content": {"parts": [{"text": '{"overallVerdict":"pass"}'}]}}
            ]
        }
    )
    assert "overallVerdict" in text


def test_parse_verdict_json_requires_overall_verdict() -> None:
    with pytest.raises(SystemExit, match="overallVerdict"):
        _mod.parse_verdict_json('{"clips": {}}')


def test_gain_match_clip_scales_quiet_wav(tmp_path: Path) -> None:
    quiet = tmp_path / "quiet.wav"
    loud = tmp_path / "loud.wav"
    _write_wav(quiet, (np.ones(2400, dtype=np.int16) * 100))
    _write_wav(loud, (np.ones(2400, dtype=np.int16) * 5000))
    target = _mod.read_wav_rms(loud)
    matched = _mod.gain_match_clip(quiet, target, tmp_path)
    assert abs(_mod.read_wav_rms(matched) - target) < 50.0
