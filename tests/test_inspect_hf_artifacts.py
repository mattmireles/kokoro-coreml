"""inspect_hf_artifacts.py: the post-publish inventory gate."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_inspector():
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location("inspect_hf_artifacts", SCRIPTS / "inspect_hf_artifacts.py")
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(SCRIPTS))


def test_missing_inventory_manifest_fails_the_gate():
    """Regression: a manifest absent at the pin had no entries to flag, so the gate exited 0."""

    inspector = load_inspector()
    report = {"manifest_checks": {
        "starter": {"present": False, "unresolved_files": []},
        "full": {"present": True, "unresolved_files": [], "digest_mismatches": []},
    }}
    assert inspector.inventory_problems(report) == ["starter: manifest missing"]


def test_quoted_hf_token_in_env_file_is_unquoted(tmp_path, monkeypatch):
    """Regression: HF_TOKEN="hf_x" in .env kept its quotes."""

    sys.path.insert(0, str(SCRIPTS))
    try:
        import hf_common
    finally:
        sys.path.remove(str(SCRIPTS))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
    monkeypatch.setattr(hf_common, "REPO_ROOT", tmp_path)
    (tmp_path / ".env").write_text('HF_TOKEN="hf_quoted"\n', encoding="utf-8")
    assert hf_common.load_hf_token() == "hf_quoted"
