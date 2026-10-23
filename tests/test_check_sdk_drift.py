"""Run scripts/check_sdk_drift.mjs so the SDK drift gate cannot go red unnoticed."""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_sdk_drift_check_passes():
    """Regression: the gate required doc snippets that had been rewritten, and failed silently outside CI."""

    result = subprocess.run(["node", "scripts/check_sdk_drift.mjs"], cwd=REPO, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "SDK drift check passed" in result.stdout
