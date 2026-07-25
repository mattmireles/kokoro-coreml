"""Regression coverage for the fixed-shape SDK download contract."""

import importlib.util
from pathlib import Path


def load_download_models_module():
    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "download_models",
        repo / "scripts" / "download_models.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_sdk_profile_downloads_only_duration_t128():
    module = load_download_models_module()

    for profile, voices, buckets in (
        ("starter", [], []),
        ("custom", ["af_heart", "af_bella"], [15]),
        ("full", [], []),
    ):
        packages = module._sdk_required_packages(profile, voices, buckets)
        duration_packages = [
            package for package in packages
            if "/kokoro_duration_" in package
        ]
        assert duration_packages == ["coreml/kokoro_duration_t128.mlpackage"]
