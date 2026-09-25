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


def test_per_bucket_sdk_profiles_download_only_duration_t128():
    module = load_download_models_module()

    for profile, voices, buckets in (
        ("starter", [], []),
        ("custom", ["af_heart", "af_bella"], [15]),
    ):
        packages = module._sdk_required_packages(profile, voices, buckets)
        duration_packages = [
            package for package in packages
            if "/kokoro_duration_" in package
        ]
        assert duration_packages == ["coreml/kokoro_duration_t128.mlpackage"]


def test_every_sdk_profile_downloads_the_one_flexible_generator():
    """SDK bundles ship one RangeDim generator, not one generator per bucket,
    and the download contract must match scripts/build_sdk_bundle.mjs."""
    module = load_download_models_module()
    builder = (Path(__file__).resolve().parents[1] / "scripts" / "build_sdk_bundle.mjs").read_text()
    assert f"'{module.SDK_FLEXIBLE_GENERATOR}'" in builder

    for profile, voices, buckets in (
        ("starter", [], []),
        ("custom", ["af_heart", "af_bella"], [15, 30]),
        ("full", [], []),
    ):
        packages = module._sdk_required_packages(profile, voices, buckets)
        generators = [package for package in packages if "/kokoro_decoder_har_post_" in package]
        assert generators == [f"coreml/{module.SDK_FLEXIBLE_GENERATOR}"], profile
        patterns = module._sdk_patterns(profile, voices, buckets)
        assert f"coreml/{module.SDK_FLEXIBLE_GENERATOR}/**" in patterns, profile


def test_full_profile_downloads_multifunction_packages_and_g2p():
    """The product profile matches build_sdk_bundle.mjs profiles.full."""
    module = load_download_models_module()
    builder = (Path(__file__).resolve().parents[1] / "scripts" / "build_sdk_bundle.mjs").read_text()

    packages = module._sdk_required_packages("full", [], [])
    assert packages == [
        "coreml/kokoro_duration_multifunction.mlpackage",
        "coreml/kokoro_f0ntrain_multifunction.mlpackage",
        "coreml/kokoro_decoder_pre_multifunction.mlpackage",
        f"coreml/{module.SDK_FLEXIBLE_GENERATOR}",
    ]
    for package in packages[:3]:
        assert f"'{package.removeprefix('coreml/')}'" in builder
    patterns = module._sdk_patterns("full", [], [])
    assert "g2p/**" in patterns
    assert not any("_t128.mlpackage" in pattern for pattern in patterns)
