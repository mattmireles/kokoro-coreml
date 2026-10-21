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


def test_every_sdk_profile_downloads_flexible_and_bucket_generators():
    """SDK bundles ship the RangeDim generator plus one generator per bucket:
    macOS 27 can abort on the RangeDim program (iOS 27 is gated as a
    precaution), so the runtime falls back to the bucket generators there. Must match scripts/build_sdk_bundle.mjs."""
    module = load_download_models_module()
    builder = (Path(__file__).resolve().parents[1] / "scripts" / "build_sdk_bundle.mjs").read_text()
    assert f"'{module.SDK_FLEXIBLE_GENERATOR}'" in builder
    assert "names.push(`kokoro_decoder_har_post_${bucket}s.mlpackage`)" in builder

    for profile, voices, buckets, expected_buckets in (
        ("starter", [], [], [15]),
        ("custom", ["af_heart", "af_bella"], [15, 30], [15, 30]),
        ("full", [], [], [3, 7, 10, 15, 30]),
    ):
        packages = module._sdk_required_packages(profile, voices, buckets)
        generators = sorted(package for package in packages if "/kokoro_decoder_har_post_" in package)
        assert generators == sorted(
            [f"coreml/{module.SDK_FLEXIBLE_GENERATOR}"]
            + [f"coreml/kokoro_decoder_har_post_{bucket}s.mlpackage" for bucket in expected_buckets]
        ), profile
        patterns = module._sdk_patterns(profile, voices, buckets)
        for generator in generators:
            assert f"{generator}/**" in patterns, profile


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
        *(f"coreml/kokoro_decoder_har_post_{bucket}s.mlpackage" for bucket in (3, 7, 10, 15, 30)),
    ]
    for package in packages[:3] + packages[4:]:
        assert f"'{package.removeprefix('coreml/')}'" in builder
    patterns = module._sdk_patterns("full", [], [])
    assert "g2p/**" in patterns
    assert not any("_t128.mlpackage" in pattern for pattern in patterns)
