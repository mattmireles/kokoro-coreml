"""Regression coverage for the SDK profile contract in scripts/sdk_profiles.json."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PROFILES = json.loads((REPO / "scripts" / "sdk_profiles.json").read_text(encoding="utf-8"))


def load_download_models_module():
    sys.path.insert(0, str(REPO / "scripts"))
    try:
        spec = importlib.util.spec_from_file_location("download_models", REPO / "scripts" / "download_models.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(REPO / "scripts"))


def node_profile(profile: str, voices: list[str], buckets: list[int]) -> dict:
    """Resolve a profile with build_sdk_bundle.mjs's own resolveProfile()."""

    args = json.dumps(["--profile", profile, "--voices", ",".join(voices) or "-", "--buckets", ",".join(map(str, buckets)) or "-"])
    script = (
        "import('./scripts/build_sdk_bundle.mjs').then((m) => {"
        f"const a = {args}; const map = new Map();"
        "for (let i = 0; i < a.length; i += 2) if (a[i + 1] !== '-') map.set(a[i].slice(2), a[i + 1]);"
        "console.log(JSON.stringify(m.resolveProfile(map)));"
        "})"
    )
    result = subprocess.run(["node", "--input-type=module", "-e", script], cwd=REPO, text=True, capture_output=True, check=True)
    return json.loads(result.stdout)


def test_starter_is_custom_template_for_af_heart_15s():
    module = load_download_models_module()
    starter = module.sdk_profile("starter", [], [])
    assert module.sdk_profile("custom", ["af_heart"], [15])["paths"] == starter["paths"]
    assert starter["voices"] == ["af_heart"] and starter["buckets"] == [15]


def test_release_profiles_ship_only_the_measured_pipeline():
    """Plan 014 never loads multifunction, whole duration/F0N, or HAR-post packages."""

    for name, profile in PROFILES["profiles"].items():
        for path in profile["paths"]:
            assert path.startswith(("coreml/", "g2p/")), (name, path)
            assert "multifunction" not in path, (name, path)
            assert "har_post" not in path, (name, path)
            assert not path.startswith(("coreml/kokoro_duration_t", "coreml/kokoro_f0ntrain_t")), (name, path)
        assert "coreml/kokoro_duration_ane_front_t128.mlpackage" in profile["paths"]
        assert len(set(profile["paths"])) == len(profile["paths"])
    assert PROFILES["duration_token_sizes"] == [128]


def test_full_profile_has_both_folds_every_hybrid_row_and_g2p():
    full = PROFILES["profiles"]["full"]
    paths = set(full["paths"])
    for bucket in full["buckets"]:
        assert f"coreml/kokoro_generator_fold_{bucket}s.mlpackage" in paths
        assert f"coreml/kokoro_generator_fold_poly_{bucket}s.mlpackage" in paths
        assert f"coreml/kokoro_decoder_pre_{bucket}s.mlpackage" in paths
    for rows in (2, 3, 4, 6, 10, 11):
        assert f"coreml/hybrid/hybrid_generator_r{rows}.mlpackage" in paths
    assert {"coreml/hybrid/plan.json", "coreml/hybrid/adain.bin", "g2p/G2PEncoder.mlmodelc"} <= paths
    assert len(full["voices"]) == 28


def test_custom_uses_poly_fold_and_bucket_max_hybrid_rows():
    module = load_download_models_module()
    paths = module.sdk_profile("custom", ["af_bella"], [3, 30])["paths"]
    assert "coreml/kokoro_generator_fold_poly_3s.mlpackage" in paths
    assert not any("generator_fold_3s" in path or "generator_fold_30s" in path for path in paths)
    assert [p for p in paths if "hybrid_generator" in p] == ["coreml/hybrid/hybrid_generator_r11.mlpackage"]
    assert module.sdk_profile("custom", ["af_bella"], [3])["paths"] == PROFILES["custom"]["paths"] + [
        "coreml/kokoro_f0ntrain_front_t120.mlpackage",
        "coreml/kokoro_f0ntrain_ane_back_t120.mlpackage",
        "coreml/kokoro_decoder_pre_3s.mlpackage",
        "coreml/kokoro_generator_fold_poly_3s.mlpackage",
    ]
    with pytest.raises(SystemExit, match="unsupported SDK bucket"):
        module.sdk_profile("custom", ["af_bella"], [5])


@pytest.mark.parametrize("profile,voices,buckets", [
    ("starter", [], []),
    ("full", [], []),
    ("custom", ["af_heart", "am_adam"], [7, 15]),
])
def test_python_and_node_resolve_profiles_identically(profile, voices, buckets):
    module = load_download_models_module()
    expected = module.sdk_profile(profile, voices, buckets)
    observed = node_profile(profile, voices, buckets)
    assert observed["paths"] == expected["paths"]
    assert observed["voices"] == expected["voices"]
    assert observed["buckets"] == expected["buckets"]


def test_sdk_patterns_cover_files_and_directories():
    module = load_download_models_module()
    patterns = module.sdk_patterns(module.sdk_profile("starter", [], []))
    assert module._matches_any("coreml/hybrid/plan.json", patterns)
    assert module._matches_any("coreml/hybrid/hybrid_generator_r6.mlpackage/Manifest.json", patterns)
    assert not module._matches_any("coreml/hybrid/hybrid_generator_r2.mlpackage/Manifest.json", patterns)
    assert not module._matches_any("coreml/kokoro_generator_fold_15s.mlpackage/Manifest.json", patterns)
    assert "kokoro.js/voices/af_heart.bin" in patterns


def test_non_lfs_download_digest_must_match_hf_blob_id(tmp_path):
    """Regression: a non-LFS file's digest was hashed from the local file alone."""

    module = load_download_models_module()
    local = tmp_path / "plan.json"
    local.write_bytes(b'{"rows": [6]}')
    item = types.SimpleNamespace(path="coreml/hybrid/plan.json", size=13, lfs=None, blob_id=module.git_blob_sha1(local))
    assert module._verified_file_record(item, local)["sha256"] == module.sha256_file(local)
    local.write_bytes(b'{"rows": [7]}')
    with pytest.raises(SystemExit, match="does not match HF blob id"):
        module._verified_file_record(item, local)


def test_force_manifest_repair_scans_only_coreml(tmp_path, monkeypatch):
    """Regression: --force repair walked every *.mlpackage under the repo root."""

    module = load_download_models_module()
    for rel in ("coreml/a.mlpackage", "coreml/hybrid/b.mlpackage", "coreml-fold/c.mlpackage", "outputs/d.mlpackage"):
        (tmp_path / rel).mkdir(parents=True)
    requested = []
    fake_hub = types.SimpleNamespace(hf_hub_download=lambda repo, file, **kw: requested.append(file))
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    monkeypatch.setattr(module, "_REPO_ROOT", tmp_path)
    assert module._repair_missing_manifests(None, force=True) == 2
    assert requested == ["coreml/a.mlpackage/Manifest.json", "coreml/hybrid/b.mlpackage/Manifest.json"]
