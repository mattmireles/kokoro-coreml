from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def digest(root: Path, relative: str) -> dict:
    data = (root / relative).read_bytes()
    return {"path": relative, "bytes": len(data), "sha256": sha256_hex(data)}


def tree_digest(root: Path, relative: str) -> dict:
    """Mirror build_sdk_bundle.mjs hashTree()."""

    top = root / relative
    h = hashlib.sha256()
    files = sorted(p for p in top.rglob("*") if p.is_file())
    for path in files:
        data = path.read_bytes()
        h.update(f"{path.relative_to(top).as_posix()}\0{len(data)}\0{sha256_hex(data)}\0".encode())
    return {
        "path": relative,
        "tree_sha256": h.hexdigest(),
        "file_count": len(files),
        "bytes": sum(p.stat().st_size for p in files),
    }


def write_bundle(bundle: Path, schema_version: int = 2, model_dirs: tuple[str, ...] = ()) -> dict:
    runtime = bundle / "runtime"
    runtime.mkdir(parents=True)
    (runtime / "kokoro-vocab.json").write_bytes(b"{}")
    (runtime / "hnsf_weights.json").write_bytes(b"{}")
    manifest = {
        "schema_version": schema_version,
        "model_packages": [tree_digest(bundle, rel) for rel in model_dirs],
        "voices": [],
        "runtime_assets": {
            "vocab": digest(bundle, "runtime/kokoro-vocab.json"),
            "hnsf_weights": digest(bundle, "runtime/hnsf_weights.json"),
        },
    }
    (bundle / "KokoroRuntimeManifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def write_complete_hosted(bundle: Path) -> None:
    files = sorted(
        p.relative_to(bundle).as_posix() for p in bundle.rglob("*")
        if p.is_file() and p.name != "HostedManifest.json"
    )
    hosted = {"version": "test", "files": [digest(bundle, rel) for rel in files]}
    (bundle / "HostedManifest.json").write_text(json.dumps(hosted), encoding="utf-8")


def validate(bundle: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["node", "scripts/validate_sdk_bundle.mjs", str(bundle)],
        cwd=REPO, text=True, capture_output=True, check=False,
    )


def test_validate_sdk_bundle_rejects_hosted_manifest_path_escape(tmp_path):
    bundle = tmp_path / "bundle"
    write_bundle(bundle)
    hosted = {"version": "bad", "files": [{"path": "../evil", "bytes": 0, "sha256": sha256_hex(b"")}]}
    (bundle / "HostedManifest.json").write_text(json.dumps(hosted), encoding="utf-8")

    result = validate(bundle)

    assert result.returncode != 0
    assert "manifest path escapes bundle root" in result.stderr


def test_validate_sdk_bundle_rejects_incomplete_hosted_file_list(tmp_path):
    bundle = tmp_path / "bundle"
    write_bundle(bundle)
    hosted = {"version": "incomplete", "files": [
        digest(bundle, "KokoroRuntimeManifest.json"),
        digest(bundle, "runtime/kokoro-vocab.json"),
    ]}
    (bundle / "HostedManifest.json").write_text(json.dumps(hosted), encoding="utf-8")

    result = validate(bundle)

    assert result.returncode != 0
    assert "does not exactly cover" in result.stderr


def make_hybrid(bundle: Path) -> None:
    hybrid = bundle / "coreml" / "hybrid"
    (hybrid / "hybrid_generator_r6.mlpackage" / "Data").mkdir(parents=True)
    (hybrid / "plan.json").write_text('{"rows": [6]}', encoding="utf-8")
    (hybrid / "adain.bin").write_bytes(b"\x00\x01")
    (hybrid / "hybrid_generator_r6.mlpackage" / "Manifest.json").write_text("{}", encoding="utf-8")
    (hybrid / "hybrid_generator_r6.mlpackage" / "Data" / "w.bin").write_bytes(b"w")


def test_validate_sdk_bundle_accepts_hybrid_directory_entry(tmp_path):
    bundle = tmp_path / "bundle"
    make_hybrid(bundle)
    write_bundle(bundle, model_dirs=("coreml/hybrid",))
    write_complete_hosted(bundle)

    result = validate(bundle)

    assert result.returncode == 0, result.stderr


def test_validate_sdk_bundle_detects_hybrid_plan_drift(tmp_path):
    bundle = tmp_path / "bundle"
    make_hybrid(bundle)
    write_bundle(bundle, model_dirs=("coreml/hybrid",))
    write_complete_hosted(bundle)
    (bundle / "coreml" / "hybrid" / "plan.json").write_text('{"rows": [7]}', encoding="utf-8")

    result = validate(bundle)

    assert result.returncode != 0
    assert "package digest drift for coreml/hybrid" in result.stderr


def test_validate_sdk_bundle_rejects_schema_version_1(tmp_path):
    bundle = tmp_path / "bundle"
    write_bundle(bundle, schema_version=1)
    write_complete_hosted(bundle)

    result = validate(bundle)

    assert result.returncode != 0
    assert "unsupported KokoroRuntimeManifest schema_version: 1" in result.stderr
