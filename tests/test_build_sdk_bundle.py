"""End-to-end build_sdk_bundle.mjs run in a throwaway git checkout with fake packages."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REVISION = "e" * 40


def make_checkout(root: Path) -> None:
    (root / "scripts").mkdir(parents=True)
    for name in ("build_sdk_bundle.mjs", "validate_sdk_bundle.mjs", "sdk_profiles.json"):
        shutil.copy2(REPO / "scripts" / name, root / "scripts" / name)
    runtime = root / "swift-tts/Sources/KokoroTTS/Resources/KokoroRuntime"
    runtime.mkdir(parents=True)
    for name in ("kokoro-vocab.json", "hnsf_weights.json"):
        (runtime / name).write_text("{}", encoding="utf-8")
    profile = json.loads((REPO / "scripts" / "sdk_profiles.json").read_text(encoding="utf-8"))["profiles"]["starter"]
    for rel in profile["paths"]:
        target = root / rel
        if rel.endswith(".mlpackage"):
            (target / "Data").mkdir(parents=True)
            (target / "Manifest.json").write_text("{}", encoding="utf-8")
            (target / "Data" / "weight.bin").write_bytes(rel.encode())
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(rel.encode())
    (root / "kokoro.js/voices").mkdir(parents=True)
    (root / "kokoro.js/voices/af_heart.bin").write_bytes(b"voice")
    (root / ".gitignore").write_text("coreml/\nkokoro.js/\nout/\n", encoding="utf-8")
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run([*git, "commit", "-qm", "fixture"], cwd=root, check=True)


def build(root: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["node", "scripts/build_sdk_bundle.mjs", "--profile", "starter", "--output", str(root / "out"),
         "--allow-local-provenance", "1", *extra],
        cwd=root, text=True, capture_output=True, check=False,
    )


def test_revision_is_required(tmp_path):
    """Regression: a stale default revision silently recorded the wrong provenance."""

    make_checkout(tmp_path)
    result = build(tmp_path)
    assert result.returncode != 0
    assert "--revision <hf-commit-sha> is required" in result.stderr


def test_dirty_build_keeps_full_identity_and_hybrid_directory_entry(tmp_path):
    """Regression: the hosted version dropped `-dirty` by truncating sdkCommit."""

    make_checkout(tmp_path)
    (tmp_path / "scripts" / "untracked.txt").write_text("dirty", encoding="utf-8")
    result = build(tmp_path, "--revision", REVISION, "--allow-dirty", "1")
    assert result.returncode == 0, result.stderr

    bundle = tmp_path / "out"
    runtime = json.loads((bundle / "KokoroRuntimeManifest.json").read_text(encoding="utf-8"))
    hosted = json.loads((bundle / "HostedManifest.json").read_text(encoding="utf-8"))
    assert runtime["schema_version"] == 2
    assert runtime["sdk_commit"].endswith("-dirty")
    assert hosted["version"] == f"starter-{runtime['sdk_commit']}"
    entries = [pkg["path"] for pkg in runtime["model_packages"]]
    assert entries[-1] == "coreml/hybrid" and len(entries) == 7
    hybrid = runtime["model_packages"][-1]
    assert {f["path"] for f in hybrid["files"]} == {
        "plan.json", "adain.bin",
        "hybrid_generator_r6.mlpackage/Manifest.json", "hybrid_generator_r6.mlpackage/Data/weight.bin",
    }
    validated = subprocess.run(["node", "scripts/validate_sdk_bundle.mjs", str(bundle)], cwd=tmp_path,
                               text=True, capture_output=True, check=False)
    assert validated.returncode == 0, validated.stderr


def test_download_manifest_file_missing_locally_fails(tmp_path):
    """Regression: only local files were checked against the manifest, so a half-downloaded package passed."""

    root = tmp_path / "repo"
    make_checkout(root)
    profile = json.loads((REPO / "scripts" / "sdk_profiles.json").read_text(encoding="utf-8"))["profiles"]["starter"]
    files = []
    for rel in [*profile["paths"], "kokoro.js/voices/af_heart.bin"]:
        top = root / rel
        for path in [top] if top.is_file() else sorted(p for p in top.rglob("*") if p.is_file()):
            data = path.read_bytes()
            files.append({"path": path.relative_to(root).as_posix(), "bytes": len(data),
                          "sha256": hashlib.sha256(data).hexdigest()})
    manifest = {"repo_id": "mattmireles/kokoro-coreml", "revision": REVISION, "files": files}
    manifest_path = tmp_path / "download.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    args = ("--revision", REVISION, "--download-manifest", str(manifest_path))
    result = build(root, *args)
    assert result.returncode == 0, result.stderr

    missing = "coreml/hybrid/hybrid_generator_r6.mlpackage/Data/weight.bin"
    (root / missing).unlink()
    result = build(root, *args)
    assert result.returncode != 0
    assert f"download manifest file missing locally: {missing}" in result.stderr
