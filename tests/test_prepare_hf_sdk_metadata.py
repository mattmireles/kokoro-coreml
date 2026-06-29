import importlib.util
import json
import sys
from pathlib import Path

import pytest


def load_helper():
    spec = importlib.util.spec_from_file_location(
        "prepare_hf_sdk_metadata",
        Path(__file__).resolve().parents[1] / "scripts" / "prepare_hf_sdk_metadata.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_profile_bundle(
    root: Path,
    profile: str,
    sdk_commit: str,
    repo_id: str,
    revision: str,
    hosted_files: list[str] | None = None,
) -> None:
    runtime = {
        "schema_version": 1,
        "sdk_commit": sdk_commit,
        "hf_repo_id": repo_id,
        "hf_revision": revision,
        "minimum_platforms": {"iOS": "18.0", "macOS": "15.0"},
        "bundle_profile": profile,
        "buckets": [15],
        "duration_token_sizes": [32],
        "model_packages": [],
        "voices": [],
    }
    hosted = {
        "version": f"{profile}-{sdk_commit[:12]}",
        "files": [{"path": path} for path in hosted_files or []],
    }
    (root / "runtime").mkdir(parents=True)
    (root / "KokoroRuntimeManifest.json").write_text(json.dumps(runtime), encoding="utf-8")
    (root / "HostedManifest.json").write_text(json.dumps(hosted), encoding="utf-8")
    (root / "runtime" / "kokoro-vocab.json").write_text("{}", encoding="utf-8")
    (root / "runtime" / "hnsf_weights.json").write_text("{}", encoding="utf-8")
    if hosted_files:
        for hosted_path in hosted_files:
            if hosted_path.startswith("coreml/") or hosted_path == "KokoroRuntimeManifest.json":
                continue
            if hosted_path.startswith("/") or "\\" in hosted_path or ".." in Path(hosted_path).parts:
                continue
            path = root / hosted_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"{hosted_path}\n", encoding="utf-8")


def test_refuses_to_overwrite_unmarked_payload_directory(tmp_path):
    helper = load_helper()
    output = tmp_path / "payload"
    output.mkdir()
    (output / "keep.txt").write_text("important", encoding="utf-8")

    with pytest.raises(SystemExit, match="unmarked payload directory"):
        helper.assert_safe_output_directory(output)


def test_allows_marked_payload_directory(tmp_path):
    helper = load_helper()
    output = tmp_path / "payload"
    output.mkdir()
    (output / helper.PAYLOAD_MARKER).write_text("kokoro-hf-sdk-metadata\n", encoding="utf-8")

    helper.assert_safe_output_directory(output)


def test_prepare_payload_rejects_mismatched_profile_revisions(tmp_path):
    helper = load_helper()
    starter = tmp_path / "starter"
    full = tmp_path / "full"
    write_profile_bundle(starter, "starter", "abc123", "mattmireles/kokoro-coreml", "rev-a")
    write_profile_bundle(full, "full", "abc123", "mattmireles/kokoro-coreml", "rev-b")
    model_card = tmp_path / "README.md"
    model_card.write_text("# card\n", encoding="utf-8")

    args = type("Args", (), {
        "repo_id": "mattmireles/kokoro-coreml",
        "output": tmp_path / "payload",
        "starter_bundle": starter,
        "full_bundle": full,
        "model_card": model_card,
        "sdk_commit": "abc123",
    })()

    with pytest.raises(SystemExit, match="profile HF revisions do not match"):
        helper.prepare_payload(args)


def test_prepare_payload_copies_top_level_hosted_runtime_and_voice_files(tmp_path):
    helper = load_helper()
    starter = tmp_path / "starter"
    full = tmp_path / "full"
    hosted_files = [
        "KokoroRuntimeManifest.json",
        "coreml/kokoro_duration_t32.mlpackage/Manifest.json",
        "runtime/kokoro-vocab.json",
        "runtime/hnsf_weights.json",
        "voices/af_heart.bin",
    ]
    write_profile_bundle(starter, "starter", "abc123", "mattmireles/kokoro-coreml", "rev-a", hosted_files)
    write_profile_bundle(full, "full", "abc123", "mattmireles/kokoro-coreml", "rev-a")
    model_card = tmp_path / "README.md"
    model_card.write_text("# card\n", encoding="utf-8")

    args = type("Args", (), {
        "repo_id": "mattmireles/kokoro-coreml",
        "output": tmp_path / "payload",
        "starter_bundle": starter,
        "full_bundle": full,
        "model_card": model_card,
        "sdk_commit": "abc123",
    })()

    helper.prepare_payload(args)

    assert (args.output / "runtime" / "kokoro-vocab.json").is_file()
    assert (args.output / "runtime" / "hnsf_weights.json").is_file()
    assert (args.output / "voices" / "af_heart.bin").is_file()
    assert not (args.output / "coreml" / "kokoro_duration_t32.mlpackage" / "Manifest.json").exists()

    release = json.loads((args.output / "sdk" / "SDKReleaseManifest.json").read_text(encoding="utf-8"))
    hosted_paths = {record["path"] for record in release["top_level_hosted_files"]}
    assert hosted_paths == {
        "runtime/kokoro-vocab.json",
        "runtime/hnsf_weights.json",
        "voices/af_heart.bin",
    }


@pytest.mark.parametrize("unsafe_path", ["../outside.txt", "/tmp/outside.txt", "voices\\af_heart.bin"])
def test_prepare_payload_rejects_unsafe_top_level_hosted_paths(tmp_path, unsafe_path):
    helper = load_helper()
    starter = tmp_path / "starter"
    full = tmp_path / "full"
    write_profile_bundle(
        starter,
        "starter",
        "abc123",
        "mattmireles/kokoro-coreml",
        "rev-a",
        ["KokoroRuntimeManifest.json", unsafe_path],
    )
    write_profile_bundle(full, "full", "abc123", "mattmireles/kokoro-coreml", "rev-a")
    model_card = tmp_path / "README.md"
    model_card.write_text("# card\n", encoding="utf-8")
    args = type("Args", (), {
        "repo_id": "mattmireles/kokoro-coreml",
        "output": tmp_path / "payload",
        "starter_bundle": starter,
        "full_bundle": full,
        "model_card": model_card,
        "sdk_commit": "abc123",
    })()

    with pytest.raises(SystemExit, match="unsafe hosted manifest path|escapes payload root"):
        helper.prepare_payload(args)
