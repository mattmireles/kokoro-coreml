#!/usr/bin/env python3
"""Verify that KokoroTTS small runtime assets are canonical and non-local."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SDK_RUNTIME_DIR = Path("swift-tts/Sources/KokoroTTS/Resources/KokoroRuntime")
SDK_MANIFEST = SDK_RUNTIME_DIR / "KokoroRuntimeAssets.json"
SDK_VOCAB = SDK_RUNTIME_DIR / "kokoro-vocab.json"
SDK_HNSF_WEIGHTS = SDK_RUNTIME_DIR / "hnsf_weights.json"
ROOT_VOCAB = Path("_kokoro_vocab.json")
ROOT_HNSF_WEIGHTS = Path("hnsf_weights.json")
IOS_BENCH_CONFIG = Path("ios-bench/Vendor/kokoro-ios/Resources/config.json")
IOS_BENCH_HNSF_WEIGHTS = Path("ios-bench/Resources/bench_inputs/hnsf_weights.json")


@dataclass(frozen=True)
class VerifiedFile:
    """A repository-relative file and its SHA-256 digest."""

    path: Path
    sha256: str


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the runtime asset verifier."""

    parser = argparse.ArgumentParser(
        description="Verify KokoroTTS checked runtime assets and their hashes."
    )
    parser.add_argument(
        "--allow-missing-optional-provenance",
        action="store_true",
        help="Reserved for future external provenance checks; internal repo files remain required.",
    )
    return parser.parse_args()


def repo_path(relative_path: Path) -> Path:
    """Return the absolute path for a repository-relative path."""

    return REPO_ROOT / relative_path


def sha256_file(path: Path) -> str:
    """Compute a SHA-256 digest for a regular file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    """Load a JSON file with a path-specific failure context."""

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"{path} is not valid JSON: {error}") from error


def assert_no_symlink_path(path: Path) -> None:
    """Reject symlinks anywhere between the repository root and a target path."""

    relative = path.relative_to(REPO_ROOT)
    current = REPO_ROOT
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise RuntimeError(f"{current.relative_to(REPO_ROOT)} must not be a symlink")


def assert_regular_repo_file(relative_path: Path) -> VerifiedFile:
    """Verify that a repository-relative path exists as a real file."""

    path = repo_path(relative_path)
    if not path.exists():
        raise RuntimeError(f"{relative_path} is missing")
    assert_no_symlink_path(path)
    if not path.is_file():
        raise RuntimeError(f"{relative_path} is not a regular file")
    return VerifiedFile(relative_path, sha256_file(path))


def canonical_json_sha256(value: Any) -> str:
    """Hash a JSON value independent of object-key order and whitespace."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def vocab_payload(path: Path) -> dict[str, int]:
    """Extract the Kokoro vocab dictionary from a vocab file or full config."""

    payload = load_json(path)
    vocab = payload.get("vocab") if isinstance(payload, dict) else None
    if not isinstance(vocab, dict):
        raise RuntimeError(f"{path.relative_to(REPO_ROOT)} does not contain a vocab object")
    if not all(isinstance(key, str) and isinstance(value, int) for key, value in vocab.items()):
        raise RuntimeError(f"{path.relative_to(REPO_ROOT)} has a malformed vocab object")
    return vocab


def verify_manifest_file(manifest: dict[str, Any], file: VerifiedFile, section: str) -> None:
    """Compare a file digest with the matching manifest section."""

    manifest_section = manifest.get(section)
    if not isinstance(manifest_section, dict):
        raise RuntimeError(f"manifest section {section!r} is missing")
    expected_path = manifest_section.get("path")
    if expected_path != file.path.name:
        raise RuntimeError(f"manifest {section}.path is {expected_path!r}, expected {file.path.name!r}")
    expected_sha = manifest_section.get("sha256")
    if expected_sha != file.sha256:
        raise RuntimeError(
            f"{file.path} SHA-256 drift: manifest has {expected_sha}, observed {file.sha256}"
        )


def verify_vocab_assets(manifest: dict[str, Any]) -> None:
    """Verify SDK vocab hashes against all checked repo vocab copies."""

    sdk_vocab = assert_regular_repo_file(SDK_VOCAB)
    verify_manifest_file(manifest, sdk_vocab, "vocab")

    sdk_vocab_payload = vocab_payload(repo_path(SDK_VOCAB))
    sdk_canonical_hash = canonical_json_sha256({"vocab": sdk_vocab_payload})
    manifest_vocab = manifest["vocab"]
    if manifest_vocab.get("canonical_json_sha256") != sdk_canonical_hash:
        raise RuntimeError("SDK vocab canonical JSON hash does not match the manifest")
    if manifest_vocab.get("entries") != len(sdk_vocab_payload):
        raise RuntimeError("SDK vocab entry count does not match the manifest")

    for relative_path in (ROOT_VOCAB, IOS_BENCH_CONFIG):
        assert_regular_repo_file(relative_path)
        observed_payload = vocab_payload(repo_path(relative_path))
        observed_hash = canonical_json_sha256({"vocab": observed_payload})
        if observed_hash != sdk_canonical_hash:
            raise RuntimeError(f"{relative_path} vocab canonical hash drifted from SDK vocab")


def verify_hnsf_weights_payload(relative_path: Path, expected_weights_sha: str) -> VerifiedFile:
    """Verify one hn-NSF weights JSON file and return its file digest."""

    verified = assert_regular_repo_file(relative_path)
    payload = load_json(repo_path(relative_path))
    weights_sha = payload.get("weights_sha256") if isinstance(payload, dict) else None
    if not isinstance(weights_sha, str) or not weights_sha or weights_sha == "unverified":
        raise RuntimeError(f"{relative_path} has invalid weights_sha256: {weights_sha!r}")
    if weights_sha != expected_weights_sha:
        raise RuntimeError(
            f"{relative_path} weights_sha256 drift: expected {expected_weights_sha}, observed {weights_sha}"
        )
    weights = payload.get("linear_weights")
    if not isinstance(weights, list) or len(weights) != 9:
        raise RuntimeError(f"{relative_path} must contain exactly 9 hn-NSF linear weights")
    if not isinstance(payload.get("linear_bias"), (float, int)):
        raise RuntimeError(f"{relative_path} must contain a numeric hn-NSF linear bias")
    return verified


def verify_hnsf_assets(manifest: dict[str, Any]) -> None:
    """Verify SDK hn-NSF weights against root and benchmark copies."""

    manifest_hnsf = manifest.get("hnsf_weights")
    if not isinstance(manifest_hnsf, dict):
        raise RuntimeError("manifest section 'hnsf_weights' is missing")
    expected_weights_sha = manifest_hnsf.get("weights_sha256")
    if not isinstance(expected_weights_sha, str) or expected_weights_sha == "unverified":
        raise RuntimeError("manifest hnsf_weights.weights_sha256 must be verified")

    sdk_hnsf = verify_hnsf_weights_payload(SDK_HNSF_WEIGHTS, expected_weights_sha)
    verify_manifest_file(manifest, sdk_hnsf, "hnsf_weights")

    for relative_path in (ROOT_HNSF_WEIGHTS, IOS_BENCH_HNSF_WEIGHTS):
        observed = verify_hnsf_weights_payload(relative_path, expected_weights_sha)
        if observed.sha256 != sdk_hnsf.sha256:
            raise RuntimeError(f"{relative_path} SHA-256 drifted from SDK hn-NSF weights")


def verify_runtime_assets() -> None:
    """Run every runtime asset provenance and hash check."""

    manifest_file = assert_regular_repo_file(SDK_MANIFEST)
    manifest = load_json(repo_path(manifest_file.path))
    if manifest.get("schema_version") != 1:
        raise RuntimeError("runtime asset manifest schema_version must be 1")
    verify_vocab_assets(manifest)
    verify_hnsf_assets(manifest)


def main() -> None:
    """CLI entry point for the runtime asset verifier."""

    parse_args()
    verify_runtime_assets()
    print("runtime assets verified")
    print(f"  vocab_sha256={sha256_file(repo_path(SDK_VOCAB))}")
    print(f"  hnsf_weights_sha256={sha256_file(repo_path(SDK_HNSF_WEIGHTS))}")


if __name__ == "__main__":
    main()
