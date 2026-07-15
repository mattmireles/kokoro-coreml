#!/usr/bin/env python3
"""Prepare the Hugging Face SDK metadata payload for KokoroTTS releases.

This script intentionally uploads only lightweight metadata and docs:

- README.md, sourced from README/hf-model-card.md.
- Top-level HostedManifest.json and KokoroRuntimeManifest.json for the starter
  profile, preserving the current public discovery contract.
- sdk/<profile>/KokoroRuntimeManifest.json for each checked bundle profile.
- sdk/SDKReleaseManifest.json, which records checksums and profile summaries.

Model packages remain the canonical large artifacts already hosted in the
Hugging Face model repo. Starter runtime and voice files are copied to the HF
root so the top-level HostedManifest.json is directly hydratable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPO_ID = "mattmireles/kokoro-coreml"
PAYLOAD_MARKER = ".kokoro-hf-sdk-metadata"


@dataclass(frozen=True)
class ProfileInput:
    """Input paths and metadata for one SDK bundle profile."""

    name: str
    bundle: Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for payload preparation and upload."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face repo ID")
    parser.add_argument("--output", required=True, type=Path, help="Directory for the prepared payload")
    parser.add_argument("--starter-bundle", required=True, type=Path, help="Validated starter SDK bundle")
    parser.add_argument("--full-bundle", required=True, type=Path, help="Validated full SDK bundle")
    parser.add_argument(
        "--model-card",
        default=REPO_ROOT / "README" / "hf-model-card.md",
        type=Path,
        help="Model card markdown to upload as README.md",
    )
    parser.add_argument(
        "--sdk-commit",
        help="Expected SDK commit. Defaults to git rev-parse HEAD in this checkout.",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload the prepared payload to the Hugging Face repo.",
    )
    return parser.parse_args()


def git_head() -> str:
    """Return the current Git HEAD commit for manifest compatibility checks."""

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def sha256_file(path: Path) -> str:
    """Compute a SHA-256 digest for one regular file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_safe_output_directory(output: Path) -> None:
    """Refuse destructive writes outside a marked metadata payload directory."""

    resolved = output.resolve()
    home = Path.home().resolve()
    dangerous = {
        Path(resolved.anchor).resolve(),
        home,
        REPO_ROOT,
        REPO_ROOT.parent,
    }
    if resolved in dangerous:
        raise SystemExit(f"refusing dangerous HF payload output path: {resolved}")
    if REPO_ROOT in resolved.parents:
        raise SystemExit("refusing to write HF payload inside the repo checkout")
    if resolved.exists() and not (resolved / PAYLOAD_MARKER).exists():
        raise SystemExit(f"refusing to overwrite unmarked payload directory: {resolved}")


def file_record(root: Path, path: Path) -> dict[str, Any]:
    """Return a checksum record for one file relative to a payload root."""

    stat = path.stat()
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": stat.st_size,
        "sha256": sha256_file(path),
    }


def contained_payload_path(root: Path, relative_path: str) -> Path:
    """Resolve a payload path while rejecting absolute paths and lexical escapes."""

    if not relative_path or relative_path.startswith("/") or "\\" in relative_path:
        raise SystemExit(f"unsafe hosted manifest path: {relative_path}")
    parts = Path(relative_path).parts
    if any(part in {"", ".", ".."} for part in parts):
        raise SystemExit(f"unsafe hosted manifest path: {relative_path}")
    resolved_root = root.resolve()
    candidate = (resolved_root / relative_path).resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise SystemExit(f"hosted manifest path escapes payload root: {relative_path}")
    return candidate


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object from disk."""

    return json.loads(path.read_text(encoding="utf-8"))


def load_hf_token() -> str | None:
    """Load an HF token using the same env/.env/cache order as the downloader."""

    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("HF_TOKEN="):
                return stripped.split("=", 1)[1].strip()
    try:
        from huggingface_hub import HfFolder

        return HfFolder.get_token()
    except Exception:
        return None


def copy_manifest_pair(profile: ProfileInput, output: Path, sdk_commit: str, repo_id: str) -> dict[str, Any]:
    """Copy profile manifests into the payload and return release metadata."""

    runtime_src = profile.bundle / "KokoroRuntimeManifest.json"
    hosted_src = profile.bundle / "HostedManifest.json"
    if not runtime_src.exists() or not hosted_src.exists():
        raise SystemExit(f"{profile.name} bundle is missing required manifests: {profile.bundle}")

    runtime = load_json(runtime_src)
    hosted = load_json(hosted_src)
    observed_commit = runtime.get("sdk_commit")
    if observed_commit != sdk_commit:
        raise SystemExit(
            f"{profile.name} manifest sdk_commit mismatch: expected {sdk_commit}, observed {observed_commit}"
        )
    if runtime.get("hf_repo_id") != repo_id:
        raise SystemExit(
            f"{profile.name} manifest hf_repo_id mismatch: expected {repo_id}, "
            f"observed {runtime.get('hf_repo_id')}"
        )

    profile_dir = output / "sdk" / profile.name
    profile_dir.mkdir(parents=True, exist_ok=True)
    runtime_dest = profile_dir / "KokoroRuntimeManifest.json"
    shutil.copy2(runtime_src, runtime_dest)

    return {
        "profile": profile.name,
        "sdk_commit": observed_commit,
        "hf_repo_id": runtime.get("hf_repo_id"),
        "hf_revision": runtime.get("hf_revision"),
        "hosted_version": hosted.get("version"),
        "hosted_manifest_scope": "top-level starter manifest is directly hydratable; profile manifests are metadata",
        "minimum_platforms": runtime.get("minimum_platforms"),
        "buckets": runtime.get("buckets"),
        "duration_token_sizes": runtime.get("duration_token_sizes"),
        "model_package_count": len(runtime.get("model_packages") or []),
        "voice_count": len(runtime.get("voices") or []),
        "runtime_manifest": file_record(output, runtime_dest),
    }


def copy_top_level_hosted_files(starter_bundle: Path, output: Path) -> list[dict[str, Any]]:
    """Copy starter hosted files that are not already canonical HF model paths."""

    hosted = load_json(starter_bundle / "HostedManifest.json")
    records = []
    for entry in hosted.get("files") or []:
        relative_path = entry.get("path")
        if not isinstance(relative_path, str):
            continue
        if relative_path == "KokoroRuntimeManifest.json" or relative_path.startswith("coreml/"):
            continue
        source = contained_payload_path(starter_bundle, relative_path)
        if not source.is_file():
            raise SystemExit(f"starter HostedManifest references missing local file: {relative_path}")
        destination = contained_payload_path(output, relative_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        records.append(file_record(output, destination))
    return sorted(records, key=lambda record: record["path"])


def prepare_payload(args: argparse.Namespace) -> Path:
    """Create a deterministic Hugging Face metadata payload directory."""

    sdk_commit = args.sdk_commit or git_head()
    output = args.output.resolve()
    assert_safe_output_directory(output)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    (output / PAYLOAD_MARKER).write_text("kokoro-hf-sdk-metadata\n", encoding="utf-8")

    readme_dest = output / "README.md"
    shutil.copy2(args.model_card, readme_dest)

    profiles = [
        ProfileInput("starter", args.starter_bundle.resolve()),
        ProfileInput("full", args.full_bundle.resolve()),
    ]
    profile_records = [copy_manifest_pair(profile, output, sdk_commit, args.repo_id) for profile in profiles]
    revisions = {profile["hf_revision"] for profile in profile_records}
    repo_ids = {profile["hf_repo_id"] for profile in profile_records}
    if repo_ids != {args.repo_id}:
        raise SystemExit(f"profile HF repo IDs do not all match {args.repo_id}: {sorted(repo_ids)}")
    if len(revisions) != 1:
        raise SystemExit(f"profile HF revisions do not match: {sorted(revisions)}")

    starter_profile_dir = output / "sdk" / "starter"
    shutil.copy2(args.starter_bundle.resolve() / "HostedManifest.json", output / "HostedManifest.json")
    shutil.copy2(starter_profile_dir / "KokoroRuntimeManifest.json", output / "KokoroRuntimeManifest.json")
    top_level_hosted_files = copy_top_level_hosted_files(args.starter_bundle.resolve(), output)

    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_id": args.repo_id,
        "sdk_commit": sdk_commit,
        "model_card": file_record(output, readme_dest),
        "top_level_hosted_manifest": file_record(output, output / "HostedManifest.json"),
        "top_level_runtime_manifest": file_record(output, output / "KokoroRuntimeManifest.json"),
        "top_level_hosted_files": top_level_hosted_files,
        "profiles": profile_records,
    }
    release_manifest = output / "sdk" / "SDKReleaseManifest.json"
    release_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"prepared HF SDK metadata payload at {output}")
    print(f"  sdk_commit={sdk_commit}")
    for profile in profile_records:
        print(
            f"  {profile['profile']}: models={profile['model_package_count']} "
            f"voices={profile['voice_count']} hosted={profile['hosted_version']}"
        )
    return output


def upload_payload(repo_id: str, payload: Path) -> None:
    """Upload the prepared metadata payload to Hugging Face Hub."""

    from huggingface_hub import HfApi

    token = load_hf_token()
    if not token:
        raise SystemExit("HF upload requires HF_TOKEN, .env HF_TOKEN, or a Hugging Face login cache")
    api = HfApi(token=token)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(payload),
        ignore_patterns=[PAYLOAD_MARKER],
        commit_message="Publish KokoroTTS SDK metadata",
    )
    for stale_path in ("sdk/starter/HostedManifest.json", "sdk/full/HostedManifest.json"):
        try:
            api.delete_file(
                stale_path,
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Remove stale {stale_path}",
            )
            print(f"removed stale HF file: {stale_path}")
        except Exception as exc:
            message = str(exc)
            if "404" not in message and "EntryNotFound" not in message:
                raise
    print(f"uploaded HF SDK metadata payload to {repo_id}")


def main() -> None:
    """Prepare and optionally upload the HF SDK metadata payload."""

    args = parse_args()
    payload = prepare_payload(args)
    if args.upload:
        upload_payload(args.repo_id, payload)


if __name__ == "__main__":
    main()
