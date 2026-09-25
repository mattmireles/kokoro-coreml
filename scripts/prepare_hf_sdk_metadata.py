#!/usr/bin/env python3
"""Prepare the Hugging Face SDK metadata payload for KokoroTTS releases.

This script uploads only lightweight metadata and docs, in ONE HF commit:

- README.md, sourced from README/hf-model-card.md.
- Top-level HostedManifest.json and KokoroRuntimeManifest.json for the starter
  profile, preserving the current public discovery contract.
- sdk/<profile>/KokoroRuntimeManifest.json for each checked bundle profile.
- sdk/full/HostedManifest.json, the product manifest.
- Runtime and voice files both manifests reference, copied to the HF root.
- sdk/SDKReleaseManifest.json, which records checksums and profile summaries.
- Deletion of stale files (sdk/starter/HostedManifest.json).

Model packages (coreml/) and G2P assets (g2p/) are uploaded beforehand by
upload_hf_artifacts.py. Every path in both hosted manifests is repo-root
relative, so each manifest is hydratable at the commit this script creates --
and only there or later. Before committing, files the payload does not carry
are checked at the parent commit; after committing, both manifests are
hydrated at the new commit SHA and every file is checked. That SHA is the pin.
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
# Hosted manifests published by this script, relative to the HF repo root.
HOSTED_MANIFESTS = {"starter": "HostedManifest.json", "full": "sdk/full/HostedManifest.json"}
# Paths hosted by upload_hf_artifacts.py, never by this metadata commit.
ARTIFACT_PREFIXES = ("coreml/", "g2p/")
STALE_PATHS = ("sdk/starter/HostedManifest.json",)


@dataclass(frozen=True)
class ProfileInput:
    """Input paths and metadata for one SDK bundle profile."""

    name: str
    bundle: Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for payload preparation and upload."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face repo ID")
    parser.add_argument("--output", type=Path, help="Directory for the prepared payload")
    parser.add_argument("--starter-bundle", type=Path, help="Validated starter SDK bundle")
    parser.add_argument("--full-bundle", type=Path, help="Validated full SDK bundle")
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
    parser.add_argument(
        "--verify",
        metavar="SHA",
        help="Only hydrate-check both hosted manifests at this HF commit.",
    )
    args = parser.parse_args()
    if not args.verify and not (args.output and args.starter_bundle and args.full_bundle):
        parser.error("--output, --starter-bundle and --full-bundle are required unless --verify is given")
    return args


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
        "hf_artifact_revision": runtime.get("hf_artifact_revision"),
        "hosted_manifest": HOSTED_MANIFESTS[profile.name],
        "hosted_version": hosted.get("version"),
        "minimum_platforms": runtime.get("minimum_platforms"),
        "buckets": runtime.get("buckets"),
        "duration_token_sizes": runtime.get("duration_token_sizes"),
        "model_package_count": len(runtime.get("model_packages") or []),
        "voice_count": len(runtime.get("voices") or []),
        "runtime_manifest": file_record(output, runtime_dest),
    }


def copy_top_level_hosted_files(bundle: Path, output: Path) -> list[dict[str, Any]]:
    """Copy a bundle's runtime and voice files to the payload root.

    Model packages and G2P assets are already canonical HF paths. A file both
    profiles carry must be byte-identical, since both land on one HF path.
    """

    hosted = load_json(bundle / "HostedManifest.json")
    records = []
    for entry in hosted.get("files") or []:
        relative_path = entry.get("path")
        if not isinstance(relative_path, str):
            continue
        if relative_path == "KokoroRuntimeManifest.json" or relative_path.startswith(ARTIFACT_PREFIXES):
            continue
        source = contained_payload_path(bundle, relative_path)
        if not source.is_file():
            raise SystemExit(f"{bundle} HostedManifest references missing local file: {relative_path}")
        destination = contained_payload_path(output, relative_path)
        if destination.exists():
            if sha256_file(destination) != sha256_file(source):
                raise SystemExit(f"profiles disagree on the bytes of {relative_path}")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        records.append(file_record(output, destination))
    return records


def write_full_hosted_manifest(full_bundle: Path, output: Path) -> None:
    """Write sdk/full/HostedManifest.json with every path HF-repo-root relative.

    The bundle layout mirrors the HF root (coreml/, voices/, runtime/, g2p/)
    except for the runtime manifest, which lives under sdk/full/ on HF.
    """

    hosted = load_json(full_bundle / "HostedManifest.json")
    for entry in hosted["files"]:
        if entry["path"] == "KokoroRuntimeManifest.json":
            entry["path"] = "sdk/full/KokoroRuntimeManifest.json"
    destination = output / HOSTED_MANIFESTS["full"]
    destination.write_text(json.dumps(hosted, indent=2) + "\n", encoding="utf-8")


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
    revisions = {profile["hf_artifact_revision"] for profile in profile_records}
    repo_ids = {profile["hf_repo_id"] for profile in profile_records}
    if repo_ids != {args.repo_id}:
        raise SystemExit(f"profile HF repo IDs do not all match {args.repo_id}: {sorted(repo_ids)}")
    if len(revisions) != 1:
        raise SystemExit(f"profile HF revisions do not match: {sorted(revisions)}")

    starter_profile_dir = output / "sdk" / "starter"
    shutil.copy2(args.starter_bundle.resolve() / "HostedManifest.json", output / "HostedManifest.json")
    shutil.copy2(starter_profile_dir / "KokoroRuntimeManifest.json", output / "KokoroRuntimeManifest.json")
    top_level_hosted_files = copy_top_level_hosted_files(args.starter_bundle.resolve(), output)
    top_level_hosted_files += copy_top_level_hosted_files(args.full_bundle.resolve(), output)
    top_level_hosted_files = sorted({r["path"]: r for r in top_level_hosted_files}.values(), key=lambda r: r["path"])
    write_full_hosted_manifest(args.full_bundle.resolve(), output)
    for profile in profile_records:
        profile["hosted_manifest_record"] = file_record(output, output / profile["hosted_manifest"])

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


def hf_api():
    """Return an authenticated HfApi."""

    from huggingface_hub import HfApi

    token = load_hf_token()
    if not token:
        raise SystemExit("HF upload requires HF_TOKEN, .env HF_TOKEN, or a Hugging Face login cache")
    return HfApi(token=token)


def check_hosted_entries(api: Any, repo_id: str, revision: str, entries: list[dict[str, Any]]) -> list[str]:
    """Return problems for hosted entries whose bytes at `revision` differ.

    Uses the same resolve/<revision>/<path> URL a consumer hydrates. LFS files
    are checked from the served size + sha256 etag; any other file (small,
    non-LFS, where the etag is a git blob id) is downloaded and hashed.
    """

    from huggingface_hub import get_hf_file_metadata, hf_hub_url

    from inspect_hf_artifacts import fetch_repo_bytes

    problems = []
    for entry in entries:
        path = entry["path"]
        try:
            meta = get_hf_file_metadata(hf_hub_url(repo_id, path, revision=revision), token=api.token)
        except Exception as exc:  # 404 and friends: the manifest names a file that is not there
            problems.append(f"{path}: not resolvable at {revision} ({type(exc).__name__})")
            continue
        if meta.commit_hash != revision:
            problems.append(f"{path}: resolved commit {meta.commit_hash}, expected {revision}")
        if meta.size == entry["bytes"] and meta.etag == entry["sha256"]:
            continue
        data = fetch_repo_bytes(repo_id, revision, path)
        observed = hashlib.sha256(data or b"").hexdigest()
        if data is None or len(data) != entry["bytes"] or observed != entry["sha256"]:
            problems.append(f"{path}: expected {entry['bytes']} B {entry['sha256'][:12]}, "
                            f"got {len(data or b'')} B {observed[:12]}")
    return problems


def verify_published(api: Any, repo_id: str, revision: str, payload: Path | None = None) -> None:
    """Hydrate-check every hosted manifest at `revision`; exit non-zero on any mismatch."""

    from inspect_hf_artifacts import fetch_repo_bytes

    failed = False
    for profile, manifest_path in HOSTED_MANIFESTS.items():
        data = fetch_repo_bytes(repo_id, revision, manifest_path)
        if data is None:
            print(f"FAIL {manifest_path}: missing at {revision}")
            failed = True
            continue
        digest = hashlib.sha256(data).hexdigest()
        if payload is not None and digest != sha256_file(payload / manifest_path):
            print(f"FAIL {manifest_path}: published bytes differ from the prepared payload")
            failed = True
            continue
        entries = json.loads(data)["files"]
        problems = check_hosted_entries(api, repo_id, revision, entries)
        total = sum(entry["bytes"] for entry in entries)
        if problems:
            failed = True
            print(f"FAIL {manifest_path} ({profile}): {len(problems)} of {len(entries)} files bad")
            for problem in problems:
                print(f"  {problem}")
        else:
            print(f"PASS {manifest_path} ({profile}): {len(entries)} files, {total} bytes, manifest sha256 {digest}")
    if failed:
        raise SystemExit(f"post-publish verification FAILED at {repo_id}@{revision}")
    print(f"verified: pin https://huggingface.co/{repo_id}/resolve/{revision}/<manifest path>")


def upload_payload(repo_id: str, payload: Path) -> str:
    """Publish the payload and stale-file deletions as ONE commit, then verify it.

    Returns the new commit SHA, the revision consumers should pin.
    """

    from huggingface_hub import CommitOperationAdd, CommitOperationDelete

    api = hf_api()
    parent = api.model_info(repo_id).sha

    # Files the manifests name but the payload does not carry must already be
    # right at the parent, or the new commit would publish broken manifests.
    carried = {p.relative_to(payload).as_posix() for p in payload.rglob("*") if p.is_file()}
    external = {}
    for manifest_path in HOSTED_MANIFESTS.values():
        for entry in load_json(payload / manifest_path)["files"]:
            if entry["path"] not in carried:
                external[entry["path"]] = entry
    problems = check_hosted_entries(api, repo_id, parent, sorted(external.values(), key=lambda e: e["path"]))
    if problems:
        raise SystemExit("refusing to publish; artifacts at parent " + parent + " do not match:\n  " + "\n  ".join(problems))
    print(f"pre-publish: {len(external)} artifact files already correct at parent {parent}")

    operations: list[Any] = [
        CommitOperationAdd(path_in_repo=relative, path_or_fileobj=str(payload / relative))
        for relative in sorted(carried - {PAYLOAD_MARKER})
    ]
    existing = {info.path for info in api.get_paths_info(repo_id, list(STALE_PATHS), revision=parent)}
    for stale_path in STALE_PATHS:
        if stale_path in existing:
            operations.append(CommitOperationDelete(path_in_repo=stale_path))
            print(f"removing stale HF file: {stale_path}")
    commit = api.create_commit(
        repo_id=repo_id,
        repo_type="model",
        operations=operations,
        commit_message="Publish KokoroTTS SDK metadata",
        parent_commit=parent,
    )
    print(f"published HF SDK metadata to {repo_id} in commit {commit.oid}")
    verify_published(api, repo_id, commit.oid, payload)
    return commit.oid


def main() -> None:
    """Prepare and optionally upload the HF SDK metadata payload."""

    args = parse_args()
    if args.verify:
        verify_published(hf_api(), args.repo_id, args.verify)
        return
    payload = prepare_payload(args)
    if args.upload:
        upload_payload(args.repo_id, payload)


if __name__ == "__main__":
    main()
