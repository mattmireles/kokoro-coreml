#!/usr/bin/env python3
"""Inspect Hugging Face Kokoro Core ML artifacts without downloading them."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_REPO_ID = "mattmireles/kokoro-coreml"


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the HF artifact inspector."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face repo ID")
    parser.add_argument("--revision", help="Optional HF revision/commit")
    parser.add_argument("--output", type=Path, help="Write JSON report to this path")
    parser.add_argument(
        "--verify-hosted-digests",
        action="store_true",
        help="Download hosted-manifest files and verify their byte counts and SHA-256 digests.",
    )
    return parser.parse_args()


def fetch_model_info(repo_id: str, revision: str | None) -> dict[str, Any]:
    """Fetch Hugging Face model metadata through the public API."""

    quoted = urllib.parse.quote(repo_id, safe="/")
    suffix = f"/revision/{urllib.parse.quote(revision, safe='')}" if revision else ""
    params = {"blobs": "true"}
    url = f"https://huggingface.co/api/models/{quoted}{suffix}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def fetch_repo_json(repo_id: str, revision: str, path: str) -> dict[str, Any] | None:
    """Fetch a JSON file from a model repo revision, returning None on 404."""

    quoted = urllib.parse.quote(repo_id, safe="/")
    quoted_revision = urllib.parse.quote(revision, safe="")
    quoted_path = urllib.parse.quote(path, safe="/")
    url = f"https://huggingface.co/{quoted}/resolve/{quoted_revision}/{quoted_path}"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def fetch_repo_bytes(repo_id: str, revision: str, path: str) -> bytes | None:
    """Fetch raw bytes from a model repo revision, returning None on 404."""

    quoted = urllib.parse.quote(repo_id, safe="/")
    quoted_revision = urllib.parse.quote(revision, safe="")
    quoted_path = urllib.parse.quote(path, safe="/")
    url = f"https://huggingface.co/{quoted}/resolve/{quoted_revision}/{quoted_path}"
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def verify_hosted_digests(repo_id: str, revision: str, hosted_manifest: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return hosted-manifest entries whose live bytes do not match the manifest."""

    mismatches = []
    for entry in (hosted_manifest or {}).get("files") or []:
        path = entry.get("path")
        if not isinstance(path, str):
            continue
        data = fetch_repo_bytes(repo_id, revision, path)
        if data is None:
            mismatches.append({"path": path, "reason": "missing"})
            continue
        expected_bytes = entry.get("bytes")
        expected_sha256 = entry.get("sha256")
        observed_sha256 = hashlib.sha256(data).hexdigest()
        if len(data) != expected_bytes or observed_sha256 != expected_sha256:
            mismatches.append({
                "path": path,
                "reason": "digest_mismatch",
                "expected_bytes": expected_bytes,
                "observed_bytes": len(data),
                "expected_sha256": expected_sha256,
                "observed_sha256": observed_sha256,
            })
    return mismatches


def package_name(path: str) -> str | None:
    """Return the `.mlpackage` path prefix for a repo file path."""

    marker = ".mlpackage/"
    if marker not in path:
        return None
    return path[: path.index(marker) + len(".mlpackage")]


def inspect_artifacts(
    info: dict[str, Any],
    repo_id: str,
    revision: str | None,
    verify_digests: bool = False,
) -> dict[str, Any]:
    """Build a stable report from Hugging Face API metadata."""

    siblings = info.get("siblings") or []
    packages: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "file_count": 0,
        "bytes": 0,
        "files": [],
    })
    voices = []
    sdk_metadata = set()
    sibling_paths = set()

    for sibling in siblings:
        path = sibling.get("rfilename")
        if not isinstance(path, str):
            continue
        sibling_paths.add(path)
        size = int(sibling.get("size") or sibling.get("lfs", {}).get("size") or 0)
        pkg = package_name(path)
        if pkg:
            packages[pkg]["file_count"] += 1
            packages[pkg]["bytes"] += size
            packages[pkg]["files"].append({
                "path": path,
                "bytes": size,
                "sha256": sibling.get("lfs", {}).get("sha256"),
            })
        elif path.startswith("kokoro.js/voices/") and path.endswith(".bin"):
            voices.append({
                "path": path,
                "voice": Path(path).stem,
                "bytes": size,
                "sha256": sibling.get("lfs", {}).get("sha256"),
            })
        elif path in {
            "KokoroRuntimeManifest.json",
            "HostedManifest.json",
            "runtime/kokoro-vocab.json",
            "runtime/hnsf_weights.json",
            "voices/af_heart.bin",
            "sdk/SDKReleaseManifest.json",
            "sdk/starter/KokoroRuntimeManifest.json",
            "sdk/full/KokoroRuntimeManifest.json",
        }:
            sdk_metadata.add(path)

    resolved_revision = info.get("sha")
    hosted_manifest = None
    sdk_release_manifest = None
    unresolved_hosted_files = []
    unexpected_sdk_hosted_manifests = sorted(
        path for path in sibling_paths
        if path.startswith("sdk/") and path.endswith("/HostedManifest.json")
    )
    if isinstance(resolved_revision, str) and "HostedManifest.json" in sibling_paths:
        hosted_manifest = fetch_repo_json(repo_id, resolved_revision, "HostedManifest.json")
        if hosted_manifest:
            for entry in hosted_manifest.get("files") or []:
                hosted_path = entry.get("path")
                if isinstance(hosted_path, str) and hosted_path not in sibling_paths:
                    unresolved_hosted_files.append(hosted_path)
    if isinstance(resolved_revision, str) and "sdk/SDKReleaseManifest.json" in sibling_paths:
        sdk_release_manifest = fetch_repo_json(repo_id, resolved_revision, "sdk/SDKReleaseManifest.json")
    hosted_digest_mismatches = []
    if verify_digests and isinstance(resolved_revision, str):
        hosted_digest_mismatches = verify_hosted_digests(repo_id, resolved_revision, hosted_manifest)

    required_sdk_metadata = {
        "KokoroRuntimeManifest.json",
        "HostedManifest.json",
        "runtime/kokoro-vocab.json",
        "runtime/hnsf_weights.json",
        "voices/af_heart.bin",
        "sdk/SDKReleaseManifest.json",
        "sdk/starter/KokoroRuntimeManifest.json",
        "sdk/full/KokoroRuntimeManifest.json",
    }

    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_id": repo_id,
        "requested_revision": revision,
        "resolved_revision": resolved_revision,
        "last_modified": info.get("lastModified"),
        "private": info.get("private"),
        "gated": info.get("gated"),
        "sibling_count": len(siblings),
        "model_packages": [
            {
                "path": path,
                "file_count": value["file_count"],
                "bytes": value["bytes"],
                "files": sorted(value["files"], key=lambda item: item["path"]),
            }
            for path, value in sorted(packages.items())
        ],
        "voices": sorted(voices, key=lambda item: item["path"]),
        "missing_sdk_metadata": sorted(required_sdk_metadata - sdk_metadata),
        "hosted_manifest_file_count": len((hosted_manifest or {}).get("files") or []),
        "unresolved_hosted_files": sorted(unresolved_hosted_files),
        "unexpected_sdk_hosted_manifests": unexpected_sdk_hosted_manifests,
        "hosted_digest_mismatches": hosted_digest_mismatches,
        "hosted_digests_verified": verify_digests,
        "sdk_release_manifest": sdk_release_manifest,
    }
    return report


def main() -> None:
    """CLI entry point for the HF artifact inspector."""

    args = parse_args()
    info = fetch_model_info(args.repo_id, args.revision)
    report = inspect_artifacts(info, args.repo_id, args.revision, verify_digests=args.verify_hosted_digests)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
