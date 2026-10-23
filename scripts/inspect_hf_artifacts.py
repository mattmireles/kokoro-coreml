#!/usr/bin/env python3
"""Inspect Hugging Face Kokoro Core ML artifacts; exits non-zero if a profile manifest names a file HF lacks."""

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

from hf_common import DEFAULT_REPO_ID, INVENTORY_MANIFESTS, load_sdk_profiles

# Files the metadata commit publishes; the root voices are the starter's.
SDK_METADATA = {
    "KokoroRuntimeManifest.json",
    "HostedManifest.json",
    "runtime/kokoro-vocab.json",
    "runtime/hnsf_weights.json",
    "sdk/SDKReleaseManifest.json",
    "sdk/starter/KokoroRuntimeManifest.json",
    "sdk/full/KokoroRuntimeManifest.json",
    *(f"voices/{voice}.bin" for voice in load_sdk_profiles()["profiles"]["starter"]["voices"]),
}


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the HF artifact inspector."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face repo ID")
    parser.add_argument("--revision", help="Optional HF revision/commit")
    parser.add_argument("--output", type=Path, help="Write JSON report to this path")
    parser.add_argument(
        "--verify-hosted-digests",
        action="store_true",
        help="Verify every file the starter hosted manifest and full runtime manifest name (LFS by HF sha256, others by download).",
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


def manifest_entries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the {path, bytes, sha256} HF files a hosted or runtime manifest names.

    A hosted manifest lists them directly. A runtime manifest names model
    directories (coreml/<x>.mlpackage, coreml/hybrid) with per-file digests and
    g2p assets at their HF paths; its voices/<v>.bin live at kokoro.js/voices/.
    """

    if "files" in manifest:
        return list(manifest["files"])
    entries = []
    for pkg in manifest.get("model_packages") or []:
        for item in pkg.get("files") or []:
            entries.append({"path": f"{pkg['path']}/{item['path']}", "bytes": item["bytes"], "sha256": item["sha256"]})
    for voice in manifest.get("voices") or []:
        entries.append({**voice, "path": "kokoro.js/" + voice["path"]})
    return entries + list(manifest.get("g2p_assets") or [])


def digest_mismatches(
    repo_id: str, revision: str, entries: list[dict[str, Any]], siblings: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Return entries whose bytes at `revision` differ from the manifest.

    LFS files are checked from the HF-reported size and sha256; any other file
    (plan.json, Manifest.json, small JSON) is downloaded and hashed.
    """

    mismatches = []
    for entry in entries:
        path = entry["path"]
        lfs = (siblings.get(path) or {}).get("lfs") or {}
        if lfs.get("sha256") == entry["sha256"] and lfs.get("size") == entry["bytes"]:
            continue
        data = fetch_repo_bytes(repo_id, revision, path)
        if data is None:
            mismatches.append({"path": path, "reason": "missing"})
            continue
        observed_sha256 = hashlib.sha256(data).hexdigest()
        if len(data) != entry["bytes"] or observed_sha256 != entry["sha256"]:
            mismatches.append({
                "path": path,
                "reason": "digest_mismatch",
                "expected_bytes": entry["bytes"],
                "observed_bytes": len(data),
                "expected_sha256": entry["sha256"],
                "observed_sha256": observed_sha256,
            })
    return mismatches


def model_entry(path: str) -> str | None:
    """Return the model directory a repo file belongs to.

    Under coreml/ that is coreml/<first segment>, so coreml/hybrid groups its
    plan.json, adain.bin and row packages. Elsewhere (the paper's coreml-*/
    record) it is the enclosing `.mlpackage`.
    """

    parts = path.split("/")
    if parts[0] == "coreml" and len(parts) >= 3:
        return "/".join(parts[:2])
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
    by_path = {}

    for sibling in siblings:
        path = sibling.get("rfilename")
        if not isinstance(path, str):
            continue
        by_path[path] = sibling
        size = int(sibling.get("size") or sibling.get("lfs", {}).get("size") or 0)
        sha256 = sibling.get("lfs", {}).get("sha256")
        pkg = model_entry(path)
        if pkg:
            packages[pkg]["file_count"] += 1
            packages[pkg]["bytes"] += size
            packages[pkg]["files"].append({"path": path, "bytes": size, "sha256": sha256, "blob_id": sibling.get("blobId")})
        elif path.startswith("kokoro.js/voices/") and path.endswith(".bin"):
            voices.append({"path": path, "voice": Path(path).stem, "bytes": size, "sha256": sha256})

    resolved_revision = info.get("sha")
    manifest_checks = {}
    for profile, manifest_path in INVENTORY_MANIFESTS.items():
        manifest = None
        if isinstance(resolved_revision, str) and manifest_path in by_path:
            manifest = fetch_repo_json(repo_id, resolved_revision, manifest_path)
        entries = manifest_entries(manifest) if manifest else []
        check = {
            "manifest": manifest_path,
            "present": manifest is not None,
            "file_count": len(entries),
            "unresolved_files": sorted(entry["path"] for entry in entries if entry["path"] not in by_path),
        }
        if verify_digests and isinstance(resolved_revision, str):
            check["digest_mismatches"] = digest_mismatches(repo_id, resolved_revision, entries, by_path)
        manifest_checks[profile] = check
    sdk_release_manifest = None
    if isinstance(resolved_revision, str) and "sdk/SDKReleaseManifest.json" in by_path:
        sdk_release_manifest = fetch_repo_json(repo_id, resolved_revision, "sdk/SDKReleaseManifest.json")

    report = {
        "schema_version": 2,
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
        "missing_sdk_metadata": sorted(SDK_METADATA - set(by_path)),
        # Only the root HostedManifest.json is hydratable; any sdk/** copy
        # would resolve paths under its own folder and 404.
        "unexpected_sdk_hosted_manifests": sorted(
            path for path in by_path if path.startswith("sdk/") and path.endswith("HostedManifest.json")
        ),
        "manifest_checks": manifest_checks,
        "hosted_digests_verified": verify_digests,
        "sdk_release_manifest": sdk_release_manifest,
    }
    return report


def inventory_problems(report: dict[str, Any]) -> list[str]:
    """One line per inventory manifest that is missing, unresolved or mismatched."""

    return [
        f"{profile}: " + ("manifest missing" if not check["present"] else
            f"{len(check['unresolved_files'])} unresolved, {len(check.get('digest_mismatches', []))} mismatched")
        for profile, check in report["manifest_checks"].items()
        if not check["present"] or check["unresolved_files"] or check.get("digest_mismatches")
    ]


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
    bad = inventory_problems(report)
    if bad:
        raise SystemExit("HF inventory problems: " + "; ".join(bad))


if __name__ == "__main__":
    main()
