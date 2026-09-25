#!/usr/bin/env python3
"""Add model artifacts to the Hugging Face repo in one commit, never overwriting.

Step 1 of a release: model packages and G2P assets must be on HF before
``build_sdk_bundle.mjs`` can verify bundle sources against a pinned revision.
Metadata (manifests, model card) is published later by
``prepare_hf_sdk_metadata.py``.

Each argument is a repo-relative file or directory (e.g.
``coreml/kokoro_duration_multifunction.mlpackage`` or ``g2p``); its local path
and HF path are the same. A file already on HF with identical bytes is
skipped; one with different bytes aborts the whole upload, because consumers
pinned to older commits rely on existing paths meaning what they meant.

usage: upload_hf_artifacts.py [--repo-id R] [--dry-run] PATH [PATH ...]
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from prepare_hf_sdk_metadata import DEFAULT_REPO_ID, REPO_ROOT, load_hf_token, sha256_file


def git_blob_sha1(path: Path) -> str:
    """Return the git blob id HF reports for a non-LFS file."""

    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def local_files(relative_paths: list[str]) -> list[str]:
    """Expand repo-relative files and directories into sorted file paths."""

    files = []
    for relative in relative_paths:
        root = REPO_ROOT / relative
        if root.is_symlink() or not root.exists():
            raise SystemExit(f"missing or symlinked artifact: {relative}")
        candidates = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file() or p.is_symlink())
        for path in candidates:
            if path.is_symlink():
                raise SystemExit(f"refusing symlink: {path}")
            if path.name == ".DS_Store":
                continue
            files.append(path.relative_to(REPO_ROOT).as_posix())
    return files


def main() -> None:
    """Upload new artifact files in one commit and print its SHA."""

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--message", default="Add model artifacts")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi(token=load_hf_token())
    parent = api.model_info(args.repo_id).sha
    files = local_files(args.paths)
    remote = {info.path: info for info in api.get_paths_info(args.repo_id, files, revision=parent, expand=True)}

    to_add = []
    for relative in files:
        local = REPO_ROOT / relative
        existing = remote.get(relative)
        if existing is None:
            to_add.append(relative)
            continue
        same = (
            existing.lfs.sha256 == sha256_file(local) if existing.lfs
            else existing.blob_id == git_blob_sha1(local)
        )
        if not same:
            raise SystemExit(f"refusing to overwrite {relative}: HF bytes at {parent} differ from local")
        print(f"already on HF, identical: {relative}")

    print(f"parent={parent} new_files={len(to_add)}")
    if not to_add:
        print(f"nothing to upload; artifacts already at {parent}")
        return
    if args.dry_run:
        for relative in to_add:
            print(f"  would add {relative}")
        return
    commit = api.create_commit(
        repo_id=args.repo_id,
        repo_type="model",
        operations=[CommitOperationAdd(path_in_repo=p, path_or_fileobj=str(REPO_ROOT / p)) for p in to_add],
        commit_message=args.message,
        parent_commit=parent,
    )
    print(f"artifact commit: {commit.oid}")


if __name__ == "__main__":
    main()
