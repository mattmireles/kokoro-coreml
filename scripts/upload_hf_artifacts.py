#!/usr/bin/env python3
"""Add model artifacts to the Hugging Face repo in ONE commit.

Step 1 of a release: model packages and G2P assets must be on HF before
``build_sdk_bundle.mjs`` can verify bundle sources against a pinned revision.
Metadata (manifests, model card) is published later by
``prepare_hf_sdk_metadata.py``.

Each argument is ``SRC`` or ``SRC=DEST``: a repo-relative local file or
directory and the HF path it lands at (``DEST`` defaults to ``SRC``). Every
file under a directory ``SRC`` maps to ``DEST/<same relative path>``, e.g.
``coreml-hybrid=coreml/hybrid``.

Overwrite rule. A file already on HF with identical bytes is skipped; one with
different bytes aborts the whole upload. Pinned consumers are safe either way,
since a commit SHA is immutable. The risk is readers of ``main``
(``download_models.py`` without ``--revision``, ``resolve/main`` URLs): they
would silently get new bytes under an old name.

``--replace DEST`` is the one exception, for a same-name package whose new
bytes are I/O-identical by contract (e.g. the ANE-layout
``coreml/kokoro_decoder_pre_{b}s.mlpackage``). In the same single commit it
deletes every HF file under ``DEST`` that the local tree lacks and adds every
local file whose bytes differ (identical files are already right). A
``--replace`` value must equal a mapped DEST exactly: a parent such as
``coreml`` would delete every package under it.

Every DEST must sit under an artifact prefix (``coreml/``, ``g2p/``); metadata
is ``prepare_hf_sdk_metadata.py``'s job.

usage: upload_hf_artifacts.py [--repo-id R] [--dry-run] [--replace DEST ...] SRC[=DEST] ...
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from hf_common import ARTIFACT_PREFIXES, DEFAULT_REPO_ID, REPO_ROOT, git_blob_sha1, load_hf_token, sha256_file


def parse_mapping(arg: str) -> tuple[str, str]:
    """Split ``SRC[=DEST]`` into repo-relative POSIX paths; DEST must be an artifact path."""

    src, _, dest = arg.partition("=")
    src, dest = src.strip("/"), (dest or src).strip("/")
    for value in (src, dest):
        if not value or value.startswith(".") or ".." in Path(value).parts:
            raise SystemExit(f"unsafe mapping: {arg}")
    if not f"{dest}/".startswith(ARTIFACT_PREFIXES):
        raise SystemExit(f"DEST {dest} is outside {', '.join(ARTIFACT_PREFIXES)}: {arg}")
    return src, dest


def check_replace(replace: list[str], mappings: list[tuple[str, str]]) -> list[str]:
    """Return the --replace prefixes, each of which must equal a mapped DEST exactly."""

    dests = {dest for _, dest in mappings}
    prefixes = [prefix.strip("/") for prefix in replace]
    for prefix in prefixes:
        if f"{prefix}/" in ARTIFACT_PREFIXES:
            raise SystemExit(f"--replace {prefix}: refusing to replace a whole artifact root")
        if prefix not in dests:
            raise SystemExit(f"--replace {prefix}: not a DEST of any mapping (must match one exactly)")
    return prefixes


def local_files(mappings: list[tuple[str, str]], root: Path = REPO_ROOT) -> dict[str, Path]:
    """Expand SRC=DEST mappings into {hf_path: local_file}, refusing symlinks."""

    files: dict[str, Path] = {}
    for src, dest in mappings:
        top = root / src
        if top.is_symlink() or not top.exists():
            raise SystemExit(f"missing or symlinked artifact: {src}")
        candidates = [top] if top.is_file() else sorted(p for p in top.rglob("*") if p.is_file() or p.is_symlink())
        for path in candidates:
            if path.is_symlink():
                raise SystemExit(f"refusing symlink: {path}")
            if path.name == ".DS_Store":
                continue
            hf_path = dest if path == top else f"{dest}/{path.relative_to(top).as_posix()}"
            if hf_path in files:
                raise SystemExit(f"two sources map to {hf_path}")
            files[hf_path] = path
    return files


def remote_id(info: Any) -> str:
    """Return the content id HF stores for a file: LFS sha256 or git blob id."""

    return info.lfs.sha256 if info.lfs else info.blob_id


def plan_upload(api: Any, repo_id: str, files: dict[str, Path], replace: list[str]) -> dict[str, Any]:
    """Return {parent, add, delete, changes} for one commit, or raise SystemExit.

    `replace` must already have passed check_replace().

    `changes` lists (hf_path, old_id, new_id) under replaced prefixes; old_id
    is None for an added file and new_id None for a deleted one.
    """

    from huggingface_hub.utils import EntryNotFoundError

    under = lambda path, prefix: path == prefix or path.startswith(prefix + "/")  # noqa: E731
    parent = api.model_info(repo_id).sha
    remote = {info.path: info for info in api.get_paths_info(repo_id, sorted(files), revision=parent, expand=True)}
    add, delete, changes = [], [], []
    for hf_path, local in sorted(files.items()):
        existing = remote.get(hf_path)
        replaced = any(under(hf_path, prefix) for prefix in replace)
        if existing is None:
            add.append(hf_path)
            if replaced:
                changes.append((hf_path, None, sha256_file(local)))
            continue
        local_id = sha256_file(local) if existing.lfs else git_blob_sha1(local)
        if local_id == remote_id(existing):
            continue
        if not replaced:
            raise SystemExit(f"refusing to overwrite {hf_path}: HF bytes at {parent} differ from local (see --replace)")
        add.append(hf_path)
        changes.append((hf_path, remote_id(existing), local_id))
    for prefix in replace:
        try:
            tree = list(api.list_repo_tree(repo_id, path_in_repo=prefix, recursive=True, revision=parent, expand=True))
        except EntryNotFoundError:
            tree = []
        for item in tree:
            if getattr(item, "blob_id", None) is not None and item.path not in files:
                delete.append(item.path)
                changes.append((item.path, remote_id(item), None))
    return {"parent": parent, "add": add, "delete": sorted(delete), "changes": changes}


def print_plan(plan: dict[str, Any], dry_run: bool) -> None:
    """Print the commit plan: old->new ids under replaced prefixes, then counts."""

    for path, old, new in plan["changes"]:
        print(f"  replace {path}: {(old or '-')[:12]} -> {(new or 'deleted')[:12]}")
    if dry_run:
        for path in plan["add"]:
            print(f"  would add {path}")
        for path in plan["delete"]:
            print(f"  would delete {path}")
    print(f"parent={plan['parent']} add={len(plan['add'])} delete={len(plan['delete'])} "
          f"replaced_changes={len(plan['changes'])}")


def commit_plan(api: Any, repo_id: str, files: dict[str, Path], plan: dict[str, Any], message: str) -> str:
    """Create the single add+delete commit on top of the planned parent."""

    from huggingface_hub import CommitOperationAdd, CommitOperationDelete

    operations: list[Any] = [CommitOperationAdd(path_in_repo=p, path_or_fileobj=str(files[p])) for p in plan["add"]]
    operations += [CommitOperationDelete(path_in_repo=p) for p in plan["delete"]]
    commit = api.create_commit(
        repo_id=repo_id,
        repo_type="model",
        operations=operations,
        commit_message=message,
        parent_commit=plan["parent"],
    )
    return commit.oid


def main() -> None:
    """Upload new artifact files in one commit and print its SHA."""

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mappings", nargs="+", metavar="SRC[=DEST]")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--message", default="Add model artifacts")
    parser.add_argument("--replace", action="append", default=[], metavar="DEST")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from huggingface_hub import HfApi

    api = HfApi(token=load_hf_token())
    mappings = [parse_mapping(arg) for arg in args.mappings]
    replace = check_replace(args.replace, mappings)
    files = local_files(mappings)
    plan = plan_upload(api, args.repo_id, files, replace)
    print_plan(plan, args.dry_run)
    if not plan["add"] and not plan["delete"]:
        print(f"nothing to upload; artifacts already at {plan['parent']}")
        return
    if args.dry_run:
        return
    print(f"artifact commit: {commit_plan(api, args.repo_id, files, plan, args.message)}")


if __name__ == "__main__":
    main()
