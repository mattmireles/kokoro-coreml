"""upload_hf_artifacts.py: SRC=DEST mapping and --replace, against a fake HfApi."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_uploader():
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location("upload_hf_artifacts", SCRIPTS / "upload_hf_artifacts.py")
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(SCRIPTS))


def lfs_file(path: str, data: bytes):
    return types.SimpleNamespace(path=path, lfs=types.SimpleNamespace(sha256=hashlib.sha256(data).hexdigest()), blob_id="b")


class FakeApi:
    """Minimal HfApi: a dict of {path: bytes} at one parent commit."""

    def __init__(self, remote: dict[str, bytes]):
        self.remote = remote
        self.commits = []

    def model_info(self, repo_id):
        return types.SimpleNamespace(sha="parent123")

    def get_paths_info(self, repo_id, paths, revision, expand):
        assert revision == "parent123"
        return [lfs_file(p, self.remote[p]) for p in paths if p in self.remote]

    def list_repo_tree(self, repo_id, path_in_repo, recursive, revision, expand):
        assert revision == "parent123"
        return [lfs_file(p, d) for p, d in self.remote.items() if p.startswith(path_in_repo + "/")]

    def create_commit(self, **kwargs):
        self.commits.append(kwargs)
        return types.SimpleNamespace(oid="new456")


def write(root: Path, rel: str, data: bytes) -> None:
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    (root / rel).write_bytes(data)


def test_directory_mapping_lands_under_dest_prefix(tmp_path):
    up = load_uploader()
    write(tmp_path, "coreml-hybrid/plan.json", b"{}")
    write(tmp_path, "coreml-hybrid/hybrid_generator_r6.mlpackage/Manifest.json", b"m")
    files = up.local_files([up.parse_mapping("coreml-hybrid=coreml/hybrid")], root=tmp_path)
    assert sorted(files) == ["coreml/hybrid/hybrid_generator_r6.mlpackage/Manifest.json", "coreml/hybrid/plan.json"]
    assert up.parse_mapping("g2p") == ("g2p", "g2p")
    with pytest.raises(SystemExit, match="unsafe mapping"):
        up.parse_mapping("coreml-fold=../x")


def test_refuses_to_overwrite_differing_bytes_without_replace(tmp_path):
    up = load_uploader()
    write(tmp_path, "coreml-dpre-ane/p.mlpackage/w.bin", b"new")
    files = up.local_files([up.parse_mapping("coreml-dpre-ane/p.mlpackage=coreml/p.mlpackage")], root=tmp_path)
    api = FakeApi({"coreml/p.mlpackage/w.bin": b"old"})
    with pytest.raises(SystemExit, match="refusing to overwrite coreml/p.mlpackage/w.bin"):
        up.plan_upload(api, "r", files, [])


def test_identical_files_are_skipped_and_new_files_added(tmp_path):
    up = load_uploader()
    write(tmp_path, "coreml-fold/f.mlpackage/w.bin", b"same")
    write(tmp_path, "coreml-fold/f.mlpackage/Manifest.json", b"new")
    files = up.local_files([up.parse_mapping("coreml-fold/f.mlpackage=coreml/f.mlpackage")], root=tmp_path)
    plan = up.plan_upload(FakeApi({"coreml/f.mlpackage/w.bin": b"same"}), "r", files, [])
    assert plan == {"parent": "parent123", "add": ["coreml/f.mlpackage/Manifest.json"], "delete": [], "changes": []}


def test_replace_deletes_stale_and_adds_changed_in_one_commit(tmp_path):
    up = load_uploader()
    write(tmp_path, "coreml-dpre-ane/p.mlpackage/Data/weight.bin", b"ane")
    write(tmp_path, "coreml-dpre-ane/p.mlpackage/Manifest.json", b"same")
    files = up.local_files([up.parse_mapping("coreml-dpre-ane/p.mlpackage=coreml/p.mlpackage")], root=tmp_path)
    api = FakeApi({
        "coreml/p.mlpackage/Data/weight.bin": b"gpu",
        "coreml/p.mlpackage/Manifest.json": b"same",
        "coreml/p.mlpackage/Data/stale.bin": b"stale",
        "coreml/p_other.mlpackage/keep.bin": b"keep",
    })
    plan = up.plan_upload(api, "r", files, ["coreml/p.mlpackage"])
    assert plan["add"] == ["coreml/p.mlpackage/Data/weight.bin"]
    assert plan["delete"] == ["coreml/p.mlpackage/Data/stale.bin"]
    assert [(path, new is None) for path, _, new in plan["changes"]] == [
        ("coreml/p.mlpackage/Data/weight.bin", False),
        ("coreml/p.mlpackage/Data/stale.bin", True),
    ]
    assert up.commit_plan(api, "r", files, plan, "msg") == "new456"
    (commit,) = api.commits
    assert commit["parent_commit"] == "parent123"
    assert [type(op).__name__ for op in commit["operations"]] == ["CommitOperationAdd", "CommitOperationDelete"]


def test_replace_refuses_prefix_not_named_as_a_mapping():
    up = load_uploader()
    mappings = [up.parse_mapping("coreml-fold/f.mlpackage=coreml/f.mlpackage")]
    with pytest.raises(SystemExit, match="not a DEST of any mapping"):
        up.check_replace(["coreml/kokoro_decoder_pre_3s.mlpackage"], mappings)
    assert up.check_replace(["/coreml/f.mlpackage/"], mappings) == ["coreml/f.mlpackage"]


def test_replace_refuses_parent_of_a_dest():
    """Regression: `--replace coreml` contained a DEST, so it would delete every coreml/ package."""

    up = load_uploader()
    mappings = [up.parse_mapping("coreml-dpre-ane/p.mlpackage=coreml/p.mlpackage")]
    for parent in ("coreml", "coreml/"):
        with pytest.raises(SystemExit, match="must match one exactly|whole artifact root"):
            up.check_replace([parent], mappings)


def test_replace_refuses_artifact_root_even_when_mapped():
    """Regression: `coreml-fold=coreml --replace coreml` named a mapped DEST, so it passed and would wipe coreml/."""

    up = load_uploader()
    mappings = [up.parse_mapping("coreml-fold=coreml")]
    for root in ("coreml", "g2p"):
        with pytest.raises(SystemExit, match="whole artifact root"):
            up.check_replace([root], mappings + [up.parse_mapping("g2p=g2p")])


def test_dest_outside_artifact_prefixes_is_refused():
    """Regression: a DEST such as README.md or sdk/... let the artifact commit publish metadata."""

    up = load_uploader()
    for arg in ("x=README.md", "x=sdk/full/KokoroRuntimeManifest.json", "x=coreml-fold/f.mlpackage", "x=g2px/a"):
        with pytest.raises(SystemExit, match="is outside coreml/, g2p/"):
            up.parse_mapping(arg)
    assert up.parse_mapping("coreml-hybrid=coreml/hybrid") == ("coreml-hybrid", "coreml/hybrid")
