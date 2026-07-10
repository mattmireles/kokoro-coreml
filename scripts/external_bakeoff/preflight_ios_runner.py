#!/usr/bin/env python3
"""Preflight the Soniqo iOS runner before a physical-device benchmark build."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_DIR = REPO_ROOT / "scripts" / "external_bakeoff" / "SoniqoKokoroIOSRunner"
DEFAULT_DEVICE_ID = "F383FC46-FD64-5346-AEC6-59E3E2F8C9CA"
DEFAULT_DERIVED_DATA = Path("/tmp/kokoro-external-bakeoff/ios-runner-derived")


def _run(command: list[str], *, cwd: Path | None = None) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            text=True,
            capture_output=True,
        )
    except FileNotFoundError as exc:
        return {
            "command": command,
            "returncode": 127,
            "stdout": "",
            "stderr": str(exc),
        }
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _device_status(device_id: str) -> dict[str, Any]:
    list_result = _run(["xcrun", "devicectl", "list", "devices"])
    list_text = list_result["stdout"] + list_result["stderr"]
    line = next((item for item in list_text.splitlines() if device_id in item), "")
    details_result = _run(["xcrun", "devicectl", "device", "info", "details", "--device", device_id])
    details_text = details_result["stdout"] + details_result["stderr"]
    is_connected = " connected " in f" {line} " or "tunnelState: connected" in details_text
    is_paired = "pairingState: paired" in details_text or "paired" in line
    return {
        "ok": list_result["returncode"] == 0 and details_result["returncode"] == 0 and is_connected and is_paired,
        "device_id": device_id,
        "line": line,
        "list_command_returncode": list_result["returncode"],
        "details_command_returncode": details_result["returncode"],
        "connected": is_connected,
        "paired": is_paired,
    }


def _signing_status() -> dict[str, Any]:
    team = os.environ.get("DEVELOPMENT_TEAM", "")
    identities_result = _run(["security", "find-identity", "-v", "-p", "codesigning"])
    identity_lines = [
        line.strip()
        for line in identities_result["stdout"].splitlines()
        if re.match(r"^\s*\d+\)", line)
    ]
    profiles_dir = Path.home() / "Library" / "MobileDevice" / "Provisioning Profiles"
    profiles = sorted(profiles_dir.glob("*.mobileprovision")) if profiles_dir.exists() else []
    return {
        "ok": bool(team and identity_lines),
        "development_team": team,
        "identity_count": len(identity_lines),
        "identities": identity_lines,
        "provisioning_profile_count": len(profiles),
    }


def _speech_swift_status() -> dict[str, Any]:
    raw = os.environ.get("SPEECH_SWIFT_PATH", "")
    path = Path(raw).expanduser() if raw else Path()
    package = path / "Package.swift" if raw else Path()
    return {
        "ok": bool(raw and package.exists()),
        "speech_swift_path": raw,
        "package_exists": package.exists() if raw else False,
    }


def _tool_status() -> dict[str, Any]:
    xcodegen = shutil.which("xcodegen")
    xcodebuild = shutil.which("xcodebuild")
    return {
        "ok": bool(xcodegen and xcodebuild),
        "xcodegen": xcodegen or "",
        "xcodebuild": xcodebuild or "",
    }


def _manifest_status() -> dict[str, Any]:
    manifest = RUNNER_DIR / "Sources" / "BenchmarkManifest.generated.swift"
    return {
        "ok": manifest.exists(),
        "path": str(manifest),
    }


def _project_status() -> dict[str, Any]:
    project = RUNNER_DIR / "SoniqoKokoroIOSRunner.xcodeproj"
    return {
        "ok": project.exists(),
        "path": str(project),
    }


def _generate_project() -> dict[str, Any]:
    manifest_result = _run(
        [sys.executable, "scripts/external_bakeoff/generate_ios_runner_manifest.py"],
        cwd=REPO_ROOT,
    )
    if manifest_result["returncode"] != 0:
        return {
            "ok": False,
            "manifest": manifest_result,
            "xcodegen": None,
        }
    xcodegen_result = _run(["xcodegen", "generate"], cwd=RUNNER_DIR)
    return {
        "ok": xcodegen_result["returncode"] == 0,
        "manifest": manifest_result,
        "xcodegen": xcodegen_result,
    }


def _build(device_id: str, derived_data: Path) -> dict[str, Any]:
    return _run(
        [
            "xcodebuild",
            "-project",
            "SoniqoKokoroIOSRunner.xcodeproj",
            "-scheme",
            "SoniqoKokoroIOSRunner",
            "-destination",
            f"id={device_id}",
            "-derivedDataPath",
            str(derived_data),
            "-allowProvisioningUpdates",
            "build",
        ],
        cwd=RUNNER_DIR,
    )


def _summarize(checks: dict[str, dict[str, Any]]) -> bool:
    return all(item.get("ok") for item in checks.values())


def _blockers(checks: dict[str, dict[str, Any]], actions: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if not checks["device"]["ok"]:
        blockers.append("connected iPhone is not available and paired")
    if not checks["signing"]["development_team"]:
        blockers.append("DEVELOPMENT_TEAM is unset")
    if checks["signing"]["identity_count"] == 0:
        blockers.append("no valid code-signing identities found")
    if not checks["speech_swift"]["ok"]:
        blockers.append("SPEECH_SWIFT_PATH does not point to a speech-swift Package.swift")
    if not checks["tools"]["ok"]:
        blockers.append("xcodegen or xcodebuild is missing")
    if not checks["manifest"]["ok"]:
        blockers.append("BenchmarkManifest.generated.swift is missing")
    if not checks["project"]["ok"]:
        blockers.append("SoniqoKokoroIOSRunner.xcodeproj is missing; rerun with --generate-project")
    generate = actions.get("generate_project")
    if generate and not generate.get("ok"):
        blockers.append("xcodegen project generation failed")
    build = actions.get("build")
    if build and build.get("returncode", 0) != 0:
        blockers.append("xcodebuild failed")
    return blockers


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-id", default=DEFAULT_DEVICE_ID)
    parser.add_argument("--derived-data", type=Path, default=DEFAULT_DERIVED_DATA)
    parser.add_argument("--generate-project", action="store_true")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    checks = {
        "device": _device_status(args.device_id),
        "signing": _signing_status(),
        "speech_swift": _speech_swift_status(),
        "tools": _tool_status(),
        "manifest": _manifest_status(),
        "project": _project_status(),
    }
    actions: dict[str, Any] = {}

    if args.generate_project:
        actions["generate_project"] = _generate_project()
        checks["manifest"] = _manifest_status()
        checks["project"] = _project_status()

    ready = _summarize(checks)
    if args.build:
        if not ready:
            actions["build"] = {
                "skipped": True,
                "reason": "preflight checks failed",
            }
        else:
            actions["build"] = _build(args.device_id, args.derived_data)

    ok = ready and not (
        args.build
        and actions.get("build", {}).get("returncode", 0) != 0
    )
    blockers = _blockers(checks, actions)
    payload = {
        "ok": ok,
        "blockers": blockers,
        "runner_dir": str(RUNNER_DIR),
        "checks": checks,
        "actions": actions,
    }

    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
