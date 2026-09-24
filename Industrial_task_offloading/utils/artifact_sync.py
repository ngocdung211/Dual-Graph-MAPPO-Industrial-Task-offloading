"""Copy completed local training runs to Google Drive artifact storage."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import uuid


def sync_completed_run(
    local_run_dir: str | Path,
    drive_artifact_root: str | Path,
) -> str:
    """Copy one completed local run into the Drive ``runs`` directory.

    Args:
        local_run_dir: Local directory containing every completed run artifact.
        drive_artifact_root: Google Drive artifact root in POSIX or Windows form.

    Returns:
        Destination run directory as a string.

    Raises:
        FileNotFoundError: If the completed local run does not exist.
        FileExistsError: If a run with the same name already exists on Drive.
        RuntimeError: If Windows-side synchronization fails from WSL.
    """
    source = Path(local_run_dir).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Completed local run was not found: {source}")

    drive_root = str(drive_artifact_root).strip()
    if not drive_root:
        raise ValueError("drive_artifact_root must not be empty")
    if _is_windows_path(drive_root) and os.name != "nt":
        return _sync_windows_drive_from_wsl(source, drive_root)
    return str(_copy_run_atomically(source, Path(drive_root) / "runs"))


def _copy_run_atomically(source: Path, runs_root: Path) -> Path:
    """Publish a completed run only after its temporary copy succeeds."""
    runs_root.mkdir(parents=True, exist_ok=True)
    destination = runs_root / source.name
    if destination.exists():
        raise FileExistsError(f"Drive run already exists: {destination}")

    temporary = runs_root / f".{source.name}.tmp-{uuid.uuid4().hex}"
    try:
        shutil.copytree(source, temporary)
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def _sync_windows_drive_from_wsl(source: Path, drive_root: str) -> str:
    """Use Windows PowerShell to copy from WSL into a Drive Desktop path."""
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "sync_completed_run.ps1"
    )
    script_windows = _convert_wsl_path(script_path)
    source_windows = _convert_wsl_path(source)
    completed = subprocess.run(
        _build_powershell_sync_command(
            script_windows=script_windows,
            source_windows=source_windows,
            drive_root=drive_root,
            run_name=source.name,
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        details = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"Drive synchronization failed: {details}")
    return completed.stdout.strip().splitlines()[-1]


def _convert_wsl_path(path: Path) -> str:
    """Convert one WSL path to its Windows representation."""
    return subprocess.run(
        ["wslpath", "-w", str(path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _build_powershell_sync_command(
    script_windows: str,
    source_windows: str,
    drive_root: str,
    run_name: str,
) -> list[str]:
    """Build a PowerShell -File invocation with explicit named parameters."""
    return [
        "powershell.exe",
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        script_windows,
        "-Source",
        source_windows,
        "-ArtifactRoot",
        drive_root,
        "-RunName",
        run_name,
    ]


def _is_windows_path(path: str) -> bool:
    """Return whether a path starts with a Windows drive prefix."""
    return len(path) >= 3 and path[0].isalpha() and path[1:3] in {":\\", ":/"}
