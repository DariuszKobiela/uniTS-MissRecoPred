"""Run manifest generation for reproducible rebuttal experiments."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def git_status_porcelain() -> str:
    try:
        return (
            subprocess.check_output(["git", "status", "--porcelain=v1"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unavailable"


def collect_file_checksums(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw in paths:
        path = Path(raw)
        if not path.is_file():
            continue
        records.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return records


def build_run_manifest(
    *,
    run_id: str,
    config_paths: Sequence[str | Path],
    seed: int | None = None,
    artifact_paths: Sequence[str | Path] | None = None,
    notes: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable run manifest."""
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "git_commit": git_commit(),
        "git_clean": git_status_porcelain() == "",
        "git_status": git_status_porcelain(),
        "seed": seed,
        "configs": collect_file_checksums(config_paths),
        "artifacts": collect_file_checksums(artifact_paths or []),
        "notes": list(notes or []),
    }
    return manifest


def write_run_manifest(path: str | Path, manifest: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(manifest), indent=2, ensure_ascii=False), encoding="utf-8")
    return target
