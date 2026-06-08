"""Shipped asset manifest and verification helpers."""

from __future__ import annotations

import hashlib
import json
from importlib import resources
from pathlib import Path
from typing import Any


def manifest_path() -> Path:
    with resources.as_file(resources.files("robodeploy").joinpath("_assets/manifest.json")) as p:
        return Path(p)


def load_manifest() -> dict[str, Any]:
    return json.loads(manifest_path().read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_assets(*, repo_root: Path | None = None) -> list[dict[str, Any]]:
    """Return per-asset verification rows (ok, missing, or hash mismatch)."""
    manifest = load_manifest()
    root = repo_root or Path(__file__).resolve().parents[1]
    rows: list[dict[str, Any]] = []
    for entry in manifest.get("assets", []):
        rel = str(entry["path"])
        path = root / rel
        expected = str(entry.get("sha256", ""))
        row = {"name": entry.get("name"), "path": rel, "expected": expected}
        if not path.is_file():
            row["status"] = "missing"
            rows.append(row)
            continue
        actual = sha256_file(path)
        row["actual"] = actual
        row["status"] = "ok" if (not expected or actual == expected) else "mismatch"
        rows.append(row)
    return rows
