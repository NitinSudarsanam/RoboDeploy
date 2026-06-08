"""Unified storage for calibration artifacts."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from robodeploy.calibration.base import SCHEMA_VERSION

_DEFAULT_ENV = "ROBODEPLOY_CALIBRATION_ROOT"


class CalibrationStore:
    """JSON artifact store under ``~/.robodeploy/calibration`` by default."""

    def __init__(self, root: Path | str | None = None) -> None:
        if root is not None:
            self._root = Path(root).expanduser()
        else:
            env = os.environ.get(_DEFAULT_ENV, "").strip()
            self._root = Path(env).expanduser() if env else Path.home() / ".robodeploy" / "calibration"

    @property
    def root(self) -> Path:
        return self._root

    def _path(self, name: str, *, robot_id: str | None = None) -> Path:
        rid = robot_id or "default"
        return self._root / rid / f"{name}.json"

    def save(
        self,
        name: str,
        payload: dict[str, Any],
        *,
        robot_id: str | None = None,
        schema_version: str = SCHEMA_VERSION,
    ) -> Path:
        path = self._path(name, robot_id=robot_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        envelope = {
            "schema_version": schema_version,
            "name": name,
            "robot_id": robot_id or "default",
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "payload": dict(payload),
        }
        path.write_text(json.dumps(envelope, indent=2), encoding="utf-8")
        return path

    def load(self, name: str, *, robot_id: str | None = None) -> dict[str, Any]:
        path = self._path(name, robot_id=robot_id)
        if not path.is_file():
            raise FileNotFoundError(f"Calibration artifact not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Invalid calibration file: {path}")
        if "payload" in data:
            return dict(data["payload"])
        return dict(data)

    def load_envelope(self, name: str, *, robot_id: str | None = None) -> dict[str, Any]:
        path = self._path(name, robot_id=robot_id)
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Invalid calibration file: {path}")
        return dict(data)

    def list_all(self) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        if not self._root.is_dir():
            return entries
        for robot_dir in sorted(self._root.iterdir()):
            if not robot_dir.is_dir():
                continue
            for path in sorted(robot_dir.glob("*.json")):
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                mtime = path.stat().st_mtime
                entries.append(
                    {
                        "name": path.stem,
                        "robot_id": robot_dir.name,
                        "schema_version": data.get("schema_version", "unknown"),
                        "modified": datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat(),
                        "path": str(path),
                    }
                )
        return entries

    def resolve_legacy_so101_path(self) -> Path:
        """Legacy SO-101 path for backward compatibility."""
        return Path.home() / ".robodeploy" / "so101_calibration.json"
