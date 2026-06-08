"""Benchmark calibration templates and artifact validation for sim2real deploy."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robodeploy.calibration.base import SCHEMA_VERSION
from robodeploy.calibration.store import CalibrationStore

_DEFAULT_REQUIRED = ("kinematic", "extrinsic", "system_id")


@dataclass(frozen=True)
class CalibrationTemplate:
    """Manifest of calibration artifacts required before real-hardware eval."""

    path: Path
    schema_version: str
    robot_id: str
    artifacts: dict[str, str]
    notes: str = ""

    def artifact_store_name(self, key: str) -> str:
        """Map template key (e.g. ``kinematic``) to CalibrationStore artifact name."""
        filename = self.artifacts.get(key, "")
        if not filename:
            raise KeyError(f"template has no artifact key {key!r}")
        return Path(filename).stem


def load_calibration_template(path: Path | str) -> CalibrationTemplate:
    """Load ``calibration_template.json`` from a benchmark task directory."""
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"calibration template must be a JSON object: {p}")
    artifacts: dict[str, str] = {}
    for key, value in data.items():
        if key in ("schema_version", "robot_id", "notes"):
            continue
        if isinstance(value, str) and value.endswith(".json"):
            artifacts[key] = value
    return CalibrationTemplate(
        path=p.resolve(),
        schema_version=str(data.get("schema_version", SCHEMA_VERSION)),
        robot_id=str(data.get("robot_id", "default")),
        artifacts=artifacts,
        notes=str(data.get("notes", "")),
    )


def find_task_calibration_template(task_dir: Path | str) -> Path | None:
    path = Path(task_dir) / "calibration_template.json"
    return path if path.is_file() else None


def validate_calibration_artifacts(
    template: CalibrationTemplate,
    store: CalibrationStore,
    *,
    required: tuple[str, ...] = _DEFAULT_REQUIRED,
) -> list[str]:
    """Return template keys whose artifacts are missing from ``store``."""
    missing: list[str] = []
    for key in required:
        if key not in template.artifacts:
            missing.append(key)
            continue
        name = template.artifact_store_name(key)
        artifact_path = store._path(name, robot_id=template.robot_id)
        if not artifact_path.is_file():
            missing.append(key)
    return missing


def seed_calibration_artifacts(
    template: CalibrationTemplate,
    store: CalibrationStore,
    *,
    defaults: dict[str, dict[str, Any]] | None = None,
) -> list[Path]:
    """Write minimal placeholder artifacts for dry-run / CI (no hardware)."""
    placeholders = defaults or _default_placeholders()
    written: list[Path] = []
    for key, filename in template.artifacts.items():
        if not filename.endswith(".json"):
            continue
        name = Path(filename).stem
        payload = placeholders.get(key, {"placeholder": True, "key": key})
        written.append(store.save(name, payload, robot_id=template.robot_id))
    return written


def _default_placeholders() -> dict[str, dict[str, Any]]:
    return {
        "kinematic": {
            "format": "robodeploy-linear-kinematic-v1",
            "joints": [{"name": "j0", "zero": 2048.0, "scale": 650.0, "soft_min": -1.0, "soft_max": 1.0}],
        },
        "extrinsic": {
            "position": [0.0, 0.0, 0.5],
            "orientation": [1.0, 0.0, 0.0, 0.0],
            "source": "placeholder",
        },
        "system_id": {
            "friction": {"joint_0": {"coulomb_Nm": 0.1, "viscous_Nm_per_rad_s": 0.01, "joint_idx": 0}},
            "payload_mass_kg": 0.0,
        },
        "ft_calibration": {
            "bias_N": [0.0, 0.0, 0.0],
            "scale": [1.0, 1.0, 1.0],
        },
    }
