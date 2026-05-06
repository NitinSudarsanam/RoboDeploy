"""SO-101 follower calibration: URDF radians ↔ Feetech raw ticks (per joint, linear model)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_MODULE_DIR = Path(__file__).resolve().parent


class MissingCalibrationError(RuntimeError):
    """No user calibration file found, or only the bundled template is selected."""


@dataclass(frozen=True)
class JointCalibration:
    """One revolute joint: linear map tick ≈ zero_ticks + q_rad * ticks_per_rad."""

    name: str
    motor_id: int
    zero_ticks: int
    ticks_per_rad: float
    soft_min_rad: float
    soft_max_rad: float


@dataclass(frozen=True)
class SO101Calibration:
    """Full arm (6 joints). ``joints`` order must match policy / URDF joint order."""

    joints: tuple[JointCalibration, ...]
    gripper_open_rad: float | None = None
    gripper_closed_rad: float | None = None

    def __post_init__(self) -> None:
        if len(self.joints) != 6:
            raise ValueError(f"SO101Calibration expects 6 joints, got {len(self.joints)}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SO101Calibration:
        data = dict(data)
        data.pop("is_template", None)
        # Compatibility: accept LeRobot-style MotorCalibration dict:
        # {
        #   "shoulder_pan": {"id": 1, "homing_offset": ..., "range_min": ..., "range_max": ...},
        #   ...
        # }
        # We convert to the neutral linear model using a nominal ticks_per_rad and treat
        # `zero_ticks` as (half_turn + homing_offset). This keeps the file usable without
        # rewriting existing calibration exports.
        if "joints" not in data and _looks_like_lerobot_calibration(data):
            return cls._from_lerobot_style(data)
        joints_in = data.get("joints") or []
        if not isinstance(joints_in, list) or len(joints_in) != 6:
            raise ValueError("calibration JSON must contain a 'joints' list of length 6")
        joints: list[JointCalibration] = []
        for item in joints_in:
            if not isinstance(item, dict):
                raise ValueError("each joint entry must be an object")
            joints.append(
                JointCalibration(
                    name=str(item["name"]),
                    motor_id=int(item["motor_id"]),
                    zero_ticks=int(item["zero_ticks"]),
                    ticks_per_rad=float(item["ticks_per_rad"]),
                    soft_min_rad=float(item["soft_min_rad"]),
                    soft_max_rad=float(item["soft_max_rad"]),
                )
            )
        return cls(
            joints=tuple(joints),
            gripper_open_rad=_optional_float(data.get("gripper_open_rad")),
            gripper_closed_rad=_optional_float(data.get("gripper_closed_rad")),
        )

    @classmethod
    def _from_lerobot_style(cls, data: dict[str, Any]) -> SO101Calibration:
        # Feetech STS3215: 4096 ticks / 2π rad ≈ 651.9
        ticks_per_rad = 4096.0 / (2.0 * float(np.pi))
        half_turn = 2048

        items: list[dict[str, Any]] = []
        for name, cfg in data.items():
            if not isinstance(cfg, dict):
                continue
            try:
                mid = {
                    "name": name,
                    "id": int(cfg["id"]),
                    "homing_offset": int(cfg.get("homing_offset", 0)),
                    "range_min": int(cfg["range_min"]),
                    "range_max": int(cfg["range_max"]),
                }
            except Exception:
                continue
            items.append(mid)

        if len(items) != 6:
            raise ValueError("lerobot-style calibration must include exactly 6 motor entries")

        items.sort(key=lambda x: int(x["id"]))
        joints: list[JointCalibration] = []
        for it in items:
            motor_id = int(it["id"])
            # Present_Position = Actual_Position - Homing_Offset (lerobot docs). We read raw ticks
            # with normalize=False, so we fold the offset into our `zero_ticks` guess.
            zero_ticks = int(half_turn + int(it["homing_offset"]))
            soft_min_rad = float((int(it["range_min"]) - zero_ticks) / ticks_per_rad)
            soft_max_rad = float((int(it["range_max"]) - zero_ticks) / ticks_per_rad)
            joints.append(
                JointCalibration(
                    name=str(motor_id),
                    motor_id=motor_id,
                    zero_ticks=zero_ticks,
                    ticks_per_rad=float(ticks_per_rad),
                    soft_min_rad=soft_min_rad,
                    soft_max_rad=soft_max_rad,
                )
            )
        return cls(joints=tuple(joints))

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "joints": [
                {
                    "name": j.name,
                    "motor_id": j.motor_id,
                    "zero_ticks": j.zero_ticks,
                    "ticks_per_rad": j.ticks_per_rad,
                    "soft_min_rad": j.soft_min_rad,
                    "soft_max_rad": j.soft_max_rad,
                }
                for j in self.joints
            ]
        }
        if self.gripper_open_rad is not None:
            out["gripper_open_rad"] = self.gripper_open_rad
        if self.gripper_closed_rad is not None:
            out["gripper_closed_rad"] = self.gripper_closed_rad
        return out

    @classmethod
    def load(cls, path: Path) -> SO101Calibration:
        raw = Path(path).read_text(encoding="utf-8")
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("calibration file must contain a JSON object")
        return cls.from_dict(data)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def to_ticks(self, q_rad: np.ndarray) -> dict[str, int]:
        """Map length-6 rad vector to motor-name → raw tick (for ``sync_write``)."""
        q = np.asarray(q_rad, dtype=np.float64).reshape(-1)
        if q.shape[0] != 6:
            raise ValueError(f"expected 6 joint positions, got {q.shape[0]}")
        out: dict[str, int] = {}
        for i, jc in enumerate(self.joints):
            tick = int(round(jc.zero_ticks + float(q[i]) * jc.ticks_per_rad))
            out[jc.name] = tick
        return out

    def to_radians(self, ticks_by_name: dict[str, float | int]) -> np.ndarray:
        """Inverse map from Present_Position dict (motor name keys) to length-6 rad."""
        q = np.zeros(6, dtype=np.float64)
        for i, jc in enumerate(self.joints):
            if jc.name not in ticks_by_name:
                raise KeyError(f"missing tick for joint {jc.name!r}")
            tick = float(ticks_by_name[jc.name])
            q[i] = (tick - jc.zero_ticks) / jc.ticks_per_rad if jc.ticks_per_rad != 0 else 0.0
        return q

    @staticmethod
    def bundled_example_path() -> Path:
        return _MODULE_DIR / "calibration" / "example.json"

    @classmethod
    def locate(
        cls,
        *,
        explicit_path: str | Path | None = None,
        allow_template: bool = False,
    ) -> tuple[Path, SO101Calibration]:
        """Resolve calibration file path and load.

        Search order:
        1. ``explicit_path`` if set
        2. ``$ROBODEPLOY_SO101_CALIBRATION``
        3. ``~/.robodeploy/so101_calibration.json``
        4. bundled ``calibration/example.json`` (raises unless ``allow_template``)
        """
        candidates: list[Path] = []
        if explicit_path:
            candidates.append(Path(explicit_path).expanduser())
        envp = os.environ.get("ROBODEPLOY_SO101_CALIBRATION", "").strip()
        if envp:
            candidates.append(Path(envp).expanduser())
        candidates.append(Path.home() / ".robodeploy" / "so101_calibration.json")
        candidates.append(cls.bundled_example_path())

        chosen: Path | None = None
        for p in candidates:
            if p.is_file():
                chosen = p.resolve()
                break
        if chosen is None:
            raise MissingCalibrationError(
                "No SO-101 calibration file found. Set ROBODEPLOY_SO101_CALIBRATION or run:\n"
                "  python -m examples.so101.calibrate_so101 --port /dev/ttyACM0 --out ~/.robodeploy/so101_calibration.json"
            )

        example_resolved = cls.bundled_example_path().resolve()
        if chosen == example_resolved and not allow_template:
            raise MissingCalibrationError(
                "Refusing to use the bundled template calibration (unsafe on real hardware). "
                "Run `python -m examples.so101.calibrate_so101 ...` or set ROBODEPLOY_SO101_CALIBRATION "
                "to a user-generated JSON. For dry runs only, pass robot0.allow_uncalibrated=true in config."
            )

        return chosen, cls.load(chosen)


def _optional_float(x: Any) -> float | None:
    if x is None:
        return None
    return float(x)


def _looks_like_lerobot_calibration(data: dict[str, Any]) -> bool:
    if not data:
        return False
    # Heuristic: values are dicts with {id, range_min, range_max} keys.
    seen = 0
    for v in data.values():
        if not isinstance(v, dict):
            continue
        if {"id", "range_min", "range_max"}.issubset(set(v.keys())):
            seen += 1
    return seen >= 4
