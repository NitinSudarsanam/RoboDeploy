"""Multi-pose linear fit for raw ↔ canonical joint coordinates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_FORMAT = "robodeploy-linear-kinematic-v1"


@dataclass(frozen=True)
class JointLinearMap:
    """tick ≈ zero + canonical * scale (generalizes SO-101 per-joint model)."""

    name: str
    zero: float
    scale: float
    soft_min: float | None = None
    soft_max: float | None = None

    def to_canonical(self, raw: float) -> float:
        if self.scale == 0.0:
            return 0.0
        return (float(raw) - self.zero) / self.scale

    def to_raw(self, canonical: float) -> float:
        return self.zero + float(canonical) * self.scale


class LinearKinematicCalibration:
    """Vectorized linear map for N joints from paired calibration poses."""

    def __init__(self, joints: tuple[JointLinearMap, ...] | list[JointLinearMap]) -> None:
        self.joints = tuple(joints)

    def fit(self, raw_to_canonical_pairs: list[tuple[Any, Any]]) -> "LinearKinematicCalibration":
        if len(raw_to_canonical_pairs) < 2:
            raise ValueError("linear kinematic fit requires at least 2 pose pairs")
        raw_stack = np.stack([np.asarray(r, dtype=np.float64).reshape(-1) for r, _ in raw_to_canonical_pairs])
        can_stack = np.stack([np.asarray(c, dtype=np.float64).reshape(-1) for _, c in raw_to_canonical_pairs])
        if raw_stack.shape != can_stack.shape:
            raise ValueError("raw and canonical pose vectors must have matching shape")
        n_joints = raw_stack.shape[1]
        joints: list[JointLinearMap] = []
        for j in range(n_joints):
            raw_j = raw_stack[:, j]
            can_j = can_stack[:, j]
            if np.std(can_j) < 1e-9:
                scale = 1.0
                zero = float(raw_j.mean())
            else:
                scale, zero = np.polyfit(can_j, raw_j, 1)
            joints.append(JointLinearMap(name=f"joint_{j}", zero=float(zero), scale=float(scale)))
        return LinearKinematicCalibration(joints)

    def to_canonical(self, raw_value: Any) -> np.ndarray:
        raw = np.asarray(raw_value, dtype=np.float64).reshape(-1)
        if raw.shape[0] != len(self.joints):
            raise ValueError(f"expected {len(self.joints)} raw values, got {raw.shape[0]}")
        return np.asarray([j.to_canonical(raw[i]) for i, j in enumerate(self.joints)], dtype=np.float64)

    def to_raw(self, canonical_value: Any) -> np.ndarray:
        can = np.asarray(canonical_value, dtype=np.float64).reshape(-1)
        if can.shape[0] != len(self.joints):
            raise ValueError(f"expected {len(self.joints)} canonical values, got {can.shape[0]}")
        return np.asarray([j.to_raw(can[i]) for i, j in enumerate(self.joints)], dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": _FORMAT,
            "joints": [
                {
                    "name": j.name,
                    "zero": j.zero,
                    "scale": j.scale,
                    "soft_min": j.soft_min,
                    "soft_max": j.soft_max,
                }
                for j in self.joints
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LinearKinematicCalibration":
        joints_in = data.get("joints") or []
        joints = [
            JointLinearMap(
                name=str(item["name"]),
                zero=float(item["zero"]),
                scale=float(item["scale"]),
                soft_min=_optional_float(item.get("soft_min")),
                soft_max=_optional_float(item.get("soft_max")),
            )
            for item in joints_in
        ]
        return cls(joints)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "LinearKinematicCalibration":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("calibration file must be a JSON object")
        return cls.from_dict(data)

    def fit_residuals(self, raw_to_canonical_pairs: list[tuple[Any, Any]]) -> np.ndarray:
        errs = []
        for raw, can in raw_to_canonical_pairs:
            pred = self.to_canonical(raw)
            errs.append(np.linalg.norm(pred - np.asarray(can, dtype=np.float64).reshape(-1)))
        return np.asarray(errs, dtype=np.float64)


def _optional_float(x: Any) -> float | None:
    if x is None:
        return None
    return float(x)
