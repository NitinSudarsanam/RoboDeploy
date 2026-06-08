"""Calibration protocol interfaces and shared types."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from robodeploy.core.types import Pose3D

SCHEMA_VERSION = "robodeploy-calibration-v1"


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole camera intrinsics (metres / pixels)."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int = 640
    height: int = 480
    dist_coeffs: tuple[float, ...] = ()

    def to_matrix(self) -> list[list[float]]:
        return [
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0],
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "width": self.width,
            "height": self.height,
            "dist_coeffs": list(self.dist_coeffs),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CameraIntrinsics":
        coeffs = data.get("dist_coeffs") or []
        return cls(
            fx=float(data["fx"]),
            fy=float(data["fy"]),
            cx=float(data["cx"]),
            cy=float(data["cy"]),
            width=int(data.get("width", 640)),
            height=int(data.get("height", 480)),
            dist_coeffs=tuple(float(c) for c in coeffs),
        )


@runtime_checkable
class IKinematicCalibration(Protocol):
    """Map raw actuator readings ↔ canonical joint coordinates."""

    def fit(self, raw_to_canonical_pairs: list[tuple[Any, Any]]) -> "IKinematicCalibration": ...

    def to_canonical(self, raw_value: Any) -> Any: ...

    def to_raw(self, canonical_value: Any) -> Any: ...

    def save(self, path: str | Path) -> None: ...

    @classmethod
    def load(cls, path: str | Path) -> "IKinematicCalibration": ...


@runtime_checkable
class IExtrinsicCalibration(Protocol):
    """Camera / sensor extrinsic calibration."""

    def fit(self, observations: list[Pose3D]) -> Pose3D: ...

    def fit_handeye(
        self,
        robot_poses: list[Pose3D],
        marker_poses: list[Pose3D],
    ) -> Pose3D: ...
