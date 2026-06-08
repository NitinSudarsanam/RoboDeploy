"""Motor encoder bus ↔ radians calibration (SO-101 style)."""

from __future__ import annotations

from typing import Any

import numpy as np

from robodeploy.calibration.kinematic.linear import JointLinearMap, LinearKinematicCalibration


class MotorBusCalibration(LinearKinematicCalibration):
    """Named joints with motor IDs; wraps linear tick ↔ radian maps."""

    def __init__(
        self,
        joints: tuple[JointLinearMap, ...] | list[JointLinearMap],
        *,
        motor_ids: tuple[int, ...] | None = None,
        gripper_open_rad: float | None = None,
        gripper_closed_rad: float | None = None,
    ) -> None:
        super().__init__(joints)
        self.motor_ids = motor_ids
        self.gripper_open_rad = gripper_open_rad
        self.gripper_closed_rad = gripper_closed_rad

    def to_ticks_by_name(self, q_rad: np.ndarray) -> dict[str, int]:
        q = np.asarray(q_rad, dtype=np.float64).reshape(-1)
        raw = self.to_raw(q)
        return {j.name: int(round(raw[i])) for i, j in enumerate(self.joints)}

    def to_radians_by_name(self, ticks_by_name: dict[str, float | int]) -> np.ndarray:
        raw = np.zeros(len(self.joints), dtype=np.float64)
        for i, j in enumerate(self.joints):
            if j.name not in ticks_by_name:
                raise KeyError(f"missing tick for joint {j.name!r}")
            raw[i] = float(ticks_by_name[j.name])
        return self.to_canonical(raw)

    def to_dict(self) -> dict[str, Any]:
        out = super().to_dict()
        if self.motor_ids is not None:
            out["motor_ids"] = list(self.motor_ids)
        if self.gripper_open_rad is not None:
            out["gripper_open_rad"] = self.gripper_open_rad
        if self.gripper_closed_rad is not None:
            out["gripper_closed_rad"] = self.gripper_closed_rad
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MotorBusCalibration":
        base = LinearKinematicCalibration.from_dict(data)
        motor_ids = data.get("motor_ids")
        return cls(
            base.joints,
            motor_ids=tuple(int(x) for x in motor_ids) if motor_ids else None,
            gripper_open_rad=data.get("gripper_open_rad"),
            gripper_closed_rad=data.get("gripper_closed_rad"),
        )
