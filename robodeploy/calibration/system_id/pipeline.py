"""System-ID orchestrator — friction + mass + DH in one pass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from robodeploy.calibration.store import CalibrationStore
from robodeploy.calibration.system_id.friction import FrictionEstimator, FrictionParams
from robodeploy.calibration.system_id.mass import PayloadMassEstimator

if TYPE_CHECKING:
    from robodeploy.env import RoboEnv


@dataclass
class SystemIdResult:
    friction: dict[str, FrictionParams]
    payload_mass_kg: float
    dh: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "friction": {k: v.to_dict() for k, v in self.friction.items()},
            "payload_mass_kg": self.payload_mass_kg,
            "dh": self.dh,
        }


class SystemIdPipeline:
    """Run friction sweep + payload mass estimate and persist to CalibrationStore."""

    def __init__(
        self,
        *,
        store: CalibrationStore | None = None,
        friction_velocities: list[float] | None = None,
    ) -> None:
        self.store = store or CalibrationStore()
        self.friction = FrictionEstimator()
        self.mass = PayloadMassEstimator()
        self.friction_velocities = friction_velocities or [0.05, 0.1, 0.2, -0.05, -0.1]

    def run(
        self,
        env: "RoboEnv",
        *,
        joint_indices: list[int] | None = None,
        robot_id: str = "default",
        artifact_name: str = "system_id",
    ) -> SystemIdResult:
        joints = joint_indices or [0]
        friction_out: dict[str, FrictionParams] = {}
        for jidx in joints:
            samples = self.friction.collect_data(
                env,
                jidx,
                velocities=self.friction_velocities,
                steady_state_steps=5,
                max_steps_per_velocity=30,
            )
            if len(samples) >= 2:
                params = self.friction.fit(samples)
                params = FrictionParams(
                    coulomb_Nm=params.coulomb_Nm,
                    viscous_Nm_per_rad_s=params.viscous_Nm_per_rad_s,
                    joint_idx=jidx,
                )
                friction_out[f"joint_{jidx}"] = params
        mass_kg = self.mass.estimate(env, joint_idx=joints[0] if joints else -2)
        result = SystemIdResult(friction=friction_out, payload_mass_kg=mass_kg)
        self.store.save(artifact_name, result.to_dict(), robot_id=robot_id)
        return result
