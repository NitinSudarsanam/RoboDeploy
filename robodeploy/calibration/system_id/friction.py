"""Coulomb + viscous joint friction estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from robodeploy.env import RoboEnv


@dataclass
class FrictionParams:
    coulomb_Nm: float
    viscous_Nm_per_rad_s: float
    joint_idx: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "coulomb_Nm": self.coulomb_Nm,
            "viscous_Nm_per_rad_s": self.viscous_Nm_per_rad_s,
            "joint_idx": self.joint_idx,
        }


@dataclass
class FrictionSample:
    velocity_rad_s: float
    torque_Nm: float


class FrictionEstimator:
    """Drive joint at constant velocity; fit steady-state torque."""

    def __init__(self, *, joint_range_guard: tuple[float, float] = (-2.5, 2.5)) -> None:
        self.joint_range_guard = (float(joint_range_guard[0]), float(joint_range_guard[1]))

    def collect_data(
        self,
        env: "RoboEnv",
        joint_idx: int,
        *,
        velocities: list[float],
        steady_state_steps: int = 20,
        max_steps_per_velocity: int = 200,
    ) -> list[FrictionSample]:
        samples: list[FrictionSample] = []
        for vel in velocities:
            torques: list[float] = []
            for _ in range(int(max_steps_per_velocity)):
                obs_by_robot = env.get_processed_obs_by_robot()
                primary = env.primary_robot
                obs = obs_by_robot[primary.robot_id]
                q = float(np.asarray(obs.joint_positions, dtype=np.float64).reshape(-1)[joint_idx])
                if q < self.joint_range_guard[0] or q > self.joint_range_guard[1]:
                    break
                from robodeploy.core.types import Action

                n = len(obs.joint_positions)
                cmd = np.asarray(obs.joint_positions, dtype=np.float32).copy()
                cmd[joint_idx] += float(vel) * 0.01
                env.step(Action(joint_positions=cmd))
                if obs.joint_torques is not None:
                    t = float(np.asarray(obs.joint_torques, dtype=np.float64).reshape(-1)[joint_idx])
                    torques.append(t)
                if len(torques) >= int(steady_state_steps):
                    break
            if torques:
                samples.append(FrictionSample(velocity_rad_s=float(vel), torque_Nm=float(np.mean(torques[-steady_state_steps:]))))
        return samples

    def fit(self, samples: list[FrictionSample]) -> FrictionParams:
        if len(samples) < 2:
            raise ValueError("friction fit requires at least 2 velocity samples")
        v = np.asarray([s.velocity_rad_s for s in samples], dtype=np.float64)
        t = np.asarray([s.torque_Nm for s in samples], dtype=np.float64)
        # torque ≈ coulomb * sign(v) + viscous * v — linearize with sign
        A = np.stack([np.sign(v), v], axis=1)
        coeffs, _, _, _ = np.linalg.lstsq(A, t, rcond=None)
        return FrictionParams(
            coulomb_Nm=float(coeffs[0]),
            viscous_Nm_per_rad_s=float(coeffs[1]),
            joint_idx=0,
        )
