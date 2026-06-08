"""Payload mass estimation from gravity-loaded joint torque."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from robodeploy.env import RoboEnv


class PayloadMassEstimator:
    """Estimate payload mass from static pose joint torque (m*g*r_cm)."""

    def __init__(self, *, gravity: float = 9.81, arm_length_m: float = 0.3) -> None:
        self.gravity = float(gravity)
        self.arm_length_m = float(arm_length_m)

    def estimate(
        self,
        env: "RoboEnv",
        *,
        joint_idx: int = -2,
        pose_test_q: np.ndarray | None = None,
        steady_steps: int = 10,
    ) -> float:
        from robodeploy.core.types import Action

        if pose_test_q is not None:
            env.step(Action(joint_positions=np.asarray(pose_test_q, dtype=np.float32)))
        torques: list[float] = []
        for _ in range(int(steady_steps)):
            obs_by_robot = env.get_processed_obs_by_robot()
            obs = obs_by_robot[env.primary_robot.robot_id]
            env.step(None)
            if obs.joint_torques is None:
                continue
            t = float(np.asarray(obs.joint_torques, dtype=np.float64).reshape(-1)[joint_idx])
            torques.append(t)
        if not torques:
            return 0.0
        tau = float(np.mean(torques))
        if abs(self.gravity * self.arm_length_m) < 1e-9:
            return 0.0
        return max(0.0, tau / (self.gravity * self.arm_length_m))
