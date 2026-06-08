"""Joint and end-effector velocity guards."""

from __future__ import annotations

import numpy as np

from robodeploy.core.types import Action, Observation

from .violation import Hazard, Severity, ViolationRecord


class VelocityGuard:
    """Checks reported joint and EE velocities against configured limits."""

    def __init__(
        self,
        *,
        max_joint_velocity: np.ndarray | None = None,
        max_ee_velocity_mps: float = 0.5,
        robot_id: str | None = None,
    ) -> None:
        self.robot_id = robot_id
        self._max_joint_velocity = (
            np.asarray(max_joint_velocity, dtype=np.float64).reshape(-1)
            if max_joint_velocity is not None
            else None
        )
        self._max_ee_velocity_mps = float(max_ee_velocity_mps)

    def check_action(
        self,
        action: Action,
        obs: Observation,
        *,
        dt: float,
    ) -> tuple[Action, list[ViolationRecord]]:
        del obs, dt
        return action, []

    def check_observation(self, obs: Observation) -> list[ViolationRecord]:
        violations: list[ViolationRecord] = []

        if self._max_joint_velocity is not None:
            qd = np.asarray(obs.joint_velocities, dtype=np.float64).reshape(-1)
            n = min(qd.shape[0], self._max_joint_velocity.shape[0])
            for idx in range(n):
                limit = float(self._max_joint_velocity[idx])
                val = abs(float(qd[idx]))
                if val > limit + 1e-6:
                    violations.append(
                        ViolationRecord(
                            hazard=Hazard.JOINT_VELOCITY_LIMIT,
                            severity=Severity.WARNING,
                            message=f"joint {idx} velocity {val:.4f} > {limit:.4f} rad/s",
                            value=val,
                            limit=limit,
                            joint_idx=idx,
                        )
                    )

        if obs.ee_velocity is not None:
            ee_v = float(np.linalg.norm(np.asarray(obs.ee_velocity, dtype=np.float64)))
            if ee_v > self._max_ee_velocity_mps + 1e-6:
                violations.append(
                    ViolationRecord(
                        hazard=Hazard.EE_VELOCITY_LIMIT,
                        severity=Severity.WARNING,
                        message=f"EE velocity {ee_v:.4f} > {self._max_ee_velocity_mps:.4f} m/s",
                        value=ee_v,
                        limit=self._max_ee_velocity_mps,
                    )
                )

        return violations
