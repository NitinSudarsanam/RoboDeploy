"""End-effector workspace bounding box guard."""

from __future__ import annotations

import numpy as np

from robodeploy.core.types import Action, Observation

from .violation import Hazard, Severity, ViolationRecord


class WorkspaceGuard:
    """Clamps Cartesian targets to a workspace axis-aligned box."""

    def __init__(
        self,
        *,
        low_xyz: np.ndarray,
        high_xyz: np.ndarray,
        on_violation: str = "clamp",
    ) -> None:
        self._low = np.asarray(low_xyz, dtype=np.float64).reshape(3)
        self._high = np.asarray(high_xyz, dtype=np.float64).reshape(3)
        self._mode = str(on_violation)

    def check_action(
        self,
        action: Action,
        obs: Observation,
        *,
        dt: float,
    ) -> tuple[Action, list[ViolationRecord]]:
        del obs, dt
        violations: list[ViolationRecord] = []
        if action.ee_position is None:
            return action, violations

        raw = np.asarray(action.ee_position, dtype=np.float64).reshape(3)
        clamped = np.clip(raw, self._low, self._high)
        if np.any(raw != clamped):
            violations.append(
                ViolationRecord(
                    hazard=Hazard.EE_WORKSPACE_LIMIT,
                    severity=Severity.WARNING,
                    message="EE position clamped to workspace box",
                    value=float(np.linalg.norm(raw - clamped)),
                )
            )
        if self._mode == "raise" and violations:
            return action, violations

        return (
            Action(
                joint_positions=action.joint_positions,
                joint_velocities=action.joint_velocities,
                joint_torques=action.joint_torques,
                ee_position=clamped.astype(np.float32),
                ee_orientation=action.ee_orientation,
                ee_velocity=action.ee_velocity,
                gripper=action.gripper,
                timestamp=action.timestamp,
            ),
            violations,
        )

    def check_observation(self, obs: Observation) -> list[ViolationRecord]:
        if obs.ee_position is None:
            return []
        pos = np.asarray(obs.ee_position, dtype=np.float64).reshape(3)
        violations: list[ViolationRecord] = []
        for axis, name in enumerate("xyz"):
            if pos[axis] < self._low[axis] or pos[axis] > self._high[axis]:
                violations.append(
                    ViolationRecord(
                        hazard=Hazard.EE_WORKSPACE_LIMIT,
                        severity=Severity.WARNING,
                        message=f"EE {name}={pos[axis]:.4f} outside workspace",
                        value=float(pos[axis]),
                        limit=float(self._high[axis] if pos[axis] > self._high[axis] else self._low[axis]),
                    )
                )
        return violations
