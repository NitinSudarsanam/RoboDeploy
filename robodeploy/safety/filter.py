"""SafetyFilter guard — wraps kinematics SafetyFilter as an ISafetyGuard."""

from __future__ import annotations

import numpy as np

from robodeploy.core.spaces import ActionSpace
from robodeploy.core.types import Action, Observation
from robodeploy.kinematics.safety import SafetyFilter

from .violation import Hazard, Severity, ViolationRecord


class SafetyFilterGuard:
    """Applies joint-limit clamping via the shared SafetyFilter."""

    def __init__(
        self,
        *,
        safety_filter: SafetyFilter,
        action_space: ActionSpace = ActionSpace.JOINT_POS,
        robot_id: str | None = None,
    ) -> None:
        self._filter = safety_filter
        self._action_space = action_space
        self.robot_id = robot_id

    def check_action(
        self,
        action: Action,
        obs: Observation,
        *,
        dt: float,
        ignore_slew: bool = False,
    ) -> tuple[Action, list[ViolationRecord]]:
        del dt
        violations: list[ViolationRecord] = []
        if self._filter.estop_active:
            violations.append(
                ViolationRecord(
                    hazard=Hazard.OPERATOR_ESTOP,
                    severity=Severity.CRITICAL,
                    message="SafetyFilter e-stop active",
                )
            )
            return self._filter.filter(action, self._action_space), violations

        raw_pos = (
            np.asarray(action.joint_positions, dtype=np.float64)
            if action.joint_positions is not None
            else None
        )
        filtered = self._filter.filter(
            action, self._action_space, obs=obs, ignore_slew=ignore_slew
        )

        if raw_pos is not None and filtered.joint_positions is not None:
            clamped = np.asarray(filtered.joint_positions, dtype=np.float64)
            if np.any(np.abs(raw_pos - clamped) > 1e-6):
                for idx in range(raw_pos.shape[0]):
                    if abs(raw_pos[idx] - clamped[idx]) > 1e-6:
                        violations.append(
                            ViolationRecord(
                                hazard=Hazard.JOINT_POSITION_LIMIT,
                                severity=Severity.WARNING,
                                message=f"joint {idx} clamped {raw_pos[idx]:.4f} -> {clamped[idx]:.4f}",
                                value=float(raw_pos[idx]),
                                limit=float(clamped[idx]),
                                joint_idx=idx,
                            )
                        )
        return filtered, violations

    def check_observation(self, obs: Observation) -> list[ViolationRecord]:
        return []
