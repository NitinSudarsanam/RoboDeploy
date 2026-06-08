"""Singularity guard — detects near-limit joints with high commanded velocity."""

from __future__ import annotations

import numpy as np

from robodeploy.core.types import Action, Observation

from .violation import Hazard, Severity, ViolationRecord


class SingularityGuard:
    """Flags configurations where a joint is near its limit and moving quickly."""

    def __init__(
        self,
        *,
        joint_position_limits: np.ndarray,
        margin_rad: float = 0.15,
        velocity_threshold_rad_s: float = 1.5,
        over_limit_strikes: int = 2,
    ) -> None:
        limits = np.asarray(joint_position_limits, dtype=np.float64)
        if limits.ndim != 2 or limits.shape[1] != 2:
            raise ValueError("joint_position_limits must have shape [dof, 2]")
        self._lower = limits[:, 0]
        self._upper = limits[:, 1]
        self._margin = float(margin_rad)
        self._vel_threshold = float(velocity_threshold_rad_s)
        self._over_limit_strikes = max(int(over_limit_strikes), 1)
        self._strikes = 0

    def check_action(
        self,
        action: Action,
        obs: Observation,
        *,
        dt: float,
    ) -> tuple[Action, list[ViolationRecord]]:
        del dt
        violations: list[ViolationRecord] = []
        q = np.asarray(obs.joint_positions, dtype=np.float64).reshape(-1)
        if action.joint_velocities is not None:
            dq = np.asarray(action.joint_velocities, dtype=np.float64).reshape(-1)
        elif action.joint_positions is not None and dt > 0:
            target = np.asarray(action.joint_positions, dtype=np.float64).reshape(-1)
            dq = (target - q) / float(dt)
        else:
            return action, violations

        for idx, (pos, rate, lo, hi) in enumerate(zip(q, dq, self._lower, self._upper)):
            near_limit = pos <= lo + self._margin or pos >= hi - self._margin
            if near_limit and abs(rate) > self._vel_threshold:
                self._strikes += 1
                severity = (
                    Severity.CRITICAL
                    if self._strikes >= self._over_limit_strikes
                    else Severity.WARNING
                )
                violations.append(
                    ViolationRecord(
                        hazard=Hazard.SINGULARITY_IMMINENT,
                        severity=severity,
                        message=(
                            f"joint {idx} near limit ({pos:.3f} rad) with |dq|={abs(rate):.3f} rad/s"
                        ),
                        value=float(abs(rate)),
                        limit=self._vel_threshold,
                        joint_idx=idx,
                    )
                )
                return action, violations

        self._strikes = max(0, self._strikes - 1)
        return action, violations

    def check_observation(self, obs: Observation) -> list[ViolationRecord]:
        del obs
        return []
