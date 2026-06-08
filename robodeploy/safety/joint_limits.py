"""Joint position and finite-difference velocity guards."""

from __future__ import annotations

import numpy as np

from robodeploy.core.types import Action, Observation

from .violation import Hazard, Severity, ViolationRecord


class JointLimitGuard:
    """Position and finite-difference velocity checks on observations."""

    def __init__(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
        vel_max: np.ndarray,
    ) -> None:
        self._lower = np.asarray(lower, dtype=np.float64).reshape(-1)
        self._upper = np.asarray(upper, dtype=np.float64).reshape(-1)
        self._vel_max = np.asarray(vel_max, dtype=np.float64).reshape(-1)
        self._q_prev: np.ndarray | None = None

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
        qv = np.asarray(obs.joint_positions, dtype=np.float64).reshape(-1)
        if qv.shape[0] != self._lower.shape[0]:
            return [
                ViolationRecord(
                    hazard=Hazard.JOINT_POSITION_LIMIT,
                    severity=Severity.CRITICAL,
                    message=f"joint limit guard dof mismatch: {qv.shape[0]} vs {self._lower.shape[0]}",
                )
            ]

        violations: list[ViolationRecord] = []
        for idx, (q, lo, hi) in enumerate(zip(qv, self._lower, self._upper)):
            if q < lo or q > hi:
                violations.append(
                    ViolationRecord(
                        hazard=Hazard.JOINT_POSITION_LIMIT,
                        severity=Severity.CRITICAL,
                        message=f"joint {idx} position {q:.4f} outside [{lo:.4f}, {hi:.4f}]",
                        value=float(q),
                        limit=float(hi if q > hi else lo),
                        joint_idx=idx,
                    )
                )

        if self._q_prev is not None:
            dt = max(float(obs.timestamp) - getattr(self, "_last_ts", obs.timestamp), 1e-6)
            dq = (qv - self._q_prev) / dt
            for idx, (rate, limit) in enumerate(zip(dq, self._vel_max)):
                if abs(rate) > float(limit) + 1e-6:
                    violations.append(
                        ViolationRecord(
                            hazard=Hazard.JOINT_VELOCITY_LIMIT,
                            severity=Severity.CRITICAL,
                            message=f"joint {idx} velocity {rate:.4f} > {limit:.4f} rad/s",
                            value=float(abs(rate)),
                            limit=float(limit),
                            joint_idx=idx,
                        )
                    )

        self._q_prev = qv.copy()
        self._last_ts = float(obs.timestamp)
        return violations

    def check(self, q: np.ndarray, *, dt: float | None) -> None:
        """ROS2 controller compatibility — raises SafetyError on violation."""
        from .violation import SafetyError

        qv = np.asarray(q, dtype=np.float64).reshape(-1)
        if qv.shape[0] != self._lower.shape[0]:
            raise SafetyError(f"joint limit guard dof mismatch: {qv.shape[0]} vs {self._lower.shape[0]}")
        if np.any(qv < self._lower) or np.any(qv > self._upper):
            raise SafetyError("Joint position outside soft limits.")
        if dt is not None and float(dt) > 1e-9 and self._q_prev is not None:
            dq = (qv - self._q_prev) / float(dt)
            if np.any(np.abs(dq) > self._vel_max + 1e-6):
                raise SafetyError("Joint velocity exceeded limit (finite-difference check).")
        self._q_prev = qv.copy()
