"""Force/torque limit guard using FT sensor observations."""

from __future__ import annotations

import numpy as np

from robodeploy.core.types import Action, Observation

from .violation import Hazard, Severity, ViolationRecord


class ForceLimitGuard:
    """Halts after repeated force/torque excursions above configured limits."""

    def __init__(
        self,
        *,
        max_force_N: float = 50.0,
        max_torque_Nm: float = 10.0,
        over_limit_strikes: int = 3,
        sensor_name: str = "wrist_ft",
        robot_id: str | None = None,
    ) -> None:
        self.robot_id = robot_id
        self._max_force_N = float(max_force_N)
        self._max_torque_Nm = float(max_torque_Nm)
        self._over_limit_strikes = max(int(over_limit_strikes), 1)
        self._sensor_name = sensor_name
        self._force_strikes = 0
        self._torque_strikes = 0

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

        if obs.ft_force is not None:
            f = float(np.linalg.norm(np.asarray(obs.ft_force, dtype=np.float64)))
            if f > self._max_force_N:
                self._force_strikes += 1
                severity = (
                    Severity.CRITICAL
                    if self._force_strikes >= self._over_limit_strikes
                    else Severity.WARNING
                )
                violations.append(
                    ViolationRecord(
                        hazard=Hazard.FORCE_LIMIT,
                        severity=severity,
                        message=f"|F|={f:.1f}N > {self._max_force_N:.1f}N",
                        value=f,
                        limit=self._max_force_N,
                        sensor_name=self._sensor_name,
                    )
                )
            else:
                self._force_strikes = max(0, self._force_strikes - 1)

        if obs.ft_torque is not None:
            t = float(np.linalg.norm(np.asarray(obs.ft_torque, dtype=np.float64)))
            if t > self._max_torque_Nm:
                self._torque_strikes += 1
                severity = (
                    Severity.CRITICAL
                    if self._torque_strikes >= self._over_limit_strikes
                    else Severity.WARNING
                )
                violations.append(
                    ViolationRecord(
                        hazard=Hazard.TORQUE_LIMIT,
                        severity=severity,
                        message=f"|T|={t:.2f}Nm > {self._max_torque_Nm:.2f}Nm",
                        value=t,
                        limit=self._max_torque_Nm,
                        sensor_name=self._sensor_name,
                    )
                )
            else:
                self._torque_strikes = max(0, self._torque_strikes - 1)

        return violations
