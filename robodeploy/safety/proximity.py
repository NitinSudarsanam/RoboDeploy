"""Human proximity guard — halts when the EE is too close to people in the workspace."""

from __future__ import annotations

import numpy as np

from robodeploy.core.types import Action, Observation

from .violation import Hazard, Severity, ViolationRecord

_DEFAULT_HUMAN_KEYS = ("human", "operator", "person", "hand")


class HumanProximityGuard:
    """Observation guard using object poses or ``obs.metadata['proximity_m']``."""

    def __init__(
        self,
        *,
        min_distance_m: float = 0.25,
        human_object_keys: tuple[str, ...] = _DEFAULT_HUMAN_KEYS,
        over_limit_strikes: int = 1,
    ) -> None:
        self._min_distance_m = float(min_distance_m)
        self._human_keys = tuple(str(k).lower() for k in human_object_keys)
        self._over_limit_strikes = max(int(over_limit_strikes), 1)
        self._strikes = 0

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
        proximity = None
        meta = getattr(obs, "metadata", None)
        if isinstance(meta, dict):
            raw = meta.get("proximity_m")
            if raw is not None:
                proximity = float(raw)

        if proximity is not None and proximity < self._min_distance_m:
            self._strikes += 1
            severity = (
                Severity.CRITICAL
                if self._strikes >= self._over_limit_strikes
                else Severity.WARNING
            )
            violations.append(
                ViolationRecord(
                    hazard=Hazard.HUMAN_PROXIMITY,
                    severity=severity,
                    message=(
                        f"proximity sensor {proximity:.3f}m < {self._min_distance_m:.3f}m minimum"
                    ),
                    value=proximity,
                    limit=self._min_distance_m,
                    sensor_name="proximity_m",
                )
            )
            return violations

        if obs.ee_position is None or not obs.objects:
            self._strikes = max(0, self._strikes - 1)
            return violations

        ee = np.asarray(obs.ee_position, dtype=np.float64).reshape(3)
        min_dist = float("inf")
        nearest = ""
        for name, pose in obs.objects.items():
            key = str(name).lower()
            if not any(token in key for token in self._human_keys):
                continue
            pos = np.asarray(pose[0], dtype=np.float64).reshape(3)
            dist = float(np.linalg.norm(ee - pos))
            if dist < min_dist:
                min_dist = dist
                nearest = str(name)

        if min_dist < self._min_distance_m:
            self._strikes += 1
            severity = (
                Severity.CRITICAL
                if self._strikes >= self._over_limit_strikes
                else Severity.WARNING
            )
            violations.append(
                ViolationRecord(
                    hazard=Hazard.HUMAN_PROXIMITY,
                    severity=severity,
                    message=(
                        f"EE {min_dist:.3f}m from {nearest!r} < {self._min_distance_m:.3f}m minimum"
                    ),
                    value=min_dist,
                    limit=self._min_distance_m,
                    sensor_name=nearest or None,
                )
            )
        else:
            self._strikes = max(0, self._strikes - 1)
        return violations
