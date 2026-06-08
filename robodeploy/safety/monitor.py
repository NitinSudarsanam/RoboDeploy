"""SafetyMonitor — aggregates guards into one env-facing entry point."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from robodeploy.core.types import Action, Observation

from .estop import EStop, EStopGuard
from .violation import Hazard, SafetyError, Severity, ViolationRecord


@runtime_checkable
class ISafetyGuard(Protocol):
    def check_action(
        self,
        action: Action,
        obs: Observation,
        *,
        dt: float,
    ) -> tuple[Action, list[ViolationRecord]]: ...

    def check_observation(self, obs: Observation) -> list[ViolationRecord]: ...


@dataclass
class SafetyStatus:
    tripped: bool
    active_violations: list[ViolationRecord]
    history_count: int
    last_violation: ViolationRecord | None


class SafetyMonitor:
    """Aggregates guards and e-stop into pre/post step validation."""

    def __init__(
        self,
        *,
        guards: list[ISafetyGuard] | None = None,
        estop: EStop | None = None,
        on_violation: Literal["clamp", "halt", "raise"] = "halt",
        on_critical: Literal["halt", "raise"] = "raise",
        violation_log: Path | None = None,
    ) -> None:
        self._guards: list[ISafetyGuard] = list(guards or [])
        self._estop = estop or EStop(signal_handlers=False)
        if not any(isinstance(g, EStopGuard) for g in self._guards):
            self._guards.append(EStopGuard(self._estop))
        self._mode = on_violation
        self._critical_mode = on_critical
        self._violations: list[ViolationRecord] = []
        self._tripped = False
        self._log = violation_log

    @property
    def estop(self) -> EStop:
        return self._estop

    def add_guard(self, guard: ISafetyGuard) -> None:
        self._guards.append(guard)

    def check_action(
        self,
        action: Action,
        obs: Observation,
        *,
        dt: float,
        robot_id: str | None = None,
        ignore_slew: bool = False,
    ) -> Action:
        if self._tripped:
            raise SafetyError(self._violations[-1] if self._violations else "safety tripped")
        try:
            self._estop.check()
        except SafetyError as exc:
            self._tripped = True
            self._violations.append(exc.violation)
            raise
        from robodeploy.safety.filter import SafetyFilterGuard

        for guard in self._guards_for(robot_id):
            if isinstance(guard, SafetyFilterGuard):
                action, violations = guard.check_action(
                    action, obs, dt=dt, ignore_slew=ignore_slew
                )
            else:
                action, violations = guard.check_action(action, obs, dt=dt)
            self._handle(violations)
        return action

    def check_observation(self, obs: Observation, *, robot_id: str | None = None) -> None:
        if self._tripped:
            raise SafetyError(self._violations[-1] if self._violations else "safety tripped")
        try:
            self._estop.check()
        except SafetyError as exc:
            self._tripped = True
            self._violations.append(exc.violation)
            raise
        for guard in self._guards_for(robot_id):
            violations = guard.check_observation(obs)
            self._handle(violations)

    @property
    def tripped(self) -> bool:
        return self._tripped

    def reset(self) -> None:
        self._tripped = False
        self._violations.clear()
        self._estop.reset()

    def halt(self, reason: str, *, hazard: Hazard = Hazard.PROGRAMMATIC_HALT) -> None:
        """Mark the monitor tripped without waiting for the next guard check."""
        self._tripped = True
        self._violations.append(
            ViolationRecord(
                hazard=hazard,
                severity=Severity.CRITICAL,
                message=str(reason),
            )
        )

    def violations(self) -> list[ViolationRecord]:
        return list(self._violations)

    def status(self) -> SafetyStatus:
        active = [v for v in self._violations if v.severity >= Severity.WARNING]
        last = self._violations[-1] if self._violations else None
        return SafetyStatus(
            tripped=self._tripped,
            active_violations=active[-8:],
            history_count=len(self._violations),
            last_violation=last,
        )

    def _guards_for(self, robot_id: str | None) -> list[ISafetyGuard]:
        if robot_id is None:
            return self._guards
        return [
            guard
            for guard in self._guards
            if getattr(guard, "robot_id", None) in (None, robot_id)
        ]

    def _handle(self, violations: list[ViolationRecord]) -> None:
        for violation in violations:
            self._violations.append(violation)
            self._append_log(violation)
            if violation.severity >= Severity.CRITICAL:
                self._tripped = True
                if self._critical_mode == "raise":
                    raise SafetyError(violation)
            elif violation.severity >= Severity.WARNING and self._mode == "raise":
                self._tripped = True
                raise SafetyError(violation)

    def _append_log(self, violation: ViolationRecord) -> None:
        if self._log is None:
            return
        try:
            payload = {
                "hazard": violation.hazard.name,
                "severity": violation.severity.name,
                "message": violation.message,
                "value": violation.value,
                "limit": violation.limit,
                "joint_idx": violation.joint_idx,
                "sensor_name": violation.sensor_name,
                "timestamp": violation.timestamp,
            }
            with self._log.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload) + "\n")
        except Exception:
            pass
