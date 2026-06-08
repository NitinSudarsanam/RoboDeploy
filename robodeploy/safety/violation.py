"""Structured safety hazards, violations, and errors."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import IntEnum


class Hazard(IntEnum):
    JOINT_POSITION_LIMIT = 1
    JOINT_VELOCITY_LIMIT = 2
    JOINT_ACCELERATION_LIMIT = 3
    EE_WORKSPACE_LIMIT = 4
    EE_VELOCITY_LIMIT = 5
    FORCE_LIMIT = 6
    TORQUE_LIMIT = 7
    TEMPERATURE_HIGH = 8
    COMMAND_TIMEOUT = 9
    STATE_TIMEOUT = 10
    CONNECTION_LOST = 11
    COMMAND_REJECTED = 12
    COLLISION_IMMINENT = 13
    OPERATOR_ESTOP = 14
    PROGRAMMATIC_HALT = 15
    HUMAN_PROXIMITY = 16
    SINGULARITY_IMMINENT = 17


class Severity(IntEnum):
    INFO = 1
    WARNING = 2
    CRITICAL = 3


@dataclass
class ViolationRecord:
    hazard: Hazard
    severity: Severity
    message: str
    value: float | None = None
    limit: float | None = None
    joint_idx: int | None = None
    sensor_name: str | None = None
    timestamp: float = field(default_factory=time.time)


class SafetyError(RuntimeError):
    """Raised when a safety guard trips and the monitor is configured to halt."""

    def __init__(self, violation: ViolationRecord | str) -> None:
        if isinstance(violation, str):
            violation = ViolationRecord(
                hazard=Hazard.PROGRAMMATIC_HALT,
                severity=Severity.CRITICAL,
                message=violation,
            )
        self.violation = violation
        super().__init__(f"{violation.hazard.name}: {violation.message}")
