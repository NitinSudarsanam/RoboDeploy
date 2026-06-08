"""Unified cross-backend safety primitives."""

from .collision import CollisionGuard
from .estop import EStop, EStopGuard
from .filter import SafetyFilterGuard
from .force_limits import ForceLimitGuard
from .injector import SafetyViolationInjector
from .joint_limits import JointLimitGuard
from .monitor import ISafetyGuard, SafetyMonitor, SafetyStatus
from .proximity import HumanProximityGuard
from .registry import (
    clear_safety_monitor,
    get_active_safety_label,
    get_active_safety_monitor,
    register_safety_monitor,
)
from .singularity import SingularityGuard
from .temperature import TemperatureGuard
from .velocity_limits import VelocityGuard
from .violation import Hazard, SafetyError, Severity, ViolationRecord
from .watchdog import Watchdog
from .workspace import WorkspaceGuard

__all__ = [
    "CollisionGuard",
    "EStop",
    "EStopGuard",
    "ForceLimitGuard",
    "Hazard",
    "HumanProximityGuard",
    "ISafetyGuard",
    "JointLimitGuard",
    "SafetyError",
    "SafetyFilterGuard",
    "SafetyMonitor",
    "SafetyStatus",
    "SafetyViolationInjector",
    "Severity",
    "SingularityGuard",
    "TemperatureGuard",
    "VelocityGuard",
    "ViolationRecord",
    "Watchdog",
    "WorkspaceGuard",
    "clear_safety_monitor",
    "get_active_safety_label",
    "get_active_safety_monitor",
    "register_safety_monitor",
]
