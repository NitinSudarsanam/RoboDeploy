"""Process-wide registry of active SafetyMonitor instances (for CLI status)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .monitor import SafetyMonitor

_ACTIVE: SafetyMonitor | None = None
_ACTIVE_LABEL: str | None = None


def register_safety_monitor(monitor: SafetyMonitor | None, *, label: str | None = None) -> None:
    global _ACTIVE, _ACTIVE_LABEL
    _ACTIVE = monitor
    _ACTIVE_LABEL = label


def get_active_safety_monitor() -> SafetyMonitor | None:
    return _ACTIVE


def get_active_safety_label() -> str | None:
    return _ACTIVE_LABEL


def clear_safety_monitor(monitor: SafetyMonitor | None = None) -> None:
    global _ACTIVE, _ACTIVE_LABEL
    if monitor is None or _ACTIVE is monitor:
        _ACTIVE = None
        _ACTIVE_LABEL = None
