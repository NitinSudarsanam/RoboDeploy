"""Typed backend errors and diagnostics helpers."""

from __future__ import annotations


class BackendError(RuntimeError):
    """Base class for backend runtime failures."""


class BackendTimeoutError(BackendError):
    """Backend timed out waiting for required state or command completion."""

    def __init__(self, subsystem: str, timeout_s: float, detail: str = "") -> None:
        message = f"Backend subsystem '{subsystem}' timed out after {timeout_s:.3f}s."
        if detail:
            message = f"{message} {detail}"
        super().__init__(message)
        self.subsystem = subsystem
        self.timeout_s = timeout_s
        self.detail = detail


class BackendNotReadyError(BackendError):
    """Backend was used before it was ready to serve data."""


class ObservationStaleWarning(Warning):
    """Observation is stale relative to expected update cadence."""
