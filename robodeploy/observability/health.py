"""Sensor health monitoring from per-step sensor_status maps."""

from __future__ import annotations

from typing import Callable, Literal

HealthStatus = Literal["ok", "degraded", "failed"]


def summarize_sensor_health(sensor_status: dict[str, str] | None) -> dict[str, object]:
    """Summarize sensor_status into counts and overall health label."""
    status = dict(sensor_status or {})
    if not status:
        return {"overall": "ok", "counts": {}, "sensors": {}}
    counts: dict[str, int] = {}
    for value in status.values():
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    overall: HealthStatus = "ok"
    if any(v != "ok" for v in status.values()):
        overall = "degraded"
    if counts.get("error", 0) > 0:
        overall = "failed"
    return {"overall": overall, "counts": counts, "sensors": status}


class HealthMonitor:
    """Track consecutive sensor failures and optionally invoke a callback."""

    def __init__(
        self,
        *,
        fail_threshold_per_sensor: int = 5,
        on_failure: Callable[[str, dict[str, str]], None] | None = None,
    ) -> None:
        self._fail_threshold = max(1, int(fail_threshold_per_sensor))
        self._on_failure = on_failure
        self._fail_counts: dict[str, int] = {}

    def observe(self, sensor_status: dict[str, str] | None) -> HealthStatus:
        status = dict(sensor_status or {})
        for name, value in status.items():
            if str(value) == "ok":
                self._fail_counts[name] = 0
                continue
            self._fail_counts[name] = self._fail_counts.get(name, 0) + 1
            if self._fail_counts[name] > self._fail_threshold:
                if self._on_failure is not None:
                    self._on_failure(name, status)
                return "failed"
        if any(str(v) != "ok" for v in status.values()):
            return "degraded"
        return "ok"

    @property
    def fail_counts(self) -> dict[str, int]:
        return dict(self._fail_counts)
