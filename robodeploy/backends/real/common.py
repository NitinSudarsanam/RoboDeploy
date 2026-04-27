"""Reusable composition helpers for real backends.

Guidance for future real backends:
    - Use `StateCache` to hold latest hardware/controller state snapshot read by
      callbacks, background threads, or polling loops.
    - Use `Commander` to centralize outbound command pacing and keep basic send
      telemetry (`count`, `sent_wall_s`) without coupling to a specific SDK.
    - This pair is meant to give non-ROS real backends same composition shape as
      ROS2 drivers: state capture + command pacing + lightweight diagnostics.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, Optional, TypeVar

T = TypeVar("T")


class StateCache(Generic[T]):
    """Thread-safe last-value cache for real I/O streams.

    Good fit for latest-known robot state, sensor snapshots, or computed
    observations that may be read by control code outside callback thread.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._value: Optional[T] = None
        self._updated_wall_s: float = 0.0

    def write(self, value: T) -> None:
        with self._lock:
            self._value = value
            self._updated_wall_s = time.time()

    def read(self) -> Optional[T]:
        with self._lock:
            return self._value

    def updated_wall_s(self) -> float:
        with self._lock:
            return self._updated_wall_s


@dataclass
class CommandRecord:
    sent_wall_s: float = 0.0
    count: int = 0


class Commander:
    """Thin wrapper that rate-limits and records outgoing commands.

    Use this when backend owns command cadence and you want best-effort pacing
    plus minimal diagnostics about last send time and command count.
    """

    def __init__(self, send_fn: Callable[[object], None], min_period_s: float = 0.0) -> None:
        self._send_fn = send_fn
        self._min_period_s = float(min_period_s)
        self._last = CommandRecord()

    @property
    def record(self) -> CommandRecord:
        return self._last

    def send(self, payload: object) -> bool:
        now = time.time()
        if self._min_period_s > 0.0 and self._last.sent_wall_s > 0.0:
            if now - self._last.sent_wall_s < self._min_period_s:
                return False
        self._send_fn(payload)
        self._last.sent_wall_s = now
        self._last.count += 1
        return True
