"""Base helpers for ROS2 sensors."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Generic, Optional, TypeVar


T = TypeVar("T")


@dataclass
class LastValue(Generic[T]):
    value: Optional[T] = None
    wall_time_s: float = 0.0
    hw_time_s: float = 0.0


class LastValueCache(Generic[T]):
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._lv: LastValue[T] = LastValue()

    def write(self, value: T, *, hw_time_s: float = 0.0) -> None:
        with self._lock:
            self._lv.value = value
            self._lv.wall_time_s = time.monotonic()
            self._lv.hw_time_s = float(hw_time_s or 0.0)

    def read(self) -> LastValue[T]:
        with self._lock:
            return LastValue(value=self._lv.value, wall_time_s=self._lv.wall_time_s, hw_time_s=self._lv.hw_time_s)

