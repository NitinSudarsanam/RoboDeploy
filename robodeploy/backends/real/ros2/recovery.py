"""ROS2 connection loss detection and exponential-backoff reconnect."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from robodeploy.safety.violation import Hazard, SafetyError, Severity, ViolationRecord


class ROS2RecoveryManager:
    """Detects stale joint state, retries reconnect, escalates on persistent failure."""

    def __init__(
        self,
        *,
        node: Any = None,
        reconnect_fn: Callable[[], bool] | None = None,
        state_timeout_s: float = 1.0,
        max_retries: int = 5,
        initial_backoff_s: float = 1.0,
        max_backoff_s: float = 30.0,
        on_lost: Callable[[], None] | None = None,
        on_recovered: Callable[[], None] | None = None,
        sleep_fn: Callable[[float], None] | None = None,
    ) -> None:
        self._node = node
        self._reconnect_fn = reconnect_fn
        self._state_timeout_s = max(float(state_timeout_s), 1e-3)
        self._max_retries = max(int(max_retries), 1)
        self._initial_backoff_s = max(float(initial_backoff_s), 0.0)
        self._max_backoff_s = max(float(max_backoff_s), self._initial_backoff_s)
        self._on_lost = on_lost
        self._on_recovered = on_recovered
        self._sleep = sleep_fn or time.sleep
        self._lost = False
        self._reconnecting = False
        self._retry_count = 0

    @property
    def connection_lost(self) -> bool:
        return self._lost

    @property
    def retry_count(self) -> int:
        return self._retry_count

    def on_state_stale(self, age_s: float) -> None:
        if age_s <= self._state_timeout_s:
            if self._lost and not self._reconnecting:
                self._lost = False
            return
        if self._reconnecting:
            return
        self._lost = True
        if self._on_lost is not None:
            try:
                self._on_lost()
            except Exception:
                pass
        self._start_reconnect()

    def _start_reconnect(self) -> None:
        self._reconnecting = True
        try:
            for retry in range(self._max_retries):
                self._retry_count = retry + 1
                backoff = min(self._initial_backoff_s * (2**retry), self._max_backoff_s)
                if backoff > 0:
                    self._sleep(backoff)
                if self._try_reconnect():
                    self._lost = False
                    self._retry_count = 0
                    if self._on_recovered is not None:
                        try:
                            self._on_recovered()
                        except Exception:
                            pass
                    return
            raise SafetyError(
                ViolationRecord(
                    hazard=Hazard.CONNECTION_LOST,
                    severity=Severity.CRITICAL,
                    message=f"reconnect failed after {self._max_retries} attempts",
                )
            )
        finally:
            self._reconnecting = False

    def _try_reconnect(self) -> bool:
        if self._reconnect_fn is not None:
            try:
                return bool(self._reconnect_fn())
            except Exception:
                return False
        if self._node is None:
            return False
        spin = getattr(self._node, "spin_once", None)
        if callable(spin):
            try:
                spin(timeout_sec=0.1)
                return True
            except Exception:
                return False
        return False
