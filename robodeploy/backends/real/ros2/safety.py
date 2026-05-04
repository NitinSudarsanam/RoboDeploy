"""Reusable safety primitives for ROS2 hardware controller adapters (stdlib only)."""

from __future__ import annotations

import signal
import sys
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from robodeploy.backends.errors import BackendError


class SafetyError(BackendError):
    """Raised when a hardware safety guard trips (torque should already be released)."""


class Watchdog:
    """Calls ``on_timeout`` once if ``feed()`` is not called within ``timeout_s``."""

    def __init__(self, timeout_s: float, on_timeout: Callable[[], None]) -> None:
        self._timeout_s = max(float(timeout_s), 1e-3)
        self._on_timeout = on_timeout
        self._last_feed = time.monotonic()
        self._armed = False
        self._fired = False
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name="robodeploy_watchdog", daemon=True)

    def arm(self) -> None:
        with self._lock:
            if self._armed:
                return
            self._armed = True
            self._last_feed = time.monotonic()
        self._thread.start()

    def disarm(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def feed(self) -> None:
        with self._lock:
            self._last_feed = time.monotonic()

    def _run(self) -> None:
        while not self._stop.wait(timeout=0.05):
            with self._lock:
                if not self._armed:
                    continue
                last = self._last_feed
                fired = self._fired
            if fired:
                continue
            if time.monotonic() - last > self._timeout_s:
                with self._lock:
                    if self._fired:
                        continue
                    self._fired = True
                try:
                    self._on_timeout()
                except Exception:
                    pass
                return


class EStop:
    """SIGINT and optional console ``q`` line trip a shared ``threading.Event``."""

    def __init__(self, *, enable_console: bool = True) -> None:
        self._event = threading.Event()
        self._old_handler: Any = None
        self._stdin_thread: threading.Thread | None = None
        self._enable_console = enable_console

    @property
    def tripped(self) -> bool:
        return self._event.is_set()

    def check(self) -> None:
        if self._event.is_set():
            raise SafetyError("E-stop active (SIGINT or console 'q').")

    def start(self) -> None:
        def _handler(signum: int, frame: Any) -> None:
            del signum, frame
            self._event.set()

        try:
            self._old_handler = signal.signal(signal.SIGINT, _handler)
        except ValueError:
            # Not main thread (e.g. some test runners)
            self._old_handler = None

        if self._enable_console and sys.stdin.isatty():

            def _stdin_loop() -> None:
                while not self._event.is_set():
                    try:
                        line = sys.stdin.readline()
                    except Exception:
                        break
                    if not line:
                        break
                    if line.strip().lower() == "q":
                        self._event.set()
                        break

            self._stdin_thread = threading.Thread(target=_stdin_loop, name="robodeploy_estop_stdin", daemon=True)
            self._stdin_thread.start()

    def stop(self) -> None:
        if self._old_handler is not None:
            try:
                signal.signal(signal.SIGINT, self._old_handler)
            except ValueError:
                pass
            self._old_handler = None


class JointLimitGuard:
    """Position and finite-difference velocity checks."""

    def __init__(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
        vel_max: np.ndarray,
    ) -> None:
        self._lower = np.asarray(lower, dtype=np.float64).reshape(-1)
        self._upper = np.asarray(upper, dtype=np.float64).reshape(-1)
        self._vel_max = np.asarray(vel_max, dtype=np.float64).reshape(-1)
        self._q_prev: np.ndarray | None = None

    def check(self, q: np.ndarray, *, dt: float | None) -> None:
        qv = np.asarray(q, dtype=np.float64).reshape(-1)
        if qv.shape[0] != self._lower.shape[0]:
            raise SafetyError(f"joint limit guard dof mismatch: {qv.shape[0]} vs {self._lower.shape[0]}")
        if np.any(qv < self._lower) or np.any(qv > self._upper):
            raise SafetyError("Joint position outside soft limits.")
        if dt is not None and float(dt) > 1e-9 and self._q_prev is not None:
            dq = (qv - self._q_prev) / float(dt)
            if np.any(np.abs(dq) > self._vel_max + 1e-6):
                raise SafetyError("Joint velocity exceeded limit (finite-difference check).")
        self._q_prev = qv.copy()


class TemperatureGuard:
    """Poll ``read_fn`` periodically; trip if any joint temperature exceeds ``max_c``."""

    def __init__(
        self,
        read_fn: Callable[[], dict[str, float]],
        *,
        max_c: float = 70.0,
        period_s: float = 0.5,
        on_violation: Callable[[str], None],
    ) -> None:
        self._read_fn = read_fn
        self._max_c = float(max_c)
        self._period_s = max(float(period_s), 0.1)
        self._on_violation = on_violation
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="robodeploy_temp_guard", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.wait(timeout=self._period_s):
            try:
                temps = self._read_fn()
            except Exception:
                continue
            for name, t in temps.items():
                if float(t) > self._max_c:
                    try:
                        self._on_violation(f"joint {name!r} temperature {t:.1f}°C > {self._max_c:.1f}°C")
                    except Exception:
                        pass
                    return
