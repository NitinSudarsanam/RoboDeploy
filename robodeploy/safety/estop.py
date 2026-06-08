"""Cross-backend emergency stop — SIGINT, console key, file flag, multiprocessing event."""

from __future__ import annotations

import multiprocessing as mp
import signal
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from robodeploy.core.types import Action, Observation

from .violation import Hazard, SafetyError, Severity, ViolationRecord


class EStop:
    """Aggregates operator/programmatic halt signals into one trip flag."""

    def __init__(
        self,
        *,
        signal_handlers: bool = True,
        console_key: str = "q",
        enable_console: bool | None = None,
        file_flag: Path | None = None,
        mp_event: mp.synchronize.Event | None = None,
        callback: Callable[[], None] | None = None,
    ) -> None:
        self._event = threading.Event()
        self._reason = "manual"
        self._old_handler: Any = None
        self._stdin_thread: threading.Thread | None = None
        use_console = enable_console if enable_console is not None else True
        self._enable_console = use_console and bool(console_key)
        self._console_key = str(console_key).strip().lower()
        self._signal_handlers = signal_handlers
        self._file_flag = Path(file_flag) if file_flag is not None else None
        self._mp_event = mp_event
        self._callback = callback
        self._started = False

    @property
    def tripped(self) -> bool:
        return self._event.is_set()

    def check(self) -> None:
        self._poll_sources()
        if self._event.is_set():
            raise SafetyError(
                ViolationRecord(
                    hazard=Hazard.OPERATOR_ESTOP,
                    severity=Severity.CRITICAL,
                    message=self._reason,
                )
            )

    def trip(self, reason: str = "manual") -> None:
        self._reason = str(reason)
        self._event.set()
        if self._callback is not None:
            try:
                self._callback()
            except Exception:
                pass

    def reset(self) -> None:
        self._event.clear()
        self._reason = "manual"
        if self._mp_event is not None:
            try:
                self._mp_event.clear()
            except Exception:
                pass

    def start(self) -> None:
        """Install signal/console listeners (ROS2 controller compatibility alias)."""
        if self._started:
            self.reset()
        self._started = True
        if not self._signal_handlers:
            if self._enable_console:
                self._start_console_thread()
            return

        def _handler(signum: int, frame: Any) -> None:
            del signum, frame
            self.trip("SIGINT")

        try:
            self._old_handler = signal.signal(signal.SIGINT, _handler)
        except ValueError:
            self._old_handler = None

        if self._enable_console:
            self._start_console_thread()

    def stop(self) -> None:
        """Restore previous signal handler."""
        if self._old_handler is not None:
            try:
                signal.signal(signal.SIGINT, self._old_handler)
            except ValueError:
                pass
            self._old_handler = None
        self._started = False

    def _start_console_thread(self) -> None:
        if not sys.stdin.isatty():
            return

        def _stdin_loop() -> None:
            while not self._event.is_set():
                try:
                    line = sys.stdin.readline()
                except Exception:
                    break
                if not line:
                    break
                if line.strip().lower() == self._console_key:
                    self.trip(f"console key '{self._console_key}'")
                    break

        self._stdin_thread = threading.Thread(
            target=_stdin_loop,
            name="robodeploy_estop_stdin",
            daemon=True,
        )
        self._stdin_thread.start()

    def _poll_sources(self) -> None:
        if self._file_flag is not None and self._file_flag.exists():
            self.trip(f"file flag {self._file_flag}")
        if self._mp_event is not None and self._mp_event.is_set():
            self.trip("multiprocessing estop event")


class EStopGuard:
    """ISafetyGuard wrapper that raises on tripped EStop during action/obs checks."""

    def __init__(self, estop: EStop) -> None:
        self._estop = estop

    def check_action(
        self,
        action: Action,
        obs: Observation,
        *,
        dt: float,
    ) -> tuple[Action, list[ViolationRecord]]:
        del obs, dt
        self._estop.check()
        return action, []

    def check_observation(self, obs: Observation) -> list[ViolationRecord]:
        del obs
        self._estop.check()
        return []
