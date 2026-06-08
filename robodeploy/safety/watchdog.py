"""Command/state watchdog — trips when feed() is not called within timeout."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable


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
