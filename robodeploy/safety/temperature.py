"""Background temperature polling guard for hardware controllers."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable


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
                        self._on_violation(
                            f"joint {name!r} temperature {t:.1f}°C > {self._max_c:.1f}°C"
                        )
                    except Exception:
                        pass
                    return
