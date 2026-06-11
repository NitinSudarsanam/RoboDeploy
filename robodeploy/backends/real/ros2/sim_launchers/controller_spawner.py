"""controller_manager spawner helpers with readiness gates."""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class ControllerSpawnConfig:
    controllers: tuple[str, ...] = ()
    controller_manager_ns: str = ""  # optional namespace prefix
    timeout_s: float = 15.0


class ControllerSpawner:
    def __init__(self, cfg: ControllerSpawnConfig) -> None:
        self._cfg = cfg

    def wait_for_controller_manager(self) -> None:
        ros2 = shutil.which("ros2")
        if not ros2:
            return
        deadline = time.monotonic() + float(self._cfg.timeout_s)
        svc = f"{self._cfg.controller_manager_ns}/controller_manager/list_controllers".replace("//", "/")
        while time.monotonic() < deadline:
            out = subprocess.run([ros2, "service", "list"], check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True).stdout
            if svc in out:
                return
            time.sleep(0.2)
        raise TimeoutError(f"controller_manager service not available before timeout: {svc}")

    def spawn_all(self) -> None:
        if not self._cfg.controllers:
            return
        ros2 = shutil.which("ros2")
        if not ros2:
            return
        self.wait_for_controller_manager()
        per_ctl_timeout = max(float(self._cfg.timeout_s), 30.0)
        for ctl in self._cfg.controllers:
            deadline = time.monotonic() + per_ctl_timeout
            last_out = ""
            while time.monotonic() < deadline:
                for args in (
                    [str(ctl), "--activate"],
                    [str(ctl)],
                ):
                    result = subprocess.run(
                        [ros2, "run", "controller_manager", "spawner", *args],
                        check=False,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    last_out = result.stdout or ""
                    if result.returncode == 0:
                        break
                else:
                    time.sleep(1.5)
                    continue
                break
            else:
                raise RuntimeError(
                    f"controller_manager spawner failed for '{ctl}' with code 1:\n{last_out}"
                )

