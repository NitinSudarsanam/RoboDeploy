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

    def _ros2_cmd(self, *args: str) -> subprocess.CompletedProcess[str]:
        ros2 = shutil.which("ros2")
        if not ros2:
            raise FileNotFoundError("Could not find `ros2` on PATH.")
        return subprocess.run(
            [ros2, *args],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

    def _controller_states(self) -> dict[str, str]:
        """Map controller name -> state string (e.g. active, inactive, configured)."""
        result = self._ros2_cmd("control", "list_controllers")
        states: dict[str, str] = {}
        for line in (result.stdout or "").splitlines():
            line = line.strip()
            if not line or line.startswith("["):
                continue
            parts = line.split()
            if len(parts) >= 2 and parts[0] and parts[-1]:
                states[parts[0]] = parts[-1].rstrip(".")
        return states

    def _controller_active(self, ctl: str) -> bool:
        return self._controller_states().get(str(ctl)) == "active"

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

    def _spawn_controller(self, ctl: str) -> tuple[bool, str]:
        if self._controller_active(ctl):
            return True, f"{ctl} already active"
        ros2 = shutil.which("ros2")
        if not ros2:
            return True, ""
        states = self._controller_states()
        known = str(ctl) in states
        arg_sets: tuple[tuple[str, ...], ...]
        if known:
            # gz_ros2_control may preload controllers; only activation is needed.
            arg_sets = ((str(ctl), "--activate"), (str(ctl),))
        else:
            arg_sets = ((str(ctl),), (str(ctl), "--activate"))
        last_out = ""
        for args in arg_sets:
            result = subprocess.run(
                [ros2, "run", "controller_manager", "spawner", *args],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            last_out = result.stdout or ""
            if result.returncode == 0 or self._controller_active(ctl):
                return True, last_out
            if "already loaded" in last_out.lower():
                if self._controller_active(ctl):
                    return True, last_out
                switch = self._ros2_cmd(
                    "control",
                    "switch_controllers",
                    "--activate",
                    str(ctl),
                )
                last_out = switch.stdout or last_out
                if switch.returncode == 0 or self._controller_active(ctl):
                    return True, last_out
        return False, last_out

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
                ok, last_out = self._spawn_controller(str(ctl))
                if ok:
                    break
                time.sleep(1.5)
            else:
                raise RuntimeError(
                    f"controller_manager spawner failed for '{ctl}' with code 1:\n{last_out}"
                )

