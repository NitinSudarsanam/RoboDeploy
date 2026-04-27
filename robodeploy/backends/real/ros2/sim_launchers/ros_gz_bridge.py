"""ros_gz_bridge launcher helpers."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RosGzBridgeConfig:
    rules: tuple[str, ...] = ()


class RosGzBridgeLauncher:
    def __init__(self, cfg: RosGzBridgeConfig) -> None:
        self._cfg = cfg
        self._proc: Optional[subprocess.Popen] = None

    def start(self) -> None:
        ros2 = shutil.which("ros2")
        if not ros2:
            return
        rules = list(self._cfg.rules)
        if not rules:
            rules = ["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"]
        self._proc = subprocess.Popen(
            [ros2, "run", "ros_gz_bridge", "parameter_bridge", *rules],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            text=True,
        )

    def stop(self) -> None:
        if self._proc is None:
            return
        try:
            self._proc.terminate()
        except Exception:
            pass
        try:
            self._proc.wait(timeout=5.0)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass
        self._proc = None

