"""ros_gz_bridge launcher helpers."""

from __future__ import annotations

import shutil
import subprocess
import warnings
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
        rules = [
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            "/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V",
            "/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model",
            *list(self._cfg.rules),
        ]
        rules = list(dict.fromkeys(rules))
        self._proc = subprocess.Popen(
            [ros2, "run", "ros_gz_bridge", "parameter_bridge", *rules],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if self._proc.poll() is not None:
            out = self._proc.stdout.read() if self._proc.stdout is not None else ""
            raise RuntimeError(f"ros_gz_bridge exited during startup:\n{out}")

    def stop(self) -> None:
        if self._proc is None:
            return
        try:
            self._proc.terminate()
        except Exception:
            warnings.warn("Failed to terminate ros_gz_bridge process.", RuntimeWarning, stacklevel=2)
            pass
        try:
            self._proc.wait(timeout=5.0)
        except Exception:
            try:
                self._proc.kill()
            except Exception as exc:
                warnings.warn(f"Failed to kill ros_gz_bridge process: {exc}", RuntimeWarning, stacklevel=2)
        self._proc = None


def image_bridge_rules(*topics: str) -> tuple[str, ...]:
    """Return ros_gz_bridge rules for Gazebo image topics."""

    rules: list[str] = []
    for topic in topics:
        t = str(topic or "").strip()
        if not t:
            continue
        if not t.startswith("/"):
            t = f"/{t}"
        rules.append(f"{t}@sensor_msgs/msg/Image[gz.msgs.Image")
    return tuple(dict.fromkeys(rules))

