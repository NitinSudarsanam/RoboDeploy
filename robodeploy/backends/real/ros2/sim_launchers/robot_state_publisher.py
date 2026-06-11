"""Launch `robot_state_publisher` as a subprocess (URDF -> /robot_description + /tf).

Important: `robot_state_publisher` consumes `sensor_msgs/JointState` to publish the moving-joint TF tree.
In multi-robot setups we namespace joint states (e.g. `/robot0/joint_states`), so we must remap the
`/joint_states` subscription accordingly.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


class RobotStatePublisherLauncher:
    """Best-effort lifecycle for one `robot_state_publisher` instance."""

    def __init__(self, urdf_text: str, *, namespace: str = "", joint_states_topic: str = "/joint_states") -> None:
        self._urdf_text = urdf_text
        self._namespace = str(namespace or "")
        self._joint_states_topic = str(joint_states_topic or "/joint_states")
        self._proc: Optional[subprocess.Popen] = None
        self._params_path: Optional[Path] = None

    def start(self) -> None:
        ros2 = shutil.which("ros2")
        if not ros2:
            return
        # Passing large URDF XML via `-p robot_description:=...` is brittle on some platforms
        # due to command-line length limits and newline/escaping issues. Prefer a params file.
        fd, p = tempfile.mkstemp(prefix="robodeploy_rsp_", suffix=".yaml")
        Path(p).write_text(
            "robot_state_publisher:\n"
            "  ros__parameters:\n"
            "    publish_frequency: 100.0\n"
            "    robot_description: |\n"
            + "\n".join(f"      {line}" for line in self._urdf_text.splitlines())
            + "\n",
            encoding="utf-8",
        )
        self._params_path = Path(p)

        # Build absolute joint_states target topic.
        # If caller provided a relative topic like "joint_states" and a namespace "/robot0",
        # this becomes "/robot0/joint_states".
        ns = self._namespace.rstrip("/")
        js = self._joint_states_topic
        if not js.startswith("/"):
            js = f"/{js}"
        if ns and not ns.startswith("/"):
            ns = f"/{ns}"
        target_joint_states = f"{ns}{js}" if ns else js

        self._proc = subprocess.Popen(
            [
                ros2,
                "run",
                "robot_state_publisher",
                "robot_state_publisher",
                "--ros-args",
                "--params-file",
                str(self._params_path),
                "-r",
                f"/joint_states:={target_joint_states}",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            text=True,
        )

    def stop(self) -> None:
        if self._proc is None:
            # Still attempt to clean up params file if one was created.
            if self._params_path is not None:
                try:
                    self._params_path.unlink(missing_ok=True)
                except Exception:
                    pass
                self._params_path = None
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

        if self._params_path is not None:
            try:
                self._params_path.unlink(missing_ok=True)
            except Exception:
                pass
            self._params_path = None
