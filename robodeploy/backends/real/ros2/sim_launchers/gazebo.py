"""Gazebo launcher for ROS2Backend (best-effort).

Goal: minimize user work by launching Gazebo + bridge processes from Python,
while keeping ROS2Backend transport-only (it still talks to ROS topics/TF).

This is intentionally conservative: we only manage subprocess lifecycles and
provide clear errors when required binaries are missing.
"""

from __future__ import annotations

import shutil
import subprocess
import time
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .controller_spawner import ControllerSpawnConfig, ControllerSpawner
from .ros_gz_bridge import RosGzBridgeConfig, RosGzBridgeLauncher
from .robot_state_publisher import RobotStatePublisherLauncher
from .urdf_spawner import UrdfSpawnConfig, UrdfSpawner


@dataclass(frozen=True)
class GazeboLaunchConfig:
    world: str  # path to .sdf/.world
    headless: bool = False
    extra_args: tuple[str, ...] = ()

    # Optional: ROS-GZ bridge command. If None, launcher won't start a bridge.
    start_ros_gz_bridge: bool = True

    # Optional: spawn a robot into Gazebo via ros_gz_sim create.
    # This is best-effort and requires ROS packages `ros_gz_sim` to be installed.
    robot_urdf: Optional[str] = None
    robot_name: str = "robot0"
    robot_pose_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    robot_pose_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Optional: controllers to spawn via ros2_control.
    controllers_to_spawn: tuple[str, ...] = ()
    # Optional: wait-for topics (absolute topic names), best-effort.
    wait_for_topics: tuple[str, ...] = ()
    # Optional: additional ros_gz_bridge parameter_bridge rules.
    bridge_rules: tuple[str, ...] = ()
    readiness_timeout_s: float = 15.0


class GazeboLauncher:
    """Starts/stops Gazebo and optional ros_gz_bridge subprocesses."""

    def __init__(self, cfg: GazeboLaunchConfig) -> None:
        self._cfg = cfg
        self._gz_proc: Optional[subprocess.Popen] = None
        self._bridge: Optional[RosGzBridgeLauncher] = None
        self._rsp: Optional[RobotStatePublisherLauncher] = None
        self._gz_log_path: Optional[Path] = None
        self._gz_log_file = None

    def start(self) -> None:
        gz = shutil.which("gz")
        if not gz:
            raise FileNotFoundError(
                "Could not find `gz` on PATH. Install Gazebo (gz-sim) and ensure the `gz` binary is available."
            )
        world_path = Path(self._cfg.world)
        if not world_path.exists():
            raise FileNotFoundError(f"Gazebo world not found: {world_path}")

        args = [gz, "sim", str(world_path)]
        if self._cfg.headless:
            args.append("-s")
        args.extend(list(self._cfg.extra_args))

        # Use current environment; user is responsible for sourcing ROS/GZ setup.
        self._gz_log_file = tempfile.NamedTemporaryFile(prefix="robodeploy_gazebo_", suffix=".log", delete=False, mode="w", encoding="utf-8")
        self._gz_log_path = Path(self._gz_log_file.name)
        self._gz_proc = subprocess.Popen(args, stdout=self._gz_log_file, stderr=subprocess.STDOUT, text=True)

        if self._cfg.start_ros_gz_bridge:
            self._bridge = RosGzBridgeLauncher(RosGzBridgeConfig(rules=tuple(self._cfg.bridge_rules)))
            self._bridge.start()

        self._wait_for_process_start(timeout_s=min(2.0, float(self._cfg.readiness_timeout_s)))

        # Optional: start robot_state_publisher so TF + RobotModel work in RViz when joint_states exist.
        if self._cfg.robot_urdf:
            urdf_path = Path(self._cfg.robot_urdf)
            if urdf_path.exists():
                try:
                    self._rsp = RobotStatePublisherLauncher(urdf_path.read_text(encoding="utf-8"))
                    self._rsp.start()
                except Exception:
                    self._rsp = None

        # Optional: spawn a robot via ros_gz_sim create (URDF).
        if self._cfg.robot_urdf:
            urdf_path = Path(self._cfg.robot_urdf)
            if urdf_path.exists():
                UrdfSpawner(UrdfSpawnConfig(
                    urdf_path=str(urdf_path),
                    name=str(self._cfg.robot_name),
                    xyz=tuple(self._cfg.robot_pose_xyz),
                    rpy=tuple(self._cfg.robot_pose_rpy),
                )).spawn()

        # Optional: spawn controllers.
        if self._cfg.controllers_to_spawn:
            ControllerSpawner(ControllerSpawnConfig(controllers=tuple(self._cfg.controllers_to_spawn))).spawn_all()

        # Optional: readiness checks (topics exist).
        if self._cfg.wait_for_topics:
            self._wait_for_topics(self._cfg.wait_for_topics, timeout_s=float(self._cfg.readiness_timeout_s))

    def _wait_for_process_start(self, *, timeout_s: float) -> None:
        assert self._gz_proc is not None
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            code = self._gz_proc.poll()
            if code is not None:
                log = self._read_gazebo_log()
                raise RuntimeError(f"Gazebo exited during startup with code {code}.\nLog: {self._gz_log_path}\n{log}")
            time.sleep(0.1)

    def _wait_for_topics(self, topics: tuple[str, ...], *, timeout_s: float) -> None:
        ros2 = shutil.which("ros2")
        if not ros2:
            raise FileNotFoundError("Could not find `ros2` on PATH while waiting for Gazebo topics.")
        deadline = time.monotonic() + float(timeout_s)
        remaining = set(str(t) for t in topics)
        while time.monotonic() < deadline and remaining:
            try:
                out = subprocess.run([ros2, "topic", "list"], check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True).stdout
            except Exception:
                time.sleep(0.2)
                continue
            for t in list(remaining):
                if t in out:
                    remaining.discard(t)
            time.sleep(0.2)
        if remaining:
            raise TimeoutError(f"Timed out waiting for Gazebo/ROS topics: {sorted(remaining)}")

    def _read_gazebo_log(self) -> str:
        if self._gz_log_file is not None:
            try:
                self._gz_log_file.flush()
            except Exception:
                pass
        if self._gz_log_path is None:
            return ""
        try:
            text = self._gz_log_path.read_text(encoding="utf-8", errors="replace")
            return text[-4000:]
        except Exception:
            return ""

    def stop(self) -> None:
        if self._rsp is not None:
            try:
                self._rsp.stop()
            except Exception:
                pass
            self._rsp = None

        if self._bridge is not None:
            try:
                self._bridge.stop()
            except Exception:
                pass
            self._bridge = None

        if self._gz_proc is not None:
            try:
                self._gz_proc.terminate()
            except Exception:
                pass
            try:
                self._gz_proc.wait(timeout=5.0)
            except Exception:
                try:
                    self._gz_proc.kill()
                except Exception:
                    pass
            self._gz_proc = None
        if self._gz_log_file is not None:
            try:
                self._gz_log_file.close()
            except Exception:
                pass
            self._gz_log_file = None

