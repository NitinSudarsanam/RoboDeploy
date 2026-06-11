"""Gazebo launcher for ROS2Backend (best-effort).

Goal: minimize user work by launching Gazebo + bridge processes from Python,
while keeping ROS2Backend transport-only (it still talks to ROS topics/TF).

This is intentionally conservative: we only manage subprocess lifecycles and
provide clear errors when required binaries are missing.
"""

from __future__ import annotations

import os
import re
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
    # URDF joint names expected on ``/joint_states`` (for RSP / TF readiness).
    expected_joint_names: tuple[str, ...] = ()
    readiness_timeout_s: float = 15.0


def _ros_gz_plugin_lib_dir() -> Path | None:
    """Locate ``libgz_ros2_control-system.so`` for Gazebo Harmonic plugin loading."""
    ros2 = shutil.which("ros2")
    if ros2:
        try:
            out = subprocess.run(
                [ros2, "pkg", "prefix", "gz_ros2_control"],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            ).stdout.strip()
            if out:
                lib_dir = Path(out) / "lib"
                if (lib_dir / "libgz_ros2_control-system.so").is_file():
                    return lib_dir
        except Exception:
            pass
    for candidate in (Path("/opt/ros/jazzy/lib"),):
        if (candidate / "libgz_ros2_control-system.so").is_file():
            return candidate
    return None


def _gz_subprocess_env() -> dict[str, str]:
    """Augment env so Harmonic finds ROS-packaged ``gz_ros2_control`` plugins."""
    env = dict(os.environ)
    lib_dir = _ros_gz_plugin_lib_dir()
    if lib_dir is None:
        return env
    lib = str(lib_dir)
    plugin_path = env.get("GZ_SIM_SYSTEM_PLUGIN_PATH", "")
    env["GZ_SIM_SYSTEM_PLUGIN_PATH"] = f"{lib}:{plugin_path}" if plugin_path else lib
    ld_path = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{lib}:{ld_path}" if ld_path else lib
    return env


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

        gz_env = _gz_subprocess_env()
        self._gz_log_file = tempfile.NamedTemporaryFile(prefix="robodeploy_gazebo_", suffix=".log", delete=False, mode="w", encoding="utf-8")
        self._gz_log_path = Path(self._gz_log_file.name)
        self._gz_proc = subprocess.Popen(
            args,
            stdout=self._gz_log_file,
            stderr=subprocess.STDOUT,
            text=True,
            env=gz_env,
        )

        if self._cfg.start_ros_gz_bridge:
            use_ros2_control = bool(self._cfg.controllers_to_spawn)
            # When gz_ros2_control + robot_state_publisher own the graph, bridging Gazebo
            # /tf fights RSP and yields flaky world/base_link->ee_link lookups.
            bridge_tf = not (use_ros2_control and self._cfg.robot_urdf)
            self._bridge = RosGzBridgeLauncher(
                RosGzBridgeConfig(
                    rules=tuple(self._cfg.bridge_rules),
                    bridge_tf=bridge_tf,
                    bridge_joint_states=not use_ros2_control,
                )
            )
            self._bridge.start()

        self._wait_for_process_start(timeout_s=min(2.0, float(self._cfg.readiness_timeout_s)))
        self._unpause_gazebo_world(world_path)

        urdf_path: Path | None = None
        urdf_text: str | None = None
        if self._cfg.robot_urdf:
            candidate = Path(self._cfg.robot_urdf)
            if candidate.exists():
                urdf_path = candidate
                urdf_text = urdf_path.read_text(encoding="utf-8")

        # gz_ros2_control controller_manager blocks until /robot_description exists.
        if urdf_text is not None:
            # Gazebo joint_states are sim-stamped via /clock; RSP must use sim time so its
            # publish_frequency timer does not stamp TF with wall time (TF_OLD_DATA spam).
            rsp_use_sim_time = bool(self._cfg.start_ros_gz_bridge)
            self._rsp = RobotStatePublisherLauncher(urdf_text, use_sim_time=rsp_use_sim_time)
            self._rsp.start()
            self._wait_for_topics(("/robot_description",), timeout_s=min(30.0, float(self._cfg.readiness_timeout_s)))

        # Spawn robot via ros_gz_sim create (URDF) before controllers activate.
        if urdf_path is not None:
            UrdfSpawner(UrdfSpawnConfig(
                urdf_path=str(urdf_path),
                name=str(self._cfg.robot_name),
                xyz=tuple(self._cfg.robot_pose_xyz),
                rpy=tuple(self._cfg.robot_pose_rpy),
            )).spawn()
            settle_s = min(25.0, max(8.0, float(self._cfg.readiness_timeout_s) * 0.2))
            time.sleep(settle_s)

        # Optional: spawn controllers (joint_state_broadcaster must be active before TF gate).
        if self._cfg.controllers_to_spawn:
            ctl_cfg = ControllerSpawnConfig(
                controllers=tuple(self._cfg.controllers_to_spawn),
                timeout_s=float(self._cfg.readiness_timeout_s),
            )
            ctl_spawner = ControllerSpawner(ctl_cfg)
            try:
                ctl_spawner.wait_for_controller_manager()
            except TimeoutError as exc:
                log = self._read_gazebo_log()
                raise TimeoutError(f"{exc}\nGazebo log tail:\n{log}") from exc
            ctl_spawner.spawn_all()

        if urdf_path is not None:
            js_timeout = float(self._cfg.readiness_timeout_s)
            self._wait_for_topics(("/joint_states",), timeout_s=js_timeout)
            if not self._wait_for_joint_states_message(timeout_s=min(45.0, js_timeout * 0.5)):
                raise TimeoutError("Timed out waiting for first /joint_states message from gz_ros2_control.")
            if self._cfg.expected_joint_names:
                if not self._wait_for_joint_state_names(
                    tuple(self._cfg.expected_joint_names),
                    timeout_s=js_timeout,
                ):
                    raise TimeoutError(
                        "Timed out waiting for /joint_states names to match URDF: "
                        f"expected {list(self._cfg.expected_joint_names)}"
                    )
            time.sleep(2.0)
            if not self._wait_for_tf_transform(timeout_s=min(90.0, js_timeout * 0.75)):
                import os
                import warnings

                if os.environ.get("ROBODEPLOY_GAZEBO_STRICT_TF", "").strip().lower() in {"1", "true", "yes"}:
                    raise TimeoutError(
                        "Timed out waiting for TF base_link->ee_link before Gazebo pick episode."
                    )
                warnings.warn(
                    "TF base_link->ee_link not ready at Gazebo launch; using joint FK fallback for EE pose.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        # Optional: readiness checks (topics exist).
        if self._cfg.wait_for_topics:
            self._wait_for_topics(self._cfg.wait_for_topics, timeout_s=float(self._cfg.readiness_timeout_s))
            self._wait_for_topic_messages(
                tuple(t for t in self._cfg.wait_for_topics if t != "/clock"),
                timeout_s=min(30.0, float(self._cfg.readiness_timeout_s) * 0.5),
            )

    @staticmethod
    def _world_name_from_sdf(world_path: Path) -> str:
        try:
            text = world_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return "world"
        match = re.search(r'<world[^>]*\bname="([^"]+)"', text)
        return match.group(1) if match else "world"

    def _unpause_gazebo_world(self, world_path: Path) -> None:
        gz = shutil.which("gz")
        if not gz:
            return
        world_name = self._world_name_from_sdf(world_path)
        service = f"/world/{world_name}/control"
        try:
            subprocess.run(
                [
                    gz,
                    "service",
                    "-s",
                    service,
                    "--reqtype",
                    "gz.msgs.WorldControl",
                    "--reptype",
                    "gz.msgs.Boolean",
                    "--timeout",
                    "3000",
                    "--req",
                    "pause: false",
                ],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=_gz_subprocess_env(),
            )
        except Exception:
            pass

    def _wait_for_process_start(self, *, timeout_s: float) -> None:
        assert self._gz_proc is not None
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            code = self._gz_proc.poll()
            if code is not None:
                log = self._read_gazebo_log()
                raise RuntimeError(f"Gazebo exited during startup with code {code}.\nLog: {self._gz_log_path}\n{log}")
            time.sleep(0.1)

    def _wait_for_tf_transform(self, *, timeout_s: float) -> bool:
        """Best-effort wait for EE TF (robot_state_publisher + joint_states)."""
        ros2 = shutil.which("ros2")
        if not ros2:
            return False
        pairs = (("world", "ee_link"), ("base_link", "ee_link"))
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            remaining = max(1.0, deadline - time.monotonic())
            if self._tf_echo_available(ros2, pairs, timeout_s=min(8.0, remaining)):
                return True
            try:
                result = subprocess.run(
                    ["ros2", "topic", "echo", "/tf", "--once"],
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=min(6.0, remaining),
                )
            except subprocess.TimeoutExpired:
                result = None
            if result is not None and result.returncode == 0:
                out = result.stdout or ""
                if "ee_link" in out and ("base_link" in out or "world" in out):
                    return True
            time.sleep(0.5)
        return False

    @staticmethod
    def _tf_echo_available(ros2: str, pairs: tuple[tuple[str, str], ...], *, timeout_s: float) -> bool:
        for parent, child in pairs:
            try:
                result = subprocess.run(
                    ["ros2", "run", "tf2_ros", "tf2_echo", parent, child],
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=max(1.0, float(timeout_s)),
                )
            except subprocess.TimeoutExpired:
                continue
            out = (result.stdout or "").lower()
            if "translation" in out:
                return True
        return False

    def _wait_for_joint_state_names(self, expected: tuple[str, ...], *, timeout_s: float) -> bool:
        """Wait until ``/joint_states`` publishes all ``expected`` joint names."""
        ros2 = shutil.which("ros2")
        if not ros2 or not expected:
            return False
        need = {str(n) for n in expected}
        deadline = time.monotonic() + float(timeout_s)
        topics = ("/joint_states", "/robot0/joint_states")
        while time.monotonic() < deadline:
            for topic in topics:
                remaining = max(1.0, deadline - time.monotonic())
                try:
                    result = subprocess.run(
                        [ros2, "topic", "echo", topic, "--once"],
                        check=False,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        timeout=min(10.0, remaining),
                    )
                except subprocess.TimeoutExpired:
                    continue
                if result.returncode != 0 or not (result.stdout or "").strip():
                    continue
                found: set[str] = set()
                in_names = False
                for line in result.stdout.splitlines():
                    s = line.strip()
                    if s.startswith("name:"):
                        in_names = True
                        tail = s.split(":", 1)[-1].strip()
                        if tail and tail != "[]":
                            found.add(tail.strip("'\""))
                        continue
                    if in_names and s.startswith("- "):
                        found.add(s[2:].strip().strip("'\""))
                    elif in_names and s and not s.startswith("-"):
                        in_names = False
                if need.issubset(found):
                    return True
            time.sleep(0.5)
        return False

    def _wait_for_topic_messages(self, topics: tuple[str, ...], *, timeout_s: float) -> None:
        """Best-effort wait for at least one message on each topic (sensors)."""
        ros2 = shutil.which("ros2")
        if not ros2:
            return
        remaining_topics = {str(t) for t in topics if str(t).strip()}
        if not remaining_topics:
            return
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline and remaining_topics:
            for topic in list(remaining_topics):
                try:
                    result = subprocess.run(
                        [ros2, "topic", "echo", topic, "--once"],
                        check=False,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        timeout=min(8.0, max(1.0, deadline - time.monotonic())),
                    )
                except subprocess.TimeoutExpired:
                    continue
                if result.returncode == 0 and (result.stdout or "").strip():
                    remaining_topics.discard(topic)
            time.sleep(0.3)

    def _wait_for_joint_states_message(self, *, timeout_s: float) -> bool:
        """Best-effort wait for ``/joint_states`` to publish at least one message."""
        ros2 = shutil.which("ros2")
        if not ros2:
            return False
        deadline = time.monotonic() + float(timeout_s)
        topics = ("/joint_states", "/robot0/joint_states")
        while time.monotonic() < deadline:
            for topic in topics:
                remaining = max(1.0, deadline - time.monotonic())
                try:
                    result = subprocess.run(
                        [ros2, "topic", "echo", topic, "--once"],
                        check=False,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        timeout=min(10.0, remaining),
                    )
                except subprocess.TimeoutExpired:
                    continue
                if result.returncode == 0 and (result.stdout or "").strip():
                    return True
            time.sleep(0.5)
        return False

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

