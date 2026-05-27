"""ROS2 joint-position controller adapter (generic).

Assumptions:
  - JointState comes from `sensor_msgs/JointState`
  - Commands are `std_msgs/Float64MultiArray` of desired joint positions

No robot-specific code. Joint ordering is configured via `ControllerConfig.joint_names`
or learned from first JointState message.
"""

from __future__ import annotations

import threading
import time
import warnings
from typing import Optional

import numpy as np

from robodeploy.backends.real.common import Commander, StateCache
from robodeploy.core.spaces import ActionSpace
from robodeploy.core.types import Action, Observation

from ._clamp import slew_limit_command
from .base import ControllerConfig, register_controller
from robodeploy.ros2 import Ros2NodeAdapter


class JointPositionControllerAdapter(Ros2NodeAdapter):
    controller_type = "joint_position"
    supported_action_spaces = [ActionSpace.JOINT_POS]

    def __init__(self, cfg: ControllerConfig, backend_config: Optional[dict] = None) -> None:
        super().__init__()
        self._robot_id = cfg.robot_id
        self._ns = (cfg.namespace or "").rstrip("/")
        self._joint_states_topic = cfg.joint_states_topic
        self._cmd_topic = cfg.cmd_topic
        self._joint_state_timeout_s = float(cfg.joint_state_timeout_s)
        self._base_frame = cfg.base_frame
        self._ee_frame = cfg.ee_frame

        self._backend_config = backend_config or {}

        self._lock = threading.Lock()
        self._has_joint_state = False
        self._last_joint_state_wall_s: float = 0.0
        self._last_joint_state_stamp_s: float = 0.0
        self._joint_state_event = threading.Event()

        self._joint_names: Optional[list[str]] = list(cfg.joint_names) if cfg.joint_names else None
        self._q = np.zeros(0, dtype=np.float64)
        self._qd = np.zeros(0, dtype=np.float64)
        self._tau = np.zeros(0, dtype=np.float64)

        self._obs_cache = StateCache()
        self._cmd_hz = float(cfg.command_hz or 0.0)
        self._max_vel = (
            np.asarray(cfg.max_joint_velocity, dtype=np.float64).reshape(-1)
            if cfg.max_joint_velocity is not None
            else None
        )
        self._commander = Commander(
            self._publish_joint_positions,
            min_period_s=(1.0 / self._cmd_hz) if self._cmd_hz > 0 else 0.0,
        )

        self.node_name = f"robodeploy_jointpos_{self._robot_id}"
        self._executor = None
        self._executor_thread: Optional[threading.Thread] = None
        self._last_tf_error: str | None = None
        self._warned_tf_failure = False

    @property
    def robot_id(self) -> str:
        return self._robot_id

    @property
    def base_frame(self) -> str:
        return self._base_frame

    @property
    def ee_frame(self) -> str:
        return self._ee_frame

    @property
    def joint_names(self) -> list[str]:
        with self._lock:
            return list(self._joint_names or [])

    def _on_node_ready(self, node) -> None:
        try:
            import tf2_ros
            from sensor_msgs.msg import JointState
            from std_msgs.msg import Float64MultiArray
        except ImportError as exc:
            raise ImportError(
                "ROS 2 packages not found. Ensure you are in a ROS 2 Jazzy Python environment "
                "with rclpy / tf2_ros / sensor_msgs / std_msgs installed and sourced."
            ) from exc

        self._cmd_msg_type = Float64MultiArray
        self._cmd_pub = node.create_publisher(
            Float64MultiArray,
            f"{self._ns}/{self._cmd_topic}" if self._ns else f"/{self._cmd_topic}",
            10,
        )

        node.create_subscription(
            JointState,
            f"{self._ns}/{self._joint_states_topic}" if self._ns else f"/{self._joint_states_topic}",
            self._on_joint_state,
            10,
        )

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, node)

        self._wait_for_joint_state(timeout_s=10.0)

    def _on_node_stopping(self, node) -> None:
        del node
        self._cmd_pub = None

    def _ensure_buffers(self, dof: int) -> None:
        if self._q.shape[0] == dof:
            return
        self._q = np.zeros(dof, dtype=np.float64)
        self._qd = np.zeros(dof, dtype=np.float64)
        self._tau = np.zeros(dof, dtype=np.float64)

    def _on_joint_state(self, msg) -> None:
        name_to_idx = {n: i for i, n in enumerate(msg.name)}

        with self._lock:
            if self._joint_names is None:
                self._joint_names = list(msg.name)
            joint_names = list(self._joint_names)

        dof = len(joint_names)
        self._ensure_buffers(dof)

        q = np.zeros(dof, dtype=np.float64)
        qd = np.zeros(dof, dtype=np.float64)
        tau = np.zeros(dof, dtype=np.float64)
        for out_idx, jn in enumerate(joint_names):
            src = name_to_idx.get(jn)
            if src is None:
                continue
            if msg.position:
                q[out_idx] = msg.position[src]
            if msg.velocity:
                qd[out_idx] = msg.velocity[src]
            if msg.effort:
                tau[out_idx] = msg.effort[src]

        with self._lock:
            self._q[:] = q
            self._qd[:] = qd
            self._tau[:] = tau
            self._has_joint_state = True
            self._last_joint_state_wall_s = time.time()
            try:
                stamp = msg.header.stamp
                self._last_joint_state_stamp_s = float(stamp.sec) + float(stamp.nanosec) * 1e-9
            except Exception:
                self._last_joint_state_stamp_s = self._last_joint_state_wall_s
        self._joint_state_event.set()

    def _wait_for_joint_state(self, *, timeout_s: float) -> None:
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            if self._has_joint_state:
                return
            self._joint_state_event.wait(timeout=0.05)
        self.stop()
        raise RuntimeError(
            f"Timed out after {timeout_s}s waiting for joint states on "
            f"{self._ns}/{self._joint_states_topic}" if self._ns else f"/{self._joint_states_topic}"
        )

    def _wait_for_new_joint_state(self, *, last_stamp_s: float, timeout_s: float) -> None:
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            with self._lock:
                cur = float(self._last_joint_state_stamp_s)
            if cur > float(last_stamp_s) + 1e-9:
                return
            self._joint_state_event.wait(timeout=0.01)
        return

    def _publish_joint_positions(self, positions_rad: np.ndarray) -> None:
        msg = self._cmd_msg_type()
        msg.data = [float(x) for x in np.asarray(positions_rad, dtype=np.float64).reshape(-1).tolist()]
        self._cmd_pub.publish(msg)

    def send_action(self, action: Action) -> None:
        if action.joint_positions is None:
            return
        q_des = np.asarray(action.joint_positions, dtype=np.float64).reshape(-1)
        with self._lock:
            q_cur = self._q.copy()
        q_des = slew_limit_command(
            q_des,
            q_cur,
            max_joint_velocity=self._max_vel,
            command_hz=self._cmd_hz,
        )
        self._commander.send(q_des)

    def send_action_and_wait(self, action: Action) -> None:
        with self._lock:
            last = float(self._last_joint_state_stamp_s)
        self.send_action(action)
        self._wait_for_new_joint_state(last_stamp_s=last, timeout_s=self._joint_state_timeout_s)

    def _get_ee_pose_from_tf(self) -> tuple[np.ndarray, np.ndarray]:
        try:
            # tf2_ros Buffer API expects rclpy.time.Time(); we keep it local to avoid hard deps.
            import rclpy.time

            tf_stamped = self._tf_buffer.lookup_transform(self._base_frame, self._ee_frame, rclpy.time.Time())
            tr = tf_stamped.transform.translation
            rot = tf_stamped.transform.rotation
            pos = np.array([tr.x, tr.y, tr.z], dtype=np.float64)
            quat = np.array([rot.w, rot.x, rot.y, rot.z], dtype=np.float64)
            self._last_tf_error = None
            return pos, quat
        except Exception as exc:
            self._last_tf_error = f"{type(exc).__name__}: {exc}"
            if not self._warned_tf_failure:
                warnings.warn(
                    f"TF lookup failed for {self._base_frame}->{self._ee_frame}; ee pose is invalid: {self._last_tf_error}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._warned_tf_failure = True
            if bool(self._backend_config.get("allow_identity_tf_fallback", False)):
                return np.zeros(3, dtype=np.float64), np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            return np.full(3, np.nan, dtype=np.float64), np.full(4, np.nan, dtype=np.float64)

    def get_obs(self) -> Observation:
        with self._lock:
            q = self._q.copy()
            qd = self._qd.copy()
            tau = self._tau.copy()
        ee_pos, ee_quat = self._get_ee_pose_from_tf()
        obs = Observation(
            joint_positions=q.astype(np.float32),
            joint_velocities=qd.astype(np.float32),
            joint_torques=tau.astype(np.float32),
            ee_position=ee_pos.astype(np.float32),
            ee_orientation=ee_quat.astype(np.float32),
            ee_velocity=np.zeros((3,), dtype=np.float32),
            ee_angular_velocity=np.zeros((3,), dtype=np.float32),
        )
        self._obs_cache.write(obs)
        return obs

    def get_diagnostics(self) -> dict:
        with self._lock:
            last_wall = float(self._last_joint_state_wall_s)
            last_stamp = float(self._last_joint_state_stamp_s)
        age_s = time.time() - last_wall if last_wall > 0 else 1e9
        return {
            "robot_id": self._robot_id,
            "controller_type": self.controller_type,
            "joint_state_timeout_s": float(self._joint_state_timeout_s),
            "last_joint_state_age_s": float(age_s),
            "last_joint_state_stamp_s": float(last_stamp),
            "command_count": self._commander.record.count,
            "last_command_wall_s": self._commander.record.sent_wall_s,
            "ee_pose_valid": self._last_tf_error is None,
            "last_tf_error": self._last_tf_error,
        }


@register_controller("joint_position")
def create_joint_position_adapter(cfg: ControllerConfig, backend_config: dict):
    return JointPositionControllerAdapter(cfg, backend_config)

