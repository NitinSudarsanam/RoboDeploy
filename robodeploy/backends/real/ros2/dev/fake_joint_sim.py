"""Tiny ROS 2 "simulator" for joint-position control demos.

Lightweight (no physics) helper used by the ROS2 backend. It publishes a
sensor_msgs/JointState computed from a first-order lag of the latest
std_msgs/Float64MultiArray command. TF for the full robot comes from
robot_state_publisher (started by the ROS2 backend when `rviz.enabled=true`).

This module is internal: users enable it via
``backend_kwargs={"dev_fake_sim": {...}}`` on the ROS2 backend and never
import rclpy themselves.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass

import numpy as np

from robodeploy.ros2 import Ros2NodeAdapter


@dataclass(frozen=True)
class FakeJointPosSimConfig:
    robot_ns: str = "/robot0"
    joint_states_topic: str = "joint_states"
    joint_pos_cmd_topic: str = "joint_position_commands"
    joint_names: tuple[str, ...] = tuple(f"joint{i}" for i in range(1, 8))
    base_frame: str = "base_link"
    ee_frame: str = "ee_link"
    publish_hz: float = 50.0
    follow_tau_s: float = 0.15  # first-order lag time constant
    # Optional per-joint max velocity (rad/s) to cap motion between publishes.
    max_joint_velocity: tuple[float, ...] | None = None


class FakeJointPosSim(Ros2NodeAdapter):
    def __init__(self, cfg: FakeJointPosSimConfig) -> None:
        super().__init__()
        self._cfg = cfg
        self._ns = (cfg.robot_ns or "").rstrip("/")
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.node_name = "robodeploy_fake_jointpos_sim"

        dof = len(cfg.joint_names)
        self._q = np.zeros(dof, dtype=np.float64)
        self._q_cmd = np.zeros(dof, dtype=np.float64)
        self._last_tick_s = time.perf_counter()

        self._pub_thread: threading.Thread | None = None

    def _on_node_ready(self, node) -> None:
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Float64MultiArray

        self._JointState = JointState
        js_topic = f"{self._ns}/{self._cfg.joint_states_topic}" if self._ns else f"/{self._cfg.joint_states_topic}"
        cmd_topic = f"{self._ns}/{self._cfg.joint_pos_cmd_topic}" if self._ns else f"/{self._cfg.joint_pos_cmd_topic}"

        self._js_pub = node.create_publisher(JointState, js_topic, 10)
        node.create_subscription(Float64MultiArray, cmd_topic, self._on_cmd, 10)

        self._stop.clear()
        self._pub_thread = threading.Thread(target=self._publish_loop, daemon=True)
        self._pub_thread.start()

    def _on_node_stopping(self, node) -> None:
        del node
        self._stop.set()
        if self._pub_thread is not None:
            try:
                self._pub_thread.join(timeout=2.0)
            except Exception:
                pass
            self._pub_thread = None

    def _on_cmd(self, msg) -> None:
        data = np.asarray(list(msg.data), dtype=np.float64).reshape(-1)
        with self._lock:
            n = min(self._q_cmd.shape[0], data.shape[0])
            self._q_cmd[:n] = data[:n]

    def _step_dynamics(self, dt: float) -> None:
        tau = max(1e-3, float(self._cfg.follow_tau_s))
        alpha = 1.0 - math.exp(-float(dt) / tau)
        with self._lock:
            dq = alpha * (self._q_cmd - self._q)
            vmax = self._cfg.max_joint_velocity
            if vmax is not None and len(vmax) == dq.shape[0]:
                cap = np.asarray(vmax, dtype=np.float64) * float(dt)
                dq = np.minimum(np.maximum(dq, -cap), cap)
            self._q[:] = self._q + dq

    def _publish_loop(self) -> None:
        period_s = 1.0 / max(1.0, float(self._cfg.publish_hz))
        next_deadline = time.perf_counter()
        while not self._stop.is_set():
            now_tick = time.perf_counter()
            dt = max(0.0, now_tick - self._last_tick_s)
            self._last_tick_s = now_tick

            self._step_dynamics(dt)
            self._publish_joint_state()

            next_deadline += period_s
            sleep_s = next_deadline - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_deadline = time.perf_counter()

    def _publish_joint_state(self) -> None:
        if self._node is None:
            return
        msg = self._JointState()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.name = list(self._cfg.joint_names)
        with self._lock:
            msg.position = [float(x) for x in self._q.tolist()]
        self._js_pub.publish(msg)
