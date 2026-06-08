"""ROS2 joint-effort controller adapter."""

from __future__ import annotations

import threading
from typing import Optional

import numpy as np

from robodeploy.backends.real.common import Commander
from robodeploy.core.spaces import ActionSpace
from robodeploy.core.types import Action

from .base import ControllerConfig, register_controller
from robodeploy.ros2 import Ros2NodeAdapter


class JointEffortControllerAdapter(Ros2NodeAdapter):
    controller_type = "joint_effort"
    supported_action_spaces = [ActionSpace.JOINT_TORQUE]

    def __init__(self, cfg: ControllerConfig, backend_config: Optional[dict] = None) -> None:
        super().__init__()
        self._robot_id = cfg.robot_id
        self._ns = (cfg.namespace or "").rstrip("/")
        self._cmd_topic = cfg.cmd_topic or "effort_controllers/commands"
        self._joint_names = list(cfg.joint_names) if cfg.joint_names else []
        self._backend_config = backend_config or {}
        self._cmd_hz = float(cfg.command_hz or 0.0)
        self._commander = Commander(
            self._publish_joint_efforts,
            min_period_s=(1.0 / self._cmd_hz) if self._cmd_hz > 0 else 0.0,
        )
        self._lock = threading.Lock()
        self.node_name = f"robodeploy_jointeff_{self._robot_id}"

    @property
    def robot_id(self) -> str:
        return self._robot_id

    @property
    def base_frame(self) -> str:
        return "base_link"

    @property
    def ee_frame(self) -> str:
        return "ee_link"

    @property
    def joint_names(self) -> list[str]:
        with self._lock:
            return list(self._joint_names)

    def _on_node_ready(self, node) -> None:
        from std_msgs.msg import Float64MultiArray

        self._cmd_msg_type = Float64MultiArray
        topic = f"{self._ns}/{self._cmd_topic}" if self._ns else f"/{self._cmd_topic}"
        self._cmd_pub = node.create_publisher(Float64MultiArray, topic, 10)

    def _on_node_stopping(self, node) -> None:
        del node
        self._cmd_pub = None

    def _publish_joint_efforts(self, efforts: np.ndarray) -> None:
        msg = self._cmd_msg_type()
        msg.data = [float(x) for x in np.asarray(efforts, dtype=np.float64).reshape(-1).tolist()]
        self._cmd_pub.publish(msg)

    def send_action(self, action: Action) -> None:
        if action.joint_torques is None:
            raise ValueError("JointEffortControllerAdapter expected action.joint_torques")
        tau = np.asarray(action.joint_torques, dtype=np.float64).reshape(-1)
        self._commander.send(tau)

    def get_obs(self):
        raise NotImplementedError("JointEffortControllerAdapter does not own observation state.")

    def get_diagnostics(self) -> dict:
        return {
            "robot_id": self._robot_id,
            "controller_type": self.controller_type,
            "command_count": self._commander.record.count,
        }


@register_controller("joint_effort")
def create_joint_effort_adapter(cfg: ControllerConfig, backend_config: dict):
    return JointEffortControllerAdapter(cfg, backend_config)
