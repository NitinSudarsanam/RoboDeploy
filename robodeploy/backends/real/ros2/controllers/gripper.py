"""ROS2 gripper controller adapter."""

from __future__ import annotations

import threading
from typing import Literal, Optional

import numpy as np

from robodeploy.backends.real.common import Commander
from robodeploy.core.types import Action

from .base import ControllerConfig, register_controller
from robodeploy.ros2 import Ros2NodeAdapter


class GripperControllerAdapter(Ros2NodeAdapter):
    controller_type = "gripper"
    supported_action_spaces: list = []

    def __init__(self, cfg: ControllerConfig, backend_config: Optional[dict] = None) -> None:
        super().__init__()
        self._robot_id = cfg.robot_id
        self._ns = (cfg.namespace or "").rstrip("/")
        self._cmd_topic = cfg.cmd_topic or "gripper_controller/command"
        self._command_type: Literal["gripper_command", "float"] = str(
            (backend_config or {}).get("gripper_command_type", "gripper_command")
        )  # type: ignore[assignment]
        self._open_width = float((backend_config or {}).get("gripper_open_width", 0.08))
        self._close_width = float((backend_config or {}).get("gripper_close_width", 0.0))
        self._cmd_hz = float(cfg.command_hz or 0.0)
        self._commander = Commander(
            self._publish_gripper,
            min_period_s=(1.0 / self._cmd_hz) if self._cmd_hz > 0 else 0.0,
        )
        self._lock = threading.Lock()
        self.node_name = f"robodeploy_gripper_{self._robot_id}"

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
        return []

    def _on_node_ready(self, node) -> None:
        topic = f"{self._ns}/{self._cmd_topic}" if self._ns else f"/{self._cmd_topic}"
        if self._command_type == "float":
            from std_msgs.msg import Float64

            self._cmd_msg_type = Float64
            self._cmd_pub = node.create_publisher(Float64, topic, 10)
        else:
            from control_msgs.msg import GripperCommand as GripperCommandMsg

            self._cmd_msg_type = GripperCommandMsg
            self._cmd_pub = node.create_publisher(GripperCommandMsg, topic, 10)

    def _on_node_stopping(self, node) -> None:
        del node
        self._cmd_pub = None

    def _publish_gripper(self, value: float) -> None:
        if self._command_type == "float":
            msg = self._cmd_msg_type()
            msg.data = float(value)
            self._cmd_pub.publish(msg)
            return
        msg = self._cmd_msg_type()
        msg.position = float(value)
        msg.max_effort = float(getattr(self, "_max_effort", 20.0))
        self._cmd_pub.publish(msg)

    def send_action(self, action: Action) -> None:
        if action.gripper is None:
            raise ValueError("GripperControllerAdapter expected action.gripper")
        g = float(np.clip(np.asarray(action.gripper, dtype=np.float64).reshape(-1)[0], 0.0, 1.0))
        width = self._open_width + (self._close_width - self._open_width) * g
        self._commander.send(width)

    def get_obs(self):
        raise NotImplementedError("GripperControllerAdapter does not own observation state.")

    def get_diagnostics(self) -> dict:
        return {
            "robot_id": self._robot_id,
            "controller_type": self.controller_type,
            "command_count": self._commander.record.count,
        }


@register_controller("gripper")
def create_gripper_adapter(cfg: ControllerConfig, backend_config: dict):
    return GripperControllerAdapter(cfg, backend_config)
