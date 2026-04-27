"""ROS2 backend interfaces and configuration.

This module keeps the ROS2 backend general by depending on a small controller
adapter interface. Robot-specific defaults live in data-only presets.

Controller adapter registration:
    - Implement `IControllerAdapter` for each controller family.
    - Register a factory with `@register_controller("joint_position")` etc.
    - Select per robot via `controller_by_robot_id` (or presets).

ROS graph mapping:
    - `namespace` -> ROS namespace prefix such as `/robot0`
    - `joint_states_topic` -> namespaced `sensor_msgs/JointState` topic
    - `joint_pos_cmd_topic` -> namespaced command topic for joint positions
    - `base_frame` / `ee_frame` -> TF frame ids used for EE pose lookup

Diagnostics convention:
    Controller adapters may expose a best-effort `get_diagnostics() -> dict`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .controllers.base import ControllerConfig, IControllerAdapter, make_controller, register_controller


@dataclass(frozen=True)
class Ros2BackendConfig:
    """Backend-wide ROS2 configuration."""

    # When set, limits ROS spinning thread CPU usage; kept conservative.
    spin_hz: float = 200.0

    # RViz publishing (optional)
    rviz_enabled: bool = False
    rviz_fixed_frame: str = "world"
    rviz_publish_hz: float = 10.0
    # When True and rviz_enabled, start robot_state_publisher from the robot URDF (first robot only).
    rviz_launch_robot_state_publisher: bool = True

    # Controller selection (controller family per robot).
    controller_by_robot_id: dict[str, str] = field(default_factory=dict)
    # Optional global/default command rate (Hz) for joint commands.
    command_hz: float = 0.0
    # Optional per-robot override for command rate (Hz).
    command_hz_by_robot_id: dict[str, float] = field(default_factory=dict)
__all__ = [
    "Ros2BackendConfig",
    "ControllerConfig",
    "IControllerAdapter",
    "register_controller",
    "make_controller",
]