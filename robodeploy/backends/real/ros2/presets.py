"""Data-only presets for ROS2 controller wiring.

These presets exist to minimize user configuration while keeping the ROS2 layer
robot-agnostic (no robot-specific Python).

Rules:
  - Presets are pure data (dicts), safe to override per robot.
  - `ROS2Backend` merges: defaults < preset < per-robot explicit keys.
"""

from __future__ import annotations

from typing import Any


# Keys are designed to map onto ControllerConfig fields + ROS2Backend per-robot config:
#   - controller_type (string, controller registry key)
#   - joint_states_topic
#   - joint_cmd_topic
#   - base_frame / ee_frame
#   - joint_names (optional)
PRESETS: dict[str, dict[str, Any]] = {
    # Generic baseline: joint position controller expecting Float64MultiArray.
    "generic_joint_position": {
        "controller_type": "joint_position",
        "joint_states_topic": "joint_states",
        "joint_cmd_topic": "joint_position_commands",
        "base_frame": "base_link",
        "ee_frame": "ee_link",
    },
    # Common ros2_control naming for JointTrajectoryController.
    "generic_joint_trajectory": {
        "controller_type": "joint_trajectory",
        "joint_states_topic": "joint_states",
        "joint_cmd_topic": "joint_trajectory_controller/joint_trajectory",
        "base_frame": "base_link",
        "ee_frame": "ee_link",
    },
    # Franka defaults (data-only; works with typical franka_ros2 + ros2_control setups).
    "franka_jtc": {
        "controller_type": "joint_trajectory",
        "joint_states_topic": "joint_states",
        "joint_cmd_topic": "joint_trajectory_controller/joint_trajectory",
        "base_frame": "panda_link0",
        "ee_frame": "panda_hand",
        "joint_names": [
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
        ],
    },
    # Kuka example baseline (user should override to match their bringup).
    "kuka_jtc": {
        "controller_type": "joint_trajectory",
        "joint_states_topic": "joint_states",
        "joint_cmd_topic": "joint_trajectory_controller/joint_trajectory",
        "base_frame": "base_link",
        "ee_frame": "ee_link",
        "joint_names": [f"joint{i}" for i in range(1, 8)],
    },
    # UR arm baseline (user should override joint_names/frames to match URDF).
    "ur_jtc": {
        "controller_type": "joint_trajectory",
        "joint_states_topic": "joint_states",
        "joint_cmd_topic": "joint_trajectory_controller/joint_trajectory",
        "base_frame": "base_link",
        "ee_frame": "tool0",
    },
}

