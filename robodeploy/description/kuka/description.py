"""
KukaDescription — minimal robot description for the demo KUKA arm.

MJCF is provided for MuJoCo simulation; URDF enables Gazebo + Pinocchio IK.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from robodeploy.core.registry import register_robot
from robodeploy.core.spaces import AssetFormat
from robodeploy.description.base import RobotDescription


@register_robot("kuka")
class KukaDescription(RobotDescription):
    # NOTE: demo arm modeled as 7-DOF in `assets/mjcf/kuka.xml`
    dof = 7
    display_name = "KUKA (demo)"
    ee_link_name = "robot0/ee_link"
    ros2_preset_name = "kuka_jtc"

    joint_names = [f"robot0/joint{i}" for i in range(1, 8)]

    # Conservative generic limits for the demo model (radians, rad/s, N·m).
    joint_position_limits = np.array([[-3.14, 3.14]] * dof, dtype=np.float64)
    joint_velocity_limits = np.array([2.0] * dof, dtype=np.float64)
    joint_torque_limits = np.array([50.0] * dof, dtype=np.float64)

    home_qpos = np.array([0.0, -0.6, 0.0, -1.8, 0.0, 1.2, 0.0], dtype=np.float64)

    def asset_path(self, fmt: AssetFormat, variant: str = "default") -> Path:
        del variant
        assets = Path(__file__).parent / "assets"

        if fmt == AssetFormat.MJCF:
            path = assets / "mjcf" / "kuka.xml"
            if not path.exists():
                raise FileNotFoundError(f"Kuka MJCF not found at {path}")
            return path

        if fmt == AssetFormat.URDF:
            path = assets / "urdf" / "kuka.urdf"
            if not path.exists():
                raise FileNotFoundError(f"Kuka URDF not found at {path}")
            return path

        raise FileNotFoundError(f"KukaDescription does not provide {fmt.value}.")

    def gazebo_sim_launch_config(self) -> dict | None:
        assets = Path(__file__).parent / "assets" / "urdf"
        return {
            "kind": "gazebo",
            "headless": False,
            "robot_urdf": str(assets / "kuka.urdf"),
            "robot_name": "robot0",
            "controllers_to_spawn": ["joint_state_broadcaster", "joint_trajectory_controller"],
            "wait_for_topics": ["/clock", "/joint_states"],
        }

    def gazebo_ros2_extra_config(self, robot_id: str) -> dict | None:
        return {
            f"{robot_id}.preset": "kuka_jtc",
            f"{robot_id}.controller": "joint_trajectory",
            # gz_ros2_control publishes at root namespace; use absolute topics.
            f"{robot_id}.joint_states_topic": "/joint_states",
            f"{robot_id}.joint_cmd_topic": "/joint_trajectory_controller/joint_trajectory",
        }

