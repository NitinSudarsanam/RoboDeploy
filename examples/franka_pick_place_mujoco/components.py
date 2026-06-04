"""Franka description aligned with bundled MuJoCo MJCF joint naming."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from robodeploy.core.registry import register_robot
from robodeploy.core.spaces import AssetFormat
from robodeploy.description.base import RobotDescription


@register_robot("example_franka_mujoco")
class ExampleFrankaMujocoDescription(RobotDescription):
    """Franka Panda using the demo MJCF names (robot0/joint*, robot0/act*)."""

    dof = 7
    display_name = "Franka (example MJCF)"
    ee_link_name = "robot0/ee_link"
    joint_names = [f"robot0/joint{i}" for i in range(1, 8)]
    home_qpos = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.8], dtype=np.float64)

    joint_position_limits = np.array([[-2.9, 2.9]] * dof, dtype=np.float64)
    joint_velocity_limits = np.array([2.0] * dof, dtype=np.float64)
    joint_torque_limits = np.array([50.0] * dof, dtype=np.float64)

    def asset_path(self, fmt: AssetFormat, variant: str = "default") -> Path:
        del variant
        path = Path(__file__).resolve().parents[2] / "robodeploy" / "description" / "franka" / "assets" / "mjcf" / "panda.xml"
        if fmt != AssetFormat.MJCF or not path.is_file():
            raise FileNotFoundError(f"Example Franka MJCF not found at {path}")
        return path
