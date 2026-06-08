"""Minimal demo robot description for plugin entry-point smoke tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from robodeploy.core.registry import register_robot
from robodeploy.core.spaces import AssetFormat
from robodeploy.description.base import RobotDescription


@register_robot("demo_arm")
class DemoArmDescription(RobotDescription):
    dof = 7
    display_name = "Demo Arm (plugin)"
    ee_link_name = "robot0/ee_link"
    joint_names = [f"robot0/joint{i}" for i in range(1, 8)]
    joint_position_limits = np.array([[-3.0, 3.0]] * 7)
    joint_velocity_limits = np.array([2.0] * 7)
    joint_torque_limits = np.array([50.0] * 7)
    home_qpos = np.zeros(7)

    def asset_path(self, fmt: AssetFormat, variant: str = "default") -> Path:
        del variant
        root = Path(__file__).resolve().parents[3]
        path = root / "robodeploy" / "description" / "kuka" / "assets" / "mjcf" / "kuka.xml"
        if fmt != AssetFormat.MJCF or not path.is_file():
            raise FileNotFoundError(f"Demo plugin MJCF not found at {path}")
        return path
