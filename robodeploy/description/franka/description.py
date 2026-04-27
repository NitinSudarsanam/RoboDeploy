"""
FrankaDescription — robot description for the Franka Emika Panda.

Covers the 7-DOF arm. Gripper (fingers) are handled separately by the
backend via the gripper field in Action and Observation.

Asset layout:
  description/franka/assets/
    urdf/panda.urdf          ← canonical source (symlink to mujoco_menagerie or copy)
    mjcf/panda.xml           ← hand-tuned MuJoCo physics
    usd/panda.usd            ← auto-converted for IsaacLab (generated on first use)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from robodeploy.core.registry import register_robot
from robodeploy.core.spaces   import AssetFormat
from robodeploy.description.base import RobotDescription


@register_robot("franka")
class FrankaDescription(RobotDescription):
    """Franka Emika Panda, 7-DOF arm."""

    # ------------------------------------------------------------------
    # Static robot definition
    # ------------------------------------------------------------------

    dof = 7
    display_name = "Franka Emika Panda"
    ee_link_name = "panda_hand"
    ros2_preset_name = "franka_jtc"

    joint_names = [
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
    ]

    # Franka Panda joint position limits (radians) from the official URDF.
    joint_position_limits = np.array([
        [-2.8973,  2.8973],   # joint1
        [-1.7628,  1.7628],   # joint2
        [-2.8973,  2.8973],   # joint3
        [-3.0718, -0.0698],   # joint4
        [-2.8973,  2.8973],   # joint5
        [-0.0175,  3.7525],   # joint6
        [-2.8973,  2.8973],   # joint7
    ])

    # Maximum joint velocities (rad/s) from Franka documentation.
    joint_velocity_limits = np.array([2.175, 2.175, 2.175, 2.175, 2.610, 2.610, 2.610])

    # Maximum joint torques (N·m) from Franka documentation.
    joint_torque_limits = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])

    # Default home configuration (same as Franka's "ready" pose).
    home_qpos = np.array([0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854])

    # ------------------------------------------------------------------
    # Asset resolution
    # ------------------------------------------------------------------

    def asset_path(self, fmt: AssetFormat, variant: str = "default") -> Path:
        """Return path to the Franka asset in the requested format.

        MJCF is served directly from the bundled hand-tuned file.
        URDF is served from the assets/urdf/ directory.
        USD is auto-converted from URDF on first call and cached.

        Args:
            fmt: Desired asset format.

        Returns:
            Path to the asset file.

        Raises:
            FileNotFoundError: If the asset or conversion tool is unavailable.
        """
        del variant
        assets = Path(__file__).parent / "assets"

        if fmt == AssetFormat.MJCF:
            path = assets / "mjcf" / "panda.xml"
            if not path.exists():
                raise FileNotFoundError(
                    f"Franka MJCF not found at {path}.\n"
                    "Copy panda.xml from mujoco_menagerie/franka_emika_panda/ "
                    "into description/franka/assets/mjcf/"
                )
            return path

        if fmt == AssetFormat.URDF:
            path = assets / "urdf" / "panda.urdf"
            if not path.exists():
                raise FileNotFoundError(
                    f"Franka URDF not found at {path}.\n"
                    "Download the official Franka URDF and place it at "
                    "description/franka/assets/urdf/panda.urdf"
                )
            return path

        if fmt == AssetFormat.USD:
            path = assets / "usd" / "panda.usd"
            if not path.exists():
                self._convert_urdf_to_usd(
                    urdf_path=self.asset_path(AssetFormat.URDF),
                    out_path=path,
                )
            return path

        raise ValueError(f"Unsupported asset format: {fmt}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _convert_urdf_to_usd(self, urdf_path: Path, out_path: Path) -> None:
        """Auto-convert URDF to USD using Isaac Lab's converter.

        This is called lazily the first time USD is requested.
        Requires Isaac Lab to be installed.

        Args:
            urdf_path: Path to the source URDF.
            out_path:  Destination USD path.

        Raises:
            ImportError: If Isaac Lab is not installed.
        """
        try:
            from omni.isaac.lab.utils.assets import convert_urdf_to_usd  # type: ignore
        except ImportError:
            raise ImportError(
                "USD conversion requires Isaac Lab.\n"
                "Install Isaac Lab or manually place panda.usd at:\n"
                f"  {out_path}"
            )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        convert_urdf_to_usd(str(urdf_path), str(out_path))
