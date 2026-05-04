"""SO-101 follower — thin ``RobotDescription`` backed by bundled URDF.

Joint names, limits, and DoF are parsed from the URDF via ``URDFRobotDescription``.
For MuJoCo, add MJCF under ``assets/mjcf/`` (or ``asset_overrides``). For Isaac Sim,
URDF import works; optional USD can live under ``assets/usd/``.

Mesh STLs live next to the URDF under ``assets/urdf/assets/``. If any STL is missing,
:func:`asset_path` returns a generated URDF with box placeholders so MuJoCo and RViz
still load.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from robodeploy.behavior import BehaviorProfile
from robodeploy.core.registry import register_robot
from robodeploy.core.spaces import AssetFormat
from robodeploy.description.base import URDFRobotDescription

from robodeploy.description.so101._urdf_assets import resolve_urdf_with_mesh_fallback


@register_robot("so101")
class SO101Description(URDFRobotDescription):
    """SO-101 follower — URDF is the source of truth for kinematics and limits."""

    real_controller_name = "so101_feetech"

    def __init__(self) -> None:
        urdf = Path(__file__).resolve().parent / "assets" / "urdf" / "so101.urdf"
        super().__init__(
            urdf,
            ee_link_name="gripper",
            display_name="SO-101 (follower)",
            home_qpos=np.zeros(6, dtype=np.float64),
            joint_order=["1", "2", "3", "4", "5", "6"],
        )
        # Canonical URDF root link is `base` (not ROS default `base_link`).
        self.ros_base_link_name = "base"
        self.ros2_preset_name = "generic_joint_position"

    def default_behavior_profile(self) -> BehaviorProfile:
        """SO-101 CAD meshes / inertials: softer tracking and extra damping by default."""
        return BehaviorProfile(preset="smooth")

    def asset_path(self, fmt: AssetFormat, variant: str = "default") -> Path:
        if fmt == AssetFormat.URDF:
            return resolve_urdf_with_mesh_fallback(self._urdf_path)
        raise FileNotFoundError(
            f"{type(self).__name__} bundles URDF only ({self._urdf_path}). "
            f"Requested {fmt.value} (variant={variant!r}). "
            "Add MJCF under description/so101/assets/mjcf/ for MuJoCo, or USD under assets/usd/ for Isaac, "
            "or pass backend config asset_overrides."
        )
