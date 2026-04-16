"""
RobotDescription — static, runtime-free definition of a robot.

A RobotDescription answers the question "what robot am I?" without
starting a simulator or opening a hardware connection. It is pure data
and pure math: joint names, limits, home configuration, asset file paths,
and access to the KinematicsSolver.

Design rules:
  - No physics engine imports.
  - No ROS2 imports.
  - No JAX / torch imports.
  - Instantiation must be fast (no file I/O except lazy asset path resolution).

Every concrete robot (Franka, UR5, Spot) subclasses RobotDescription and:
  1. Fills in the class-level attributes (dof, joint_names, etc.).
  2. Puts its assets under description/<name>/assets/<format>/.
  3. Registers itself with @register_robot("<name>").

The description is passed to:
  - IBackend.initialize()  — to load the robot into the physics engine.
  - KinematicsSolver()     — to compute FK/IK without a running sim.
  - SafetyFilter()         — to know joint limits for action clamping.

Adding a new robot:
  1. Create description/<name>/description.py.
  2. Subclass RobotDescription, fill in all class attributes.
  3. Place URDF under description/<name>/assets/urdf/<name>.urdf.
  4. Optionally provide hand-tuned MJCF and USD in the same assets/ folder.
  5. Decorate with @register_robot("<name>").
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from robodeploy.core.spaces import AssetFormat


class RobotDescription(ABC):
    """Static robot definition. Subclass one per robot type."""

    # ------------------------------------------------------------------
    # Class-level attributes — fill these in every subclass
    # ------------------------------------------------------------------

    #: Number of controlled degrees of freedom (arm only, excluding gripper).
    dof: int

    #: Ordered list of joint names matching the URDF order.
    joint_names: list[str]

    #: Joint position limits, shape [dof, 2] — [[min0, max0], [min1, max1], ...].
    #: Values in radians.
    joint_position_limits: np.ndarray    # [dof, 2]

    #: Joint velocity limits, shape [dof] — max absolute velocity in rad/s.
    joint_velocity_limits: np.ndarray    # [dof]

    #: Joint torque limits, shape [dof] — max absolute torque in N·m.
    joint_torque_limits: np.ndarray      # [dof]

    #: Home (rest) joint configuration in radians, shape [dof].
    home_qpos: np.ndarray                # [dof]

    #: Name of the end-effector link as it appears in the URDF/MJCF.
    ee_link_name: str

    #: Display name for logging and UI.
    display_name: str

    # ------------------------------------------------------------------
    # Asset resolution (implemented by subclasses)
    # ------------------------------------------------------------------

    @abstractmethod
    def asset_path(self, fmt: AssetFormat, variant: str = "default") -> Path:
        """Return the filesystem path to the robot asset in the requested format.

        Real and sim URDFs are often not identical:
          - Sim URDF needs friction, damping, and inertia tags tuned for the
            physics engine, which may cause issues with real IK solvers.
          - Real URDF may lack the dense collision meshes required by sim.

        Use variant to select the correct asset for each context:
            asset_path(AssetFormat.URDF, variant="sim")   # franka_sim.urdf
            asset_path(AssetFormat.URDF, variant="real")  # franka_real.urdf
            asset_path(AssetFormat.URDF)                  # franka.urdf (shared default)

        If the exact format is not available, the implementation should
        attempt auto-conversion from the canonical URDF source and cache
        the result. Raises FileNotFoundError if conversion is not possible.

        Args:
            fmt:     Desired asset format (URDF, MJCF, or USD).
            variant: "default" uses the shared canonical asset. "sim" and
                     "real" select physics-tuned or hardware-calibrated
                     variants when the robot description provides them.
                     Subclasses may define additional variants.

        Returns:
            Path: Absolute path to the asset file.

        Raises:
            FileNotFoundError: If the asset cannot be found or converted.
        """
        ...

    # ------------------------------------------------------------------
    # Kinematics access
    # ------------------------------------------------------------------

    def get_kinematics_solver(self):
        """Return a KinematicsSolver initialised for this robot.

        Lazy-initialised on first call; cached for subsequent calls.
        Requires the URDF asset to be present.

        Returns:
            KinematicsSolver: FK/IK solver for this robot.
        """
        if not hasattr(self, "_kinematics_solver"):
            from robodeploy.kinematics.solver import KinematicsSolver
            self._kinematics_solver = KinematicsSolver(self)
        return self._kinematics_solver

    def get_safety_filter(self):
        """Return a SafetyFilter initialised with this robot's limits.

        Returns:
            SafetyFilter: Action validation and clamping for this robot.
        """
        if not hasattr(self, "_safety_filter"):
            from robodeploy.kinematics.safety import SafetyFilter
            self._safety_filter = SafetyFilter(self)
        return self._safety_filter

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def assets_dir(self) -> Path:
        """Root assets directory for this robot description.

        Resolves to description/<name>/assets/ relative to this file.
        Subclasses may override for non-standard layouts.
        """
        return Path(__file__).parent / type(self).__name__.lower().replace("description", "") / "assets"

    def __repr__(self) -> str:
        return f"{type(self).__name__}(dof={self.dof}, ee={self.ee_link_name})"
