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

    #: Optional ROS2 wiring preset name (data-only) used by the ROS2 backends when
    #: the user does not explicitly provide `robotX.preset`.
    #: See `robodeploy/backends/real/ros2/presets.py`.
    ros2_preset_name: str | None = None

    #: Base link name as used in URDF / TF for ROS2 backends (``backend_for_simulator``).
    ros_base_link_name: str = "base_link"

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
    # ROS / multi-simulator hooks (used by ``backend_for_simulator``)
    # ------------------------------------------------------------------

    def ros_transport_joint_names(self) -> list[str]:
        """Joint names as they appear on ``sensor_msgs/JointState`` (URDF leaf names).

        MuJoCo MJCF often uses a namespace prefix (``robot0/joint1``); ROS topics use
        ``joint1``. Default: strip the first path segment when ``joint_names`` entries
        contain ``/``, otherwise return names unchanged.
        """
        out: list[str] = []
        for jn in self.joint_names:
            if "/" in jn:
                out.append(jn.split("/", 1)[1])
            else:
                out.append(jn)
        return out

    def ros_ee_frame_id(self) -> str:
        """End-effector link id for ROS TF / EE topics (no namespace prefix)."""
        n = self.ee_link_name
        return n.split("/", 1)[1] if "/" in n else n

    def ros_base_frame_id(self) -> str:
        """Base link id for ROS TF (matches ``ros_base_link_name``)."""
        return str(self.ros_base_link_name)

    def gazebo_sim_launch_config(self) -> dict | None:
        """If returning a dict, merged into ``config['sim']`` for ``ROS2GazeboBackend``.

        Must include at least ``kind: "gazebo"`` and a ``world`` path when non-empty.
        Return ``None`` if this description does not define a default Gazebo layout
        (callers pass ``config_overrides`` with a ``sim`` mapping instead).
        """
        return None

    def gazebo_ros2_extra_config(self, robot_id: str) -> dict | None:
        """Extra ROS2 transport keys when using ``ROS2GazeboBackend`` (per-robot topics, sensors).

        Keys should use the ``{robot_id}.`` prefix (e.g. ``robot0.joint_states_topic``).
        """
        return None

    def mujoco_backend_extra_config(self) -> dict | None:
        """Extra ``MuJoCoBackend`` config merged after library defaults."""
        return None

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


class URDFRobotDescription(RobotDescription):
    """RobotDescription backed by a URDF XML file (canonical source).

    This class is meant to be used by library users. A user can either
    instantiate it directly, or subclass it and pass a fixed URDF path.
    """

    def __init__(
        self,
        urdf_path: str | Path,
        *,
        ee_link_name: str,
        display_name: str | None = None,
        home_qpos: np.ndarray | None = None,
        joint_order: list[str] | None = None,
        default_velocity_limit: float = 2.0,
        default_effort_limit: float = 50.0,
    ) -> None:
        self._urdf_path = Path(urdf_path)
        if not self._urdf_path.exists():
            raise FileNotFoundError(f"URDF not found: {self._urdf_path}")

        self.ee_link_name = ee_link_name

        parsed = self._parse_urdf(
            self._urdf_path,
            joint_order=joint_order,
            default_velocity_limit=default_velocity_limit,
            default_effort_limit=default_effort_limit,
        )
        self.joint_names = parsed["joint_names"]
        self.dof = len(self.joint_names)
        self.joint_position_limits = parsed["pos_limits"]
        self.joint_velocity_limits = parsed["vel_limits"]
        self.joint_torque_limits = parsed["eff_limits"]
        self.home_qpos = (
            np.asarray(home_qpos, dtype=np.float64)
            if home_qpos is not None
            else np.zeros(self.dof, dtype=np.float64)
        )
        if self.home_qpos.shape[0] != self.dof:
            raise ValueError("home_qpos must match dof.")

        self.display_name = display_name or parsed["robot_name"] or f"URDFRobot({self._urdf_path.name})"

    @classmethod
    def from_urdf(
        cls,
        path: str | Path,
        *,
        ee_link_name: str,
        display_name: str | None = None,
        home_qpos: np.ndarray | None = None,
        joint_order: list[str] | None = None,
    ) -> "URDFRobotDescription":
        return cls(
            path,
            ee_link_name=ee_link_name,
            display_name=display_name,
            home_qpos=home_qpos,
            joint_order=joint_order,
        )

    def asset_path(self, fmt: AssetFormat, variant: str = "default") -> Path:
        del variant
        if fmt == AssetFormat.URDF:
            return self._urdf_path
        raise FileNotFoundError(
            f"{type(self).__name__} provides only URDF by default. "
            f"Backend requested {fmt.value}. Provide an override or a derived asset."
        )

    @staticmethod
    def _parse_urdf(
        urdf_path: Path,
        *,
        joint_order: list[str] | None,
        default_velocity_limit: float,
        default_effort_limit: float,
    ) -> dict:
        import math
        import xml.etree.ElementTree as ET

        root = ET.parse(str(urdf_path)).getroot()
        robot_name = root.attrib.get("name", "")

        joints: list[dict] = []
        for joint in root.findall("joint"):
            jtype = joint.attrib.get("type", "fixed")
            if jtype == "fixed":
                continue
            name = joint.attrib.get("name", "")
            if not name:
                continue

            limit = joint.find("limit")
            lower = None
            upper = None
            vel = None
            eff = None
            if limit is not None:
                if "lower" in limit.attrib:
                    lower = float(limit.attrib["lower"])
                if "upper" in limit.attrib:
                    upper = float(limit.attrib["upper"])
                if "velocity" in limit.attrib:
                    vel = float(limit.attrib["velocity"])
                if "effort" in limit.attrib:
                    eff = float(limit.attrib["effort"])

            if jtype == "continuous":
                lower = lower if lower is not None else -math.pi
                upper = upper if upper is not None else math.pi
            else:
                lower = lower if lower is not None else -math.pi
                upper = upper if upper is not None else math.pi

            joints.append({
                "name": name,
                "lower": lower,
                "upper": upper,
                "vel": vel if vel is not None else default_velocity_limit,
                "eff": eff if eff is not None else default_effort_limit,
            })

        if joint_order is not None:
            by_name = {j["name"]: j for j in joints}
            ordered = []
            for name in joint_order:
                if name not in by_name:
                    raise KeyError(f"Joint '{name}' not found in URDF {urdf_path}")
                ordered.append(by_name[name])
            joints = ordered

        joint_names = [j["name"] for j in joints]
        dof = len(joint_names)
        pos_limits = (
            np.asarray([[j["lower"], j["upper"]] for j in joints], dtype=np.float64)
            if dof
            else np.zeros((0, 2), dtype=np.float64)
        )
        vel_limits = (
            np.asarray([j["vel"] for j in joints], dtype=np.float64)
            if dof
            else np.zeros((0,), dtype=np.float64)
        )
        eff_limits = (
            np.asarray([j["eff"] for j in joints], dtype=np.float64)
            if dof
            else np.zeros((0,), dtype=np.float64)
        )

        return {
            "robot_name": robot_name,
            "joint_names": joint_names,
            "pos_limits": pos_limits,
            "vel_limits": vel_limits,
            "eff_limits": eff_limits,
        }
