"""
ROS2GazeboBackend — Gazebo Harmonic simulation via ROS 2 transport.

This backend is simulated (`is_real = False`) and owns Gazebo/bridge/controller
process lifecycles. It reuses the ROS 2 controller adapters and sensor plumbing
from the ROS2RealBackend transport implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from robodeploy.core.registry import register_backend

from robodeploy.backends.real.ros2.backend import ROS2RealBackend

if TYPE_CHECKING:
    from robodeploy.core.interfaces.sensor import ISensor
    from robodeploy.core.robot import Robot
    from robodeploy.core.types import SceneSpec


@register_backend("ros2_gazebo")
class ROS2GazeboBackend(ROS2RealBackend):
    """Gazebo-simulated backend using ROS2 transport."""

    is_real = False

    def initialize_multi(self, robots: list["Robot"], scene: "SceneSpec", shared_sensors: list["ISensor"]) -> None:  # type: ignore[override]
        sim_cfg = self.config.get("sim", None)
        if not isinstance(sim_cfg, dict) or str(sim_cfg.get("kind", "")).lower() != "gazebo":
            raise ValueError("ROS2GazeboBackend requires config.sim.kind == 'gazebo'.")

        from robodeploy.backends.real.ros2.sim_launchers.gazebo import GazeboLaunchConfig, GazeboLauncher

        self._sim_launcher = GazeboLauncher(
            GazeboLaunchConfig(
                world=str(sim_cfg.get("world", "")),
                headless=bool(sim_cfg.get("headless", False)),
                robot_urdf=str(sim_cfg.get("robot_urdf")) if sim_cfg.get("robot_urdf") else None,
                robot_name=str(sim_cfg.get("robot_name", "robot0")),
                controllers_to_spawn=tuple(sim_cfg.get("controllers_to_spawn", ()) or ()),
                wait_for_topics=tuple(sim_cfg.get("wait_for_topics", ()) or ()),
                bridge_rules=tuple(sim_cfg.get("bridge_rules", ()) or ()),
            )
        )
        self._sim_launcher.start()

        # GazeboLauncher will start robot_state_publisher when robot_urdf is provided.
        # Prevent transport backend from also starting one from description URDF.
        rviz_cfg = self.config.get("rviz", None)
        if isinstance(rviz_cfg, dict) and rviz_cfg.get("enabled", False):
            rviz_cfg = dict(rviz_cfg)
            rviz_cfg.setdefault("launch_robot_state_publisher", False)
            self.config["rviz"] = rviz_cfg

        return super().initialize_multi(robots, scene, shared_sensors)

