"""
ROS2GazeboBackend — Gazebo Harmonic simulation via ROS 2 transport.

This backend is simulated (`is_real = False`) and owns Gazebo/bridge/controller
process lifecycles. It reuses the ROS 2 controller adapters and sensor plumbing
from the ROS2RealBackend transport implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from robodeploy.core.registry import register_backend
from robodeploy.core.spaces import AssetFormat

from robodeploy.backends.real.ros2.backend import ROS2RealBackend

if TYPE_CHECKING:
    from robodeploy.core.interfaces.sensor import ISensor
    from robodeploy.core.robot import Robot
    from robodeploy.core.types import SceneSpec


@register_backend("ros2_gazebo")
class ROS2GazeboBackend(ROS2RealBackend):
    """Gazebo-simulated backend using ROS2 transport."""

    is_real = False
    sensor_backend_name = "gazebo"

    def initialize_multi(self, robots: list["Robot"], scene: "SceneSpec", shared_sensors: list["ISensor"]) -> None:  # type: ignore[override]
        sim_cfg = self.config.get("sim", None)
        if not isinstance(sim_cfg, dict) or str(sim_cfg.get("kind", "")).lower() != "gazebo":
            raise ValueError("ROS2GazeboBackend requires config.sim.kind == 'gazebo'.")

        from robodeploy.backends.real.ros2.sim_launchers.gazebo import GazeboLaunchConfig, GazeboLauncher
        from robodeploy.backends.real.ros2.sim_launchers.ros_gz_bridge import image_bridge_rules
        from robodeploy.backends.sim.gazebo.scene_builder import GazeboSceneBuilder

        bridge_rules = [*tuple(sim_cfg.get("bridge_rules", ()) or ())]
        wait_for_topics = [*tuple(sim_cfg.get("wait_for_topics", ()) or ())]
        for robot in robots:
            sensors_cfg = self.config.get(f"{robot.robot_id}.sensors", None)
            topics = self._image_topics_from_sensor_config(robot.robot_id, sensors_cfg)
            bridge_rules.extend(image_bridge_rules(*topics))
            wait_for_topics.extend(topics)
            for sensor in robot.sensors:
                topics = self._image_topics_from_sensor_object(robot.robot_id, sensor)
                bridge_rules.extend(image_bridge_rules(*topics))
                wait_for_topics.extend(topics)
        for sensor in shared_sensors:
            topics = self._image_topics_from_sensor_object(None, sensor)
            bridge_rules.extend(image_bridge_rules(*topics))
            wait_for_topics.extend(topics)

        robot_urdf = sim_cfg.get("robot_urdf")
        if not robot_urdf and robots:
            try:
                robot_urdf = str(robots[0].description.asset_path(AssetFormat.URDF))
            except Exception:
                robot_urdf = None
        self._generated_world_path = None
        world_path = sim_cfg.get("world")
        if not world_path:
            world_path = str(GazeboSceneBuilder().write_temp_world(scene.to_world()))
            self._generated_world_path = world_path

        self._sim_launcher = GazeboLauncher(
            GazeboLaunchConfig(
                world=str(world_path or ""),
                headless=bool(sim_cfg.get("headless", False)),
                robot_urdf=str(robot_urdf) if robot_urdf else None,
                robot_name=str(sim_cfg.get("robot_name", robots[0].robot_id if robots else "robot0")),
                controllers_to_spawn=tuple(sim_cfg.get("controllers_to_spawn", ()) or ()),
                wait_for_topics=tuple(dict.fromkeys(wait_for_topics)),
                bridge_rules=tuple(dict.fromkeys(bridge_rules)),
                readiness_timeout_s=float(sim_cfg.get("readiness_timeout_s", 15.0)),
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

    def _close_impl(self) -> None:
        try:
            super()._close_impl()
        finally:
            generated_world = getattr(self, "_generated_world_path", None)
            if generated_world:
                try:
                    Path(str(generated_world)).unlink(missing_ok=True)
                except Exception:
                    pass
                self._generated_world_path = None

    @staticmethod
    def _qualify_topic(namespace: str | None, topic: str | None) -> str | None:
        raw = str(topic or "").strip()
        if not raw:
            return None
        if raw.startswith("/"):
            return raw
        ns = str(namespace or "").strip().rstrip("/")
        if not ns:
            return f"/{raw}"
        return f"{ns}/{raw.lstrip('/')}"

    @classmethod
    def _image_topics_from_sensor_config(cls, robot_id: str, sensors_cfg) -> list[str]:
        if not isinstance(sensors_cfg, list):
            return []
        out: list[str] = []
        namespace = f"/{robot_id}"
        for item in sensors_cfg:
            if not isinstance(item, dict):
                continue
            for key in ("rgb", "depth"):
                topic = cls._qualify_topic(namespace, item.get(key))
                if topic:
                    out.append(topic)
        return list(dict.fromkeys(out))

    @classmethod
    def _image_topics_from_sensor_object(cls, robot_id: str | None, sensor: "ISensor") -> list[str]:
        cfg = dict(getattr(sensor, "config", {}) or {})
        namespace = cfg.get("namespace")
        if not isinstance(namespace, str) or not namespace.strip():
            namespace = f"/{robot_id}" if robot_id else ""
        out: list[str] = []
        for key in ("rgb", "depth"):
            topic = cls._qualify_topic(namespace, cfg.get(key))
            if topic:
                out.append(topic)
        return list(dict.fromkeys(out))

