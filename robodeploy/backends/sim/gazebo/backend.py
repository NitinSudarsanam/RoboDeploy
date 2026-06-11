"""
ROS2GazeboBackend — Gazebo Harmonic simulation via ROS 2 transport.

This backend is simulated (`is_real = False`) and owns Gazebo/bridge/controller
process lifecycles. It reuses the ROS 2 controller adapters and sensor plumbing
from the ROS2RealBackend transport implementation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from robodeploy.backends.sim.gazebo.contact import GazeboContactMonitor
from robodeploy.core.registry import register_backend
from robodeploy.core.spaces import AssetFormat
from robodeploy.core.types import Action, Observation

from robodeploy.backends.real.ros2.backend import ROS2RealBackend

if TYPE_CHECKING:
    from robodeploy.core.interfaces.sensor import ISensor
    from robodeploy.core.robot import Robot
    from robodeploy.core.types import SceneSpec

logger = logging.getLogger(__name__)


@register_backend("ros2_gazebo")
class ROS2GazeboBackend(ROS2RealBackend):
    """Gazebo-simulated backend using ROS2 transport."""

    is_real = False
    sensor_backend_name = "gazebo"

    def initialize_multi(self, robots: list["Robot"], scene: "SceneSpec", shared_sensors: list["ISensor"]) -> None:  # type: ignore[override]
        if len(robots) > 1:
            raise ValueError(
                "Gazebo backend supports single robot until multi-spawn is implemented."
            )
        from robodeploy.backends.simulator import merge_simulator_config

        sim_cfg = dict(self.config.get("sim", None) or {})
        if str(sim_cfg.get("kind", "")).lower() != "gazebo":
            raise ValueError("ROS2GazeboBackend requires config.sim.kind == 'gazebo'.")

        for robot in robots:
            launch_cfg = robot.description.gazebo_sim_launch_config()
            if isinstance(launch_cfg, dict):
                sim_cfg = merge_simulator_config(launch_cfg, sim_cfg)
            extra = robot.description.gazebo_ros2_extra_config(robot.robot_id)
            if isinstance(extra, dict):
                self.config = merge_simulator_config(self.config, extra)

        from robodeploy.backends.real.ros2.sim_launchers.gazebo import GazeboLaunchConfig, GazeboLauncher
        from robodeploy.backends.real.ros2.sim_launchers.ros_gz_bridge import (
            camera_info_bridge_rules,
            image_bridge_rules,
            imu_bridge_rules,
            wrench_bridge_rules,
        )
        from robodeploy.backends.sim.gazebo.prop_pose_sync import PropPoseSyncer
        from robodeploy.backends.sim.gazebo.urdf_sensors import (
            patch_urdf_controller_yaml,
            write_urdf_with_sensors,
        )
        from robodeploy.backends.sim.gazebo.scene_builder import GazeboSceneBuilder

        bridge_rules = [*tuple(sim_cfg.get("bridge_rules", ()) or ())]
        wait_for_topics = [*tuple(sim_cfg.get("wait_for_topics", ()) or ())]
        all_robot_sensors: list["ISensor"] = []
        for robot in robots:
            all_robot_sensors.extend(list(robot.sensors))
            sensors_cfg = self.config.get(f"{robot.robot_id}.sensors", None)
            topics = self._image_topics_from_sensor_config(robot.robot_id, sensors_cfg)
            bridge_rules.extend(image_bridge_rules(*topics))
            wait_for_topics.extend(topics)
            for sensor in robot.sensors:
                topics = self._image_topics_from_sensor_object(robot.robot_id, sensor)
                bridge_rules.extend(image_bridge_rules(*topics))
                wait_for_topics.extend(topics)
                info_topics = self._camera_info_topics_from_sensor_object(robot.robot_id, sensor)
                bridge_rules.extend(camera_info_bridge_rules(*info_topics))
                wait_for_topics.extend(info_topics)
                wrench_topic = self._wrench_topic_from_sensor_object(robot.robot_id, sensor)
                if wrench_topic:
                    bridge_rules.extend(wrench_bridge_rules(wrench_topic))
                    wait_for_topics.append(wrench_topic)
                imu_topic = self._imu_topic_from_sensor_object(robot.robot_id, sensor)
                if imu_topic:
                    bridge_rules.extend(imu_bridge_rules(imu_topic))
                    wait_for_topics.append(imu_topic)
        for sensor in shared_sensors:
            topics = self._image_topics_from_sensor_object(None, sensor)
            bridge_rules.extend(image_bridge_rules(*topics))
            wait_for_topics.extend(topics)
            info_topics = self._camera_info_topics_from_sensor_object(None, sensor)
            bridge_rules.extend(camera_info_bridge_rules(*info_topics))
            wait_for_topics.extend(info_topics)
            wrench_topic = self._wrench_topic_from_sensor_object(None, sensor)
            if wrench_topic:
                bridge_rules.extend(wrench_bridge_rules(wrench_topic))
                wait_for_topics.append(wrench_topic)
            imu_topic = self._imu_topic_from_sensor_object(None, sensor)
            if imu_topic:
                bridge_rules.extend(imu_bridge_rules(imu_topic))
                wait_for_topics.append(imu_topic)

        robot_urdf = sim_cfg.get("robot_urdf")
        if not robot_urdf and robots:
            try:
                robot_urdf = str(robots[0].description.asset_path(AssetFormat.URDF))
            except Exception:
                robot_urdf = None
        self._generated_robot_urdf_path = None
        if robot_urdf:
            try:
                if all_robot_sensors:
                    patched = write_urdf_with_sensors(robot_urdf, all_robot_sensors)
                else:
                    import os
                    import tempfile

                    text = patch_urdf_controller_yaml(Path(robot_urdf).read_text(encoding="utf-8"), robot_urdf)
                    fd, out = tempfile.mkstemp(prefix="robodeploy_gazebo_robot_", suffix=".urdf")
                    Path(out).write_text(text, encoding="utf-8")
                    os.close(fd)
                    patched = Path(out)
                self._generated_robot_urdf_path = str(patched)
                robot_urdf = str(patched)
            except Exception as exc:
                logger.warning("Failed to patch URDF for Gazebo sensors: %s", exc)
                if bool(sim_cfg.get("require_sensors", False)):
                    raise
        self._generated_world_path = None
        world_path = sim_cfg.get("world")
        world_spec = scene.to_world()
        if world_path:
            wp = Path(str(world_path))
            wname = GazeboLauncher._world_name_from_sdf(wp)
            scene_builder = GazeboSceneBuilder(world_name=wname)
        else:
            scene_builder = GazeboSceneBuilder()

        # Props must be present in the SDF for physics, contacts, and grasp pose sync.
        if not world_path or world_spec.props:
            world_path = str(scene_builder.write_temp_world(world_spec))
            self._generated_world_path = world_path

        world = world_spec
        self._gz_world_name = str(scene_builder.world_name)
        gz_node = sim_cfg.get("gz_transport_node")
        if gz_node is None:
            gz_node = self._default_gz_transport_node()
        self._gz_transport_node = gz_node
        self._prop_pose_syncer = PropPoseSyncer(gz_node=gz_node)
        self._pending_gz_prop_sync: set[str] = set()
        self._scene_prop_poses = {
            prop.name: (tuple(prop.position), tuple(prop.orientation))
            for prop in world.props
        }
        self._grasp_prop: str | None = None
        self._grasp_offset: tuple[float, float, float] = (0.0, 0.0, 0.03)
        self._grasp_mode: str = "follow"
        self._ee_link = str(
            robots[0].description.ee_link_name if robots else self.config.get("ee_link", "ee_link")
        )
        self._kinematics_solver = None
        if robots:
            try:
                self._kinematics_solver = robots[0].description.get_kinematics_solver()
            except Exception:
                self._kinematics_solver = None
        self._contact_monitor = GazeboContactMonitor()
        if gz_node is not None:
            self._contact_monitor.bind_transport(
                gz_node,
                topic=f"/world/{self._gz_world_name}/contacts",
            )

        expected_js_names: tuple[str, ...] = ()
        if robots:
            try:
                expected_js_names = tuple(robots[0].description.ros_transport_joint_names())
            except Exception:
                pass
        if self._kinematics_solver is not None:
            self.config["_kinematics_solver"] = self._kinematics_solver
        self._sim_launcher = GazeboLauncher(
            GazeboLaunchConfig(
                world=str(world_path or ""),
                headless=bool(sim_cfg.get("headless", False)),
                robot_urdf=str(robot_urdf) if robot_urdf else None,
                robot_name=str(sim_cfg.get("robot_name", robots[0].robot_id if robots else "robot0")),
                controllers_to_spawn=tuple(sim_cfg.get("controllers_to_spawn", ()) or ()),
                wait_for_topics=tuple(dict.fromkeys(wait_for_topics)),
                bridge_rules=tuple(dict.fromkeys(bridge_rules)),
                expected_joint_names=expected_js_names,
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

    def step_multi(self, actions: list[Action]) -> list[Observation]:
        out = super().step_multi(actions)
        self._sync_grasped_prop()
        self._flush_pending_gz_prop_sync()
        return out

    def get_prop_pose(self, name: str):
        poses = getattr(self, "_scene_prop_poses", {})
        if name not in poses:
            raise KeyError(f"Unknown Gazebo prop '{name}'.")
        return poses[name]

    def set_prop_pose(self, name: str, position, orientation) -> None:  # noqa: ANN001
        poses = getattr(self, "_scene_prop_poses", {})
        if name not in poses:
            raise KeyError(f"Unknown Gazebo prop '{name}'.")
        pos = tuple(float(v) for v in position)
        quat = tuple(float(v) for v in orientation)
        poses[name] = (pos, quat)
        self._pending_gz_prop_sync.add(str(name))

    def _flush_pending_gz_prop_sync(self) -> None:
        pending = getattr(self, "_pending_gz_prop_sync", None)
        syncer = getattr(self, "_prop_pose_syncer", None)
        world = getattr(self, "_gz_world_name", None)
        if not pending or syncer is None or not world:
            return
        poses = getattr(self, "_scene_prop_poses", {})
        for name in list(pending):
            if name not in poses:
                pending.discard(name)
                continue
            pos, quat = poses[name]
            ok = syncer.set_entity_pose(
                world_name=str(world),
                entity_name=str(name),
                position=pos,
                orientation=quat,
            )
            if not ok:
                logger.warning("Gazebo prop pose sync failed for '%s'", name)
            pending.discard(name)

    def _sync_gz_entity_pose(self, name: str, position, orientation) -> None:  # noqa: ANN001
        del name, position, orientation
        self._flush_pending_gz_prop_sync()

    def teleport_object(self, name: str, position: tuple[float, float, float]) -> None:
        _, quat = self.get_prop_pose(name)
        self.set_prop_pose(name, position, quat)

    def has_prop_contact(self, prop_name: str, *, other_body: str | None = None) -> bool:
        monitor = getattr(self, "_contact_monitor", None)
        if monitor is None:
            return False
        other = other_body or getattr(self, "_ee_link", None)
        return monitor.has_contact(prop_name, other)

    def set_grasp_prop(
        self,
        prop_name: str | None,
        *,
        offset: tuple[float, float, float] | None = None,
        mode: str = "follow",
    ) -> None:
        if mode not in ("follow",):
            raise NotImplementedError("Gazebo backend only supports grasp mode='follow'.")
        self._grasp_mode = str(mode)
        self._grasp_prop = str(prop_name) if prop_name else None
        if offset is not None:
            self._grasp_offset = (float(offset[0]), float(offset[1]), float(offset[2]))
        if self._grasp_prop:
            self._sync_grasped_prop()

    def _sync_grasped_prop(self) -> None:
        if not self._grasp_prop or self._grasp_mode != "follow":
            return
        if self._grasp_prop not in getattr(self, "_scene_prop_poses", {}):
            return
        ee_pos, ee_quat = self._get_ee_pose()
        pos = (
            float(ee_pos[0]) + self._grasp_offset[0],
            float(ee_pos[1]) + self._grasp_offset[1],
            float(ee_pos[2]) + self._grasp_offset[2],
        )
        _, quat = self.get_prop_pose(self._grasp_prop)
        poses = getattr(self, "_scene_prop_poses", {})
        poses[self._grasp_prop] = (pos, ee_quat if ee_quat is not None else quat)
        self._pending_gz_prop_sync.add(str(self._grasp_prop))

    def _get_ee_pose(self) -> tuple[np.ndarray, np.ndarray | None]:
        drivers = getattr(self, "_drivers", {})
        if not drivers:
            return np.zeros(3, dtype=np.float64), None
        driver = next(iter(drivers.values()))
        obs = driver.get_obs()
        ee_pos = np.asarray(obs.ee_position, dtype=np.float64).reshape(3)
        ee_quat = np.asarray(obs.ee_orientation, dtype=np.float64).reshape(4)
        if np.isfinite(ee_pos).all() and np.isfinite(ee_quat).all():
            return ee_pos, ee_quat
        solver = getattr(self, "_kinematics_solver", None)
        q = getattr(obs, "joint_positions", None)
        if solver is not None and q is not None:
            try:
                pos, quat = solver.fk(np.asarray(q, dtype=np.float64).reshape(-1))
                return np.asarray(pos, dtype=np.float64).reshape(3), np.asarray(quat, dtype=np.float64).reshape(4)
            except Exception:
                pass
        return ee_pos, ee_quat

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
            generated_urdf = getattr(self, "_generated_robot_urdf_path", None)
            if generated_urdf:
                try:
                    Path(str(generated_urdf)).unlink(missing_ok=True)
                except Exception:
                    pass
                self._generated_robot_urdf_path = None

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
    def _depth_topic_from_cfg(cls, namespace: str, cfg: dict) -> str | None:
        depth_val = cfg.get("depth")
        if depth_val in (False, None, "false"):
            return None
        if depth_val in (True, "true"):
            return cls._qualify_topic(namespace, cfg.get("depth_topic", "depth/image_raw"))
        return cls._qualify_topic(namespace, depth_val)

    @classmethod
    def _image_topics_from_sensor_config(cls, robot_id: str, sensors_cfg) -> list[str]:
        if not isinstance(sensors_cfg, list):
            return []
        out: list[str] = []
        namespace = f"/{robot_id}"
        for item in sensors_cfg:
            if not isinstance(item, dict):
                continue
            topic = cls._qualify_topic(namespace, item.get("rgb"))
            if topic:
                out.append(topic)
            depth_topic = cls._depth_topic_from_cfg(namespace, item)
            if depth_topic:
                out.append(depth_topic)
        return list(dict.fromkeys(out))

    @classmethod
    def _image_topics_from_sensor_object(cls, robot_id: str | None, sensor: "ISensor") -> list[str]:
        cfg = dict(getattr(sensor, "config", {}) or {})
        namespace = cfg.get("namespace")
        if not isinstance(namespace, str) or not namespace.strip():
            namespace = f"/{robot_id}" if robot_id else ""
        out: list[str] = []
        topic = cls._qualify_topic(namespace, cfg.get("rgb"))
        if topic:
            out.append(topic)
        depth_topic = cls._depth_topic_from_cfg(namespace, cfg)
        if depth_topic:
            out.append(depth_topic)
        return list(dict.fromkeys(out))

    @classmethod
    def _camera_info_topics_from_sensor_object(cls, robot_id: str | None, sensor: "ISensor") -> list[str]:
        cfg = dict(getattr(sensor, "config", {}) or {})
        namespace = cfg.get("namespace")
        if not isinstance(namespace, str) or not namespace.strip():
            namespace = f"/{robot_id}" if robot_id else ""
        topic = cls._qualify_topic(namespace, cfg.get("info"))
        return [topic] if topic else []

    @classmethod
    def _imu_topic_from_sensor_object(cls, robot_id: str | None, sensor: "ISensor") -> str | None:
        cls_name = type(sensor).__name__.lower()
        name = str(getattr(sensor, "name", "")).lower()
        if "imu" not in cls_name and "imu" not in name:
            return None
        cfg = dict(getattr(sensor, "config", {}) or {})
        namespace = cfg.get("namespace")
        if not isinstance(namespace, str) or not namespace.strip():
            namespace = f"/{robot_id}" if robot_id else ""
        rel = cfg.get("imu_topic") or cfg.get("topic") or "imu"
        return cls._qualify_topic(namespace, rel)

    @staticmethod
    def _default_gz_transport_node():
        for module_name in ("gz.transport13", "gz.transport12", "gz.transport"):
            try:
                module = __import__(module_name, fromlist=["Node"])
                return module.Node()
            except Exception:
                continue
        return None

    @classmethod
    def _wrench_topic_from_sensor_object(cls, robot_id: str | None, sensor: "ISensor") -> str | None:
        cfg = dict(getattr(sensor, "config", {}) or {})
        namespace = cfg.get("namespace")
        if not isinstance(namespace, str) or not namespace.strip():
            namespace = f"/{robot_id}" if robot_id else ""
        rel = cfg.get("wrench_topic") or cfg.get("topic")
        if not rel and "ft" in str(getattr(sensor, "name", "")).lower():
            rel = "wrench"
        return cls._qualify_topic(namespace, rel)

