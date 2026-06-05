"""Wrench (force/torque) sensor for ROS2 / Gazebo bridges."""

from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np

from robodeploy.core.registry import register_sensor, register_sensor_pair
from robodeploy.core.types import SensorData
from robodeploy.ros2 import Ros2NodeAdapter
from robodeploy.sensors.base import SensorBase

from .base import LastValueCache
from .interfaces import IRos2Sensor, Ros2SensorConfig
from .registry import register_ros2_sensor


def _topic(ns: str, rel: str) -> str:
    ns2 = (ns or "").rstrip("/")
    rel2 = (rel or "").lstrip("/")
    if not ns2:
        return f"/{rel2}"
    return f"{ns2}/{rel2}"


class Ros2WrenchSensor(Ros2NodeAdapter, IRos2Sensor):
    sensor_type = "wrench"

    def __init__(self, cfg: Ros2SensorConfig, backend_config: Optional[dict] = None) -> None:
        Ros2NodeAdapter.__init__(self)
        self._cfg = cfg
        self.name = str(cfg.name or "wrist_ft")
        self._backend_config = backend_config or {}
        self.node_name = f"robodeploy_wrench_{cfg.robot_id}_{cfg.name}"
        self._topic = str(cfg.topics.get("wrench") or cfg.topics.get("topic") or "wrench")
        self._cache: LastValueCache[tuple[np.ndarray, np.ndarray]] = LastValueCache()

    def _on_node_ready(self, node) -> None:
        from geometry_msgs.msg import WrenchStamped

        node.create_subscription(
            WrenchStamped,
            _topic(self._cfg.namespace, self._topic),
            self._on_wrench,
            int(self._cfg.qos_depth),
        )

    def _on_wrench(self, msg) -> None:
        stamp = getattr(getattr(msg, "header", None), "stamp", None)
        hw = 0.0
        if stamp is not None:
            hw = float(getattr(stamp, "sec", 0)) + float(getattr(stamp, "nanosec", 0)) * 1e-9
        force = np.asarray(
            [
                float(msg.wrench.force.x),
                float(msg.wrench.force.y),
                float(msg.wrench.force.z),
            ],
            dtype=np.float32,
        )
        torque = np.asarray(
            [
                float(msg.wrench.torque.x),
                float(msg.wrench.torque.y),
                float(msg.wrench.torque.z),
            ],
            dtype=np.float32,
        )
        self._cache.write((force, torque), hw_time_s=hw)

    def read(self) -> SensorData:
        lv = self._cache.read()
        recv = time.monotonic()
        hw = float(lv.hw_time_s)
        stamp = hw or recv
        force, torque = lv.value if lv.value is not None else (None, None)
        ft_forces = {self.name: force} if force is not None else {}
        ft_torques = {self.name: torque} if torque is not None else {}
        return SensorData(
            ft_force=force,
            ft_torque=torque,
            ft_forces=ft_forces,
            ft_torques=ft_torques,
            timestamp=stamp,
            timestamp_hw=stamp,
            timestamp_recv=recv,
            timestamp_source="hardware" if hw else "wall",
        )

    def get_diagnostics(self) -> dict[str, Any]:
        lv = self._cache.read()
        return {
            "sensor_type": self.sensor_type,
            "robot_id": self._cfg.robot_id,
            "name": self.name,
            "topic": self._topic,
            "has_wrench": lv.value is not None,
            "hw_time_s": float(lv.hw_time_s),
        }


@register_ros2_sensor("wrench")
def _make_wrench(cfg: Ros2SensorConfig, backend_config: dict) -> IRos2Sensor:
    return Ros2WrenchSensor(cfg, backend_config)


@register_sensor("ros2_wrench_real")
class Ros2WrenchISensor(SensorBase):
    def __init__(self, config: dict | None = None) -> None:
        cfg = dict(config or {})
        super().__init__(name=str(cfg.get("name", "wrist_ft")), is_real=True, config=cfg)

    def _init_impl(self, backend) -> None:
        robot_id = str(self.config.get("robot_id", getattr(backend, "_robot_id", "robot0")))
        namespace = str(self.config.get("namespace", f"/{robot_id}"))
        topic = str(self.config.get("wrench_topic") or self.config.get("topic") or "wrench")
        cfg = Ros2SensorConfig(
            robot_id=robot_id,
            name=self.name,
            namespace=namespace,
            topics={"wrench": topic},
        )
        self._impl = Ros2WrenchSensor(cfg, getattr(backend, "config", {}))
        self._impl.start()

    def _read_impl(self) -> SensorData:
        return self._impl.read()

    def _close_impl(self) -> None:
        self._impl.stop()


@register_sensor_pair(
    "ros2_wrench",
    real=Ros2WrenchISensor,
    by_backend={
        "ros2": Ros2WrenchISensor,
        "ros2_rviz": Ros2WrenchISensor,
        "gazebo": Ros2WrenchISensor,
    },
)
class Ros2WrenchPair:
    pass
