"""IMU sensor for ROS2 / Gazebo bridges."""

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


class Ros2ImuSensor(Ros2NodeAdapter, IRos2Sensor):
    sensor_type = "imu"

    def __init__(self, cfg: Ros2SensorConfig, backend_config: Optional[dict] = None) -> None:
        Ros2NodeAdapter.__init__(self)
        self._cfg = cfg
        self.name = str(cfg.name or "wrist_imu")
        self._backend_config = backend_config or {}
        self.node_name = f"robodeploy_imu_{cfg.robot_id}_{cfg.name}"
        self._topic = str(cfg.topics.get("imu") or cfg.topics.get("topic") or "imu")
        self._cache: LastValueCache[tuple[np.ndarray, np.ndarray]] = LastValueCache()

    def _on_node_ready(self, node) -> None:
        from sensor_msgs.msg import Imu

        node.create_subscription(
            Imu,
            _topic(self._cfg.namespace, self._topic),
            self._on_imu,
            int(self._cfg.qos_depth),
        )

    def _on_imu(self, msg) -> None:
        stamp = getattr(getattr(msg, "header", None), "stamp", None)
        hw = 0.0
        if stamp is not None:
            hw = float(getattr(stamp, "sec", 0)) + float(getattr(stamp, "nanosec", 0)) * 1e-9
        accel = np.asarray(
            [
                float(msg.linear_acceleration.x),
                float(msg.linear_acceleration.y),
                float(msg.linear_acceleration.z),
            ],
            dtype=np.float32,
        )
        gyro = np.asarray(
            [
                float(msg.angular_velocity.x),
                float(msg.angular_velocity.y),
                float(msg.angular_velocity.z),
            ],
            dtype=np.float32,
        )
        self._cache.write((accel, gyro), hw_time_s=hw)

    def read(self) -> SensorData:
        lv = self._cache.read()
        recv = time.monotonic()
        hw = float(lv.hw_time_s)
        stamp = hw or recv
        accel, gyro = lv.value if lv.value is not None else (None, None)
        return SensorData(
            imu_acceleration=accel,
            imu_angular_velocity=gyro,
            timestamp=stamp,
            timestamp_hw=hw or stamp,
            timestamp_recv=recv,
            timestamp_source="hardware" if hw else "wall",
        )

    def get_diagnostics(self) -> dict[str, Any]:
        lv = self._cache.read()
        return {
            "sensor_type": self.sensor_type,
            "robot_id": self._cfg.robot_id,
            "name": self.name,
            "topic": _topic(self._cfg.namespace, self._topic),
            "has_imu": lv.value is not None,
            "hw_time_s": float(lv.hw_time_s),
        }


@register_ros2_sensor("imu")
def _make_imu(cfg: Ros2SensorConfig, backend_config: dict) -> IRos2Sensor:
    return Ros2ImuSensor(cfg, backend_config)


@register_sensor("ros2_imu_real")
class Ros2ImuISensor(SensorBase):
    def __init__(self, config: dict | None = None) -> None:
        cfg = dict(config or {})
        super().__init__(name=str(cfg.get("name", "wrist_imu")), is_real=True, config=cfg)

    def _init_impl(self, backend) -> None:
        robot_id = str(self.config.get("robot_id", getattr(backend, "_robot_id", "robot0")))
        namespace = str(self.config.get("namespace", f"/{self.name}"))
        topic = str(self.config.get("imu_topic") or self.config.get("topic") or "imu")
        cfg = Ros2SensorConfig(
            robot_id=robot_id,
            name=self.name,
            namespace=namespace,
            topics={"imu": topic},
        )
        self._impl = Ros2ImuSensor(cfg, getattr(backend, "config", {}))
        self._impl.start()

    def _read_impl(self) -> SensorData:
        return self._impl.read()

    def _close_impl(self) -> None:
        self._impl.stop()


@register_sensor_pair(
    "ros2_imu",
    real=Ros2ImuISensor,
    by_backend={
        "ros2": Ros2ImuISensor,
        "ros2_rviz": Ros2ImuISensor,
        "gazebo": Ros2ImuISensor,
    },
)
class Ros2ImuPair:
    pass
