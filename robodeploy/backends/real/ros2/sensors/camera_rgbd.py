"""RGBD camera sensor for ROS2Backend.

Subscribes to ROS topics and exposes the latest RGB/depth frames via SensorData.
Designed to be optional and best-effort: missing topics yield None fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from robodeploy.core.registry import register_sensor, register_sensor_pair
from robodeploy.core.types import SensorData

from .base import LastValueCache
from .interfaces import IRos2Sensor, Ros2SensorConfig
from .registry import register_ros2_sensor
from robodeploy.ros2 import Ros2NodeAdapter
from robodeploy.sensors.base import SensorBase


def _topic(ns: str, rel: str) -> str:
    ns2 = (ns or "").rstrip("/")
    rel2 = (rel or "").lstrip("/")
    if not ns2:
        return f"/{rel2}"
    return f"{ns2}/{rel2}"


def _image_to_numpy(msg, *, desired_encoding: Optional[str] = None) -> np.ndarray:
    """Best-effort conversion for common encodings without requiring cv_bridge."""
    enc = str(desired_encoding or getattr(msg, "encoding", "") or "").lower()
    h = int(getattr(msg, "height"))
    w = int(getattr(msg, "width"))
    step = int(getattr(msg, "step", 0) or 0)
    data = bytes(getattr(msg, "data"))

    # RGB8/BGR8
    if enc in ("rgb8", "bgr8"):
        arr = np.frombuffer(data, dtype=np.uint8)
        arr = arr.reshape((h, step))[:, : w * 3]
        img = arr.reshape((h, w, 3))
        if enc == "bgr8":
            img = img[:, :, ::-1]
        return img

    # MONO8
    if enc in ("mono8", "8uc1"):
        arr = np.frombuffer(data, dtype=np.uint8)
        arr = arr.reshape((h, step))[:, :w]
        return arr

    # Depth 32FC1
    if enc in ("32fc1",):
        arr = np.frombuffer(data, dtype=np.float32)
        arr = arr.reshape((h, step // 4))[:, :w]
        return arr

    # Depth 16UC1 (millimeters) -> meters float32
    if enc in ("16uc1",):
        arr = np.frombuffer(data, dtype=np.uint16)
        arr = arr.reshape((h, step // 2))[:, :w]
        return (arr.astype(np.float32) * 0.001)

    raise ValueError(f"Unsupported image encoding '{enc}'. Consider installing cv_bridge.")


def _stamp_to_seconds(msg) -> float:  # noqa: ANN001
    stamp = getattr(getattr(msg, "header", None), "stamp", None)
    if stamp is None:
        return 0.0
    return float(getattr(stamp, "sec", 0)) + float(getattr(stamp, "nanosec", 0)) * 1e-9


@dataclass
class _CameraTopics:
    rgb: Optional[str] = None
    depth: Optional[str] = None
    info: Optional[str] = None


class Ros2RgbdCameraSensor(Ros2NodeAdapter, IRos2Sensor):
    sensor_type = "rgbd"

    def __init__(self, cfg: Ros2SensorConfig, backend_config: Optional[dict] = None) -> None:
        Ros2NodeAdapter.__init__(self)
        self._cfg = cfg
        self.name = str(cfg.name or "rgbd")
        self._backend_config = backend_config or {}
        self.node_name = f"robodeploy_rgbd_{cfg.robot_id}_{cfg.name}"
        self._topics = _CameraTopics(
            rgb=cfg.topics.get("rgb"),
            depth=cfg.topics.get("depth"),
            info=cfg.topics.get("info"),
        )
        self._rgb_cache: LastValueCache[np.ndarray] = LastValueCache()
        self._depth_cache: LastValueCache[np.ndarray] = LastValueCache()
        self._info_cache: LastValueCache[Any] = LastValueCache()
        self._last_error: str | None = None

    def _on_node_ready(self, node) -> None:
        try:
            from sensor_msgs.msg import CameraInfo, Image
        except ImportError as exc:
            raise ImportError("ROS2 RGBD sensor requires sensor_msgs.") from exc

        self._Image = Image
        self._CameraInfo = CameraInfo

        if self._topics.rgb:
            node.create_subscription(
                Image,
                _topic(self._cfg.namespace, self._topics.rgb),
                self._on_rgb,
                int(self._cfg.qos_depth),
            )
        if self._topics.depth:
            node.create_subscription(
                Image,
                _topic(self._cfg.namespace, self._topics.depth),
                self._on_depth,
                int(self._cfg.qos_depth),
            )
        if self._topics.info:
            node.create_subscription(
                CameraInfo,
                _topic(self._cfg.namespace, self._topics.info),
                self._on_info,
                int(self._cfg.qos_depth),
            )

    def _on_rgb(self, msg) -> None:
        try:
            img = _image_to_numpy(msg)
            if img.ndim == 2:
                img = np.repeat(img[:, :, None], 3, axis=2)
            self._rgb_cache.write(img.astype(np.uint8, copy=False), hw_time_s=_stamp_to_seconds(msg))
            self._last_error = None
        except Exception as exc:
            self._last_error = f"rgb: {type(exc).__name__}: {exc}"
            return

    def _on_depth(self, msg) -> None:
        try:
            depth = _image_to_numpy(msg)
            if depth.dtype != np.float32:
                depth = depth.astype(np.float32)
            self._depth_cache.write(depth, hw_time_s=_stamp_to_seconds(msg))
            self._last_error = None
        except Exception as exc:
            self._last_error = f"depth: {type(exc).__name__}: {exc}"
            return

    def _on_info(self, msg) -> None:
        self._info_cache.write(msg)

    def read(self) -> SensorData:
        rgb_lv = self._rgb_cache.read()
        depth_lv = self._depth_cache.read()
        recv = float(max(rgb_lv.wall_time_s, depth_lv.wall_time_s))
        hw = float(max(rgb_lv.hw_time_s, depth_lv.hw_time_s))
        stamp = hw or recv
        return SensorData(
            rgb=rgb_lv.value,
            depth=depth_lv.value,
            timestamp=stamp,
            timestamp_hw=stamp,
            timestamp_recv=recv,
            timestamp_source="hardware" if hw else "wall",
        )

    def get_diagnostics(self) -> dict[str, Any]:
        rgb_lv = self._rgb_cache.read()
        depth_lv = self._depth_cache.read()
        return {
            "sensor_type": self.sensor_type,
            "robot_id": self._cfg.robot_id,
            "name": self.name,
            "topics": {
                "rgb": self._topics.rgb,
                "depth": self._topics.depth,
                "info": self._topics.info,
            },
            "has_rgb": rgb_lv.value is not None,
            "has_depth": depth_lv.value is not None,
            "rgb_hw_time_s": float(rgb_lv.hw_time_s),
            "depth_hw_time_s": float(depth_lv.hw_time_s),
            "rgb_recv_time_s": float(rgb_lv.wall_time_s),
            "depth_recv_time_s": float(depth_lv.wall_time_s),
            "rgb_depth_skew_s": (
                abs(float(rgb_lv.hw_time_s) - float(depth_lv.hw_time_s))
                if rgb_lv.hw_time_s > 0.0 and depth_lv.hw_time_s > 0.0
                else None
            ),
            "last_error": self._last_error,
        }


@register_ros2_sensor("rgbd")
def _make_rgbd(cfg: Ros2SensorConfig, backend_config: dict) -> IRos2Sensor:
    return Ros2RgbdCameraSensor(cfg, backend_config)


@register_sensor("ros2_rgbd_camera_real")
class Ros2RgbdCameraISensor(SensorBase):
    """ISensor adapter for the ROS2 RGBD subscription implementation."""

    def __init__(self, config: dict | None = None) -> None:
        cfg = dict(config or {})
        super().__init__(name=str(cfg.get("name", "ros2_rgbd_camera")), is_real=True, config=cfg)

    def _init_impl(self, backend) -> None:
        robot_id = str(self.config.get("robot_id", getattr(backend, "_robot_id", "robot0")))
        namespace = str(self.config.get("namespace", f"/{robot_id}"))
        topics = {
            "rgb": self.config.get("rgb"),
            "depth": self.config.get("depth"),
            "info": self.config.get("info"),
        }
        topics = {k: v for k, v in topics.items() if isinstance(v, str) and v}
        cfg = Ros2SensorConfig(robot_id=robot_id, name=self.name, namespace=namespace, topics=topics)
        self._impl = Ros2RgbdCameraSensor(cfg, getattr(backend, "config", {}))
        self._impl.start()

    def _read_impl(self) -> SensorData:
        return self._impl.read()

    def _close_impl(self) -> None:
        self._impl.stop()


@register_sensor_pair(
    "ros2_rgbd_camera",
    real=Ros2RgbdCameraISensor,
    by_backend={
        "ros2": Ros2RgbdCameraISensor,
        "ros2_rviz": Ros2RgbdCameraISensor,
        "gazebo": Ros2RgbdCameraISensor,
    },
)
class Ros2RgbdCameraPair:
    pass


@register_sensor_pair(
    "wrist_camera",
    by_backend={
        "ros2": Ros2RgbdCameraISensor,
        "ros2_rviz": Ros2RgbdCameraISensor,
        "gazebo": Ros2RgbdCameraISensor,
    },
)
class Ros2WristCameraPair:
    pass

