"""RGBD camera sensor for ROS2Backend.

Subscribes to ROS topics and exposes the latest RGB/depth frames via SensorData.
Designed to be optional and best-effort: missing topics yield None fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from robodeploy.core.types import SensorData

from .base import LastValueCache
from .interfaces import IRos2Sensor, Ros2SensorConfig
from .registry import register_ros2_sensor
from robodeploy.ros2 import Ros2NodeAdapter


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
            self._rgb_cache.write(img.astype(np.uint8, copy=False))
        except Exception:
            return

    def _on_depth(self, msg) -> None:
        try:
            depth = _image_to_numpy(msg)
            if depth.dtype != np.float32:
                depth = depth.astype(np.float32)
            self._depth_cache.write(depth)
        except Exception:
            return

    def _on_info(self, msg) -> None:
        self._info_cache.write(msg)

    def read(self) -> SensorData:
        rgb_lv = self._rgb_cache.read()
        depth_lv = self._depth_cache.read()
        now = float(rgb_lv.wall_time_s or depth_lv.wall_time_s or 0.0)
        return SensorData(
            rgb=rgb_lv.value,
            depth=depth_lv.value,
            timestamp=now,
            timestamp_hw=now,
            timestamp_recv=now,
        )

    def get_diagnostics(self) -> dict[str, Any]:
        rgb_lv = self._rgb_cache.read()
        depth_lv = self._depth_cache.read()
        return {
            "sensor_type": self.sensor_type,
            "robot_id": self._cfg.robot_id,
            "name": self._cfg.name,
            "has_rgb": rgb_lv.value is not None,
            "has_depth": depth_lv.value is not None,
        }


@register_ros2_sensor("rgbd")
def _make_rgbd(cfg: Ros2SensorConfig, backend_config: dict) -> IRos2Sensor:
    return Ros2RgbdCameraSensor(cfg, backend_config)

