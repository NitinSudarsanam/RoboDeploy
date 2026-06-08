"""RealSense camera sensor (real)."""

from __future__ import annotations

import time

import numpy as np

from robodeploy.core.registry import register_sensor, register_sensor_pair
from robodeploy.core.types import SensorData
from robodeploy.sensors.base import SensorBase


@register_sensor("wrist_camera_real")
class RealSenseCamera(SensorBase):
    def __init__(self, config: dict | None = None) -> None:
        cfg = dict(config or {})
        super().__init__(name=str(cfg.get("name", "wrist_camera")), is_real=True, config=cfg)

    def _init_impl(self, backend) -> None:
        del backend
        try:
            import pyrealsense2 as rs  # type: ignore[import-not-found]
        except Exception as exc:
            raise ImportError("RealSenseCamera requires pyrealsense2.") from exc

        self._rs = rs
        self._pipeline = rs.pipeline()
        cfg = rs.config()
        width = int(self.config.get("width", 640))
        height = int(self.config.get("height", 480))
        fps = int(self.config.get("fps", 30))
        cfg.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        if bool(self.config.get("depth", True)):
            cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self._profile = self._pipeline.start(cfg)
        self._timeout_ms = int(self.config.get("timeout_ms", 100))

    def _read_impl(self) -> SensorData:
        frames = self._pipeline.wait_for_frames(self._timeout_ms)
        color = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        rgb = np.asanyarray(color.get_data()) if color else None
        depth = None
        hw_ts = 0.0
        if color:
            hw_ts = float(color.get_timestamp()) * 0.001
        if depth_frame:
            depth_scale = float(self._profile.get_device().first_depth_sensor().get_depth_scale())
            depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale
            hw_ts = hw_ts or float(depth_frame.get_timestamp()) * 0.001
        recv = time.monotonic()
        intrinsics = self._camera_intrinsics()
        return SensorData(
            rgb=rgb,
            depth=depth,
            frame_id=str(self.name),
            intrinsics=intrinsics,
            extrinsics=self.mount_extrinsics(),
            timestamp=hw_ts or recv,
            timestamp_hw=hw_ts or recv,
            timestamp_recv=recv,
            timestamp_source="hardware" if hw_ts else "wall",
        )

    def _camera_intrinsics(self) -> dict[str, float] | None:
        profile = getattr(self, "_profile", None)
        if profile is None:
            return None
        try:
            stream = profile.get_stream(self._rs.stream.color)
            intr = stream.as_video_stream_profile().get_intrinsics()
            return {
                "width": float(intr.width),
                "height": float(intr.height),
                "fx": float(intr.fx),
                "fy": float(intr.fy),
                "cx": float(intr.ppx),
                "cy": float(intr.ppy),
            }
        except Exception:
            width = int(self.config.get("width", 640))
            height = int(self.config.get("height", 480))
            return {
                "width": float(width),
                "height": float(height),
                "fx": float(width),
                "fy": float(height),
                "cx": float(width) * 0.5,
                "cy": float(height) * 0.5,
            }

    def _close_impl(self) -> None:
        pipeline = getattr(self, "_pipeline", None)
        if pipeline is not None:
            pipeline.stop()


@register_sensor_pair("wrist_camera", real=RealSenseCamera)
class RealSenseWristCameraPair:
    pass

