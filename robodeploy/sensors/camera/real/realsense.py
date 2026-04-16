"""RealSense camera sensor stub (real)."""

from __future__ import annotations

from robodeploy.core.registry import register_sensor
from robodeploy.core.types import SensorData
from robodeploy.sensors.base import SensorBase


@register_sensor("wrist_camera_real")
class RealSenseCamera(SensorBase):
    def __init__(self, config: dict | None = None) -> None:
        super().__init__(name="wrist_camera", is_real=True, config=config)

    def _init_impl(self, backend) -> None:
        raise NotImplementedError("RealSense driver not implemented yet.")

    def _read_impl(self) -> SensorData:
        raise NotImplementedError

    def _close_impl(self) -> None:
        pass

