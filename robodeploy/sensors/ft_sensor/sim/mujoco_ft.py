"""MuJoCo force/torque sensor stub (sim)."""

from __future__ import annotations

from robodeploy.core.registry import register_sensor
from robodeploy.core.types import SensorData
from robodeploy.sensors.base import SensorBase


@register_sensor("ft_sensor_sim")
class MuJoCoFTSensor(SensorBase):
    def __init__(self, config: dict | None = None) -> None:
        super().__init__(name="ft_sensor", is_real=False, config=config)

    def _init_impl(self, backend) -> None:
        raise NotImplementedError("MuJoCo FT sensor not implemented yet.")

    def _read_impl(self) -> SensorData:
        raise NotImplementedError

    def _close_impl(self) -> None:
        pass

