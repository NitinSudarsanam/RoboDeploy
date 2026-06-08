"""Tactile pressure-array stub — deferred until hardware driver lands."""

from __future__ import annotations

import time

import numpy as np

from robodeploy.core.registry import register_sensor, register_sensor_pair
from robodeploy.core.types import SensorData
from robodeploy.sensors.base import SensorBase

_TACTILE_DEFERRED = (
    "Tactile pressure-array hardware is not integrated. "
    "This stub returns a zero pressure grid for API compatibility."
)


@register_sensor("tactile_array_stub")
class TactileArrayStubSensor(SensorBase):
    """Placeholder tactile sensor returning a flat pressure grid."""

    def __init__(self, config: dict | None = None) -> None:
        cfg = dict(config or {})
        super().__init__(name=str(cfg.get("name", "tactile_array")), is_real=False, config=cfg)
        rows = int(cfg.get("rows", 4))
        cols = int(cfg.get("cols", 4))
        self._shape = (rows, cols)

    def _init_impl(self, backend) -> None:
        del backend

    def _read_impl(self) -> SensorData:
        now = time.monotonic()
        grid = np.zeros(self._shape, dtype=np.float32)
        return SensorData(
            timestamp=now,
            timestamp_hw=now,
            timestamp_recv=now,
            timestamp_source="stub",
            status="deferred",
            objects={"_tactile_pressure": ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))},
        )

    def _close_impl(self) -> None:
        return

    @property
    def deferred_reason(self) -> str:
        return _TACTILE_DEFERRED


@register_sensor_pair("tactile_array", sim=TactileArrayStubSensor, real=TactileArrayStubSensor)
class TactileArrayPair:
    """Deferred tactile array — sim and real both resolve to the stub."""

    pass
