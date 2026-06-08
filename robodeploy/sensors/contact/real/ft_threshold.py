"""FT-threshold contact sensor for real hardware (binary grasp detection)."""

from __future__ import annotations

import time

import numpy as np

from robodeploy.core.registry import register_sensor, register_sensor_pair
from robodeploy.core.types import SensorData
from robodeploy.sensors.contact.base import ContactSensorBase


@register_sensor("contact_ft_real")
class FTThresholdContactSensor(ContactSensorBase):
    """Treat FT magnitude above threshold as contact/grasp."""

    def __init__(self, config: dict | None = None) -> None:
        cfg = dict(config or {})
        super().__init__(
            name=str(cfg.get("name", "wrist_contact")),
            is_real=True,
            config=cfg,
        )
        self._ft_sensor = None
        self._threshold_N = float(cfg.get("force_threshold_N", 2.0))

    def _init_impl(self, backend) -> None:
        ft_name = str(self.config.get("ft_sensor_name", "wrist_ft"))
        self._ft_sensor = None
        for sensor in getattr(backend, "_sensors", []) or []:
            if str(getattr(sensor, "name", "")) == ft_name:
                self._ft_sensor = sensor
                break

    def _read_impl(self) -> SensorData:
        if self._ft_sensor is None:
            now = time.monotonic()
            return SensorData(
                contact_state={self.name: False},
                timestamp=now,
                timestamp_hw=now,
                timestamp_recv=now,
                timestamp_source="hardware",
                status="stale",
            )
        ft = self._ft_sensor.read()
        force = ft.ft_force
        magnitude = float(np.linalg.norm(force)) if force is not None else 0.0
        now = time.monotonic()
        return SensorData(
            contact_state={self.name: magnitude >= self._threshold_N},
            timestamp=now,
            timestamp_hw=float(ft.timestamp_hw or now),
            timestamp_recv=now,
            timestamp_source="hardware",
        )

    def _close_impl(self) -> None:
        if self._ft_sensor is not None:
            self._ft_sensor.close()
            self._ft_sensor = None


@register_sensor_pair("wrist_contact", real=FTThresholdContactSensor)
class FTThresholdContactPair:
    pass
