"""MuJoCo contact sensor — exposes prop contact as obs.contact_state."""

from __future__ import annotations

import time

from robodeploy.core.registry import register_sensor, register_sensor_pair
from robodeploy.core.types import SensorData, SensorMount
from robodeploy.sensors.contact.base import ContactSensorBase


@register_sensor("contact_sim")
class MuJoCoContactSensor(ContactSensorBase):
    def __init__(
        self,
        name: str | dict | None = None,
        *,
        config: dict | None = None,
        mount: SensorMount | None = None,
    ) -> None:
        if isinstance(name, dict) and config is None:
            cfg = dict(name)
            sensor_name = str(cfg.get("name", "wrist_contact"))
        else:
            cfg = dict(config or {})
            sensor_name = str(name or cfg.get("name", "wrist_contact"))
        if mount is None and isinstance(cfg.get("mount"), dict):
            mount = SensorMount(**cfg["mount"])
        super().__init__(name=sensor_name, is_real=False, config=cfg, mount=mount)
        self._backend = None
        self._prop_name = str(cfg.get("prop_name", "source"))
        self._ee_distance_threshold = float(cfg.get("ee_distance_threshold", 0.04))

    def _init_impl(self, backend) -> None:
        self._backend = backend

    def _read_impl(self) -> SensorData:
        assert self._backend is not None
        has_contact_fn = getattr(self._backend, "has_prop_contact", None)
        near_fn = getattr(self._backend, "prop_near_ee", None)
        contact = bool(has_contact_fn(self._prop_name)) if callable(has_contact_fn) else False
        near = (
            bool(near_fn(self._prop_name, threshold=self._ee_distance_threshold))
            if callable(near_fn)
            else False
        )
        ts = time.monotonic()
        data = getattr(self._backend, "_data", None)
        if data is not None and hasattr(data, "time"):
            ts = float(data.time)
        return SensorData(
            contact_state={self.name: contact or near},
            timestamp=ts,
            timestamp_hw=ts,
            timestamp_recv=time.monotonic(),
            timestamp_source="sim",
        )

    def _close_impl(self) -> None:
        self._backend = None


@register_sensor_pair(
    "wrist_contact",
    sim=MuJoCoContactSensor,
    by_backend={"mujoco": MuJoCoContactSensor, "ros2_rviz": MuJoCoContactSensor},
)
class WristContactPair:
    pass


# Gazebo pair registered in gazebo_contact.py to avoid import cycles.
