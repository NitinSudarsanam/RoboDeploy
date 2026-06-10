"""Gazebo contact sensor — binary touch via gz-transport contacts topic."""

from __future__ import annotations

import time

from robodeploy.core.registry import register_sensor, register_sensor_pair
from robodeploy.core.types import SensorData, SensorMount
from robodeploy.sensors.contact.base import ContactSensorBase


@register_sensor("contact_gazebo")
class GazeboContactSensor(ContactSensorBase):
    """Expose Gazebo contact monitor state as ``obs.contact_state``."""

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
        self._other_body = cfg.get("other_body")
        self._contacts_topic = str(cfg.get("contacts_topic", "contacts"))

    def _init_impl(self, backend) -> None:
        self._backend = backend
        monitor = getattr(backend, "_contact_monitor", None)
        if monitor is None or not hasattr(monitor, "bind_transport"):
            return
        if getattr(monitor, "_subscriber", None) is not None:
            return
        gz_node = getattr(backend, "_gz_transport_node", None)
        if gz_node is None:
            sim_cfg = getattr(backend, "config", {}) or {}
            gz_node = sim_cfg.get("gz_transport_node")
        if gz_node is None:
            return
        world = getattr(backend, "_gz_world_name", None)
        topic = f"/world/{world}/contacts" if world else self._contacts_topic
        monitor.bind_transport(gz_node, topic=topic)

    def _read_impl(self) -> SensorData:
        assert self._backend is not None
        has_contact_fn = getattr(self._backend, "has_prop_contact", None)
        contact = False
        if callable(has_contact_fn):
            other = str(self._other_body) if self._other_body else None
            contact = bool(has_contact_fn(self._prop_name, other_body=other))
        ts = time.monotonic()
        return SensorData(
            contact_state={self.name: contact},
            timestamp=ts,
            timestamp_hw=ts,
            timestamp_recv=time.monotonic(),
            timestamp_source="sim",
        )

    def _close_impl(self) -> None:
        self._backend = None


@register_sensor_pair(
    "wrist_contact",
    sim=GazeboContactSensor,
    by_backend={"gazebo": GazeboContactSensor, "ros2": GazeboContactSensor},
)
class GazeboWristContactPair:
    pass
