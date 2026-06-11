"""Sim-only prop pose sensor — exposes backend object poses as SensorData."""

from __future__ import annotations

import time

from robodeploy.core.registry import register_sensor, register_sensor_pair
from robodeploy.core.types import SensorData
from robodeploy.sensors.base import SensorBase


@register_sensor("sim_prop_pose_sim")
class SimPropPoseSensor(SensorBase):
    """Read scene prop poses through the backend and publish them as sensor objects.

    This is the sim oracle path: policies should consume ``Observation.objects``,
    not call ``backend.get_prop_pose`` directly.
    """

    def __init__(self, name: str | dict | None = None, *, config: dict | None = None) -> None:
        if isinstance(name, dict) and config is None:
            cfg = dict(name)
            sensor_name = str(cfg.get("name", "prop_pose"))
        else:
            cfg = dict(config or {})
            sensor_name = str(name or cfg.get("name", "prop_pose"))
        super().__init__(name=sensor_name, is_real=False, config=cfg)
        self._backend = None
        self._prop_names = [str(x) for x in cfg.get("prop_names", ["source", "target"])]

    def _init_impl(self, backend) -> None:
        self._backend = backend
        if not hasattr(backend, "get_prop_pose"):
            raise RuntimeError("SimPropPoseSensor requires a backend with get_prop_pose().")

    def _read_impl(self) -> SensorData:
        assert self._backend is not None
        objects: dict[str, tuple[tuple[float, float, float], tuple[float, float, float, float]]] = {}
        for prop_name in self._prop_names:
            try:
                pos, quat = self._backend.get_prop_pose(prop_name)
                objects[prop_name] = (
                    (float(pos[0]), float(pos[1]), float(pos[2])),
                    (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])),
                )
            except KeyError:
                continue
        ts = time.monotonic()
        data = getattr(self._backend, "_data", None)
        if data is not None and hasattr(data, "time"):
            ts = float(data.time)
        return SensorData(
            objects=objects,
            timestamp=ts,
            timestamp_hw=ts,
            timestamp_recv=time.monotonic(),
            timestamp_source="sim",
            status="ok" if objects else "stale",
        )

    def _close_impl(self) -> None:
        self._backend = None


@register_sensor_pair("sim_prop_pose", sim=SimPropPoseSensor)
class SimPropPosePair:
    pass
