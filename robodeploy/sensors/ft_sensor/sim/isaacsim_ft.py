"""Isaac Sim force/torque sensor (sim)."""

from __future__ import annotations

import time

import numpy as np

from robodeploy.core.registry import register_sensor
from robodeploy.core.types import SensorData, SensorMount
from robodeploy.sensors.base import SensorBase


@register_sensor("wrist_ft_isaacsim")
class IsaacSimFTSensor(SensorBase):
    def __init__(
        self,
        name: str | dict | None = None,
        *,
        config: dict | None = None,
        mount: SensorMount | None = None,
    ) -> None:
        if isinstance(name, dict) and config is None:
            cfg = dict(name)
            sensor_name = str(cfg.get("name", "wrist_ft"))
        else:
            cfg = dict(config or {})
            sensor_name = str(name or cfg.get("name", "wrist_ft"))
        if mount is None and isinstance(cfg.get("mount"), dict):
            mount = SensorMount(**cfg["mount"])
        super().__init__(name=sensor_name, is_real=False, config=cfg, mount=mount)

    def _init_impl(self, backend) -> None:
        if not hasattr(backend, "_robot"):
            raise RuntimeError("IsaacSimFTSensor requires an initialized IsaacSimBackend.")
        self._backend = backend
        self._joint_index = self._resolve_joint_index()

    def _read_impl(self) -> SensorData:
        robot = self._backend._robot
        force = np.zeros(3, dtype=np.float32)
        torque = np.zeros(3, dtype=np.float32)

        measured = getattr(robot, "get_measured_joint_forces", None)
        if callable(measured):
            wrench = np.asarray(measured(), dtype=np.float32)
            if wrench.ndim == 1:
                wrench = wrench.reshape(1, -1)
            if wrench.size:
                row = wrench[min(max(self._joint_index, 0), wrench.shape[0] - 1)]
                force = np.asarray(row[:3], dtype=np.float32)
                torque = np.asarray(row[3:6] if row.shape[0] >= 6 else np.zeros(3), dtype=np.float32)
        else:
            efforts = getattr(robot, "get_measured_joint_efforts", None)
            if callable(efforts):
                data = np.asarray(efforts(), dtype=np.float32).reshape(-1)
                if data.size:
                    torque = np.zeros(3, dtype=np.float32)
                    torque[-1] = float(data[min(max(self._joint_index, 0), data.shape[0] - 1)])

        sim_time = float(getattr(self._backend, "_sim_time", 0.0))
        return SensorData(
            ft_force=force,
            ft_torque=torque,
            ft_forces={self.name: force},
            ft_torques={self.name: torque},
            timestamp=sim_time,
            timestamp_hw=sim_time,
            timestamp_recv=time.monotonic(),
            timestamp_source="sim",
        )

    def _close_impl(self) -> None:
        return

    def _resolve_joint_index(self) -> int:
        explicit = self.config.get("joint_index")
        if explicit is not None:
            try:
                return int(explicit)
            except Exception:
                pass
        joint_name = str(self.config.get("joint_name", "") or "")
        if joint_name and hasattr(self._backend, "_description"):
            names = list(getattr(self._backend._description, "joint_names", []) or [])
            if joint_name in names:
                return int(names.index(joint_name))
        names = list(getattr(getattr(self._backend, "_description", None), "joint_names", []) or [])
        return max(0, len(names) - 1)
