"""MuJoCo force/torque sensor (sim)."""

from __future__ import annotations

import time

import numpy as np

from robodeploy.core.registry import register_sensor, register_sensor_pair
from robodeploy.core.types import SensorData, SensorMount
from robodeploy.sensors.base import SensorBase
from robodeploy.sensors.ft_sensor.sim.isaacsim_ft import IsaacSimFTSensor


@register_sensor("ft_sensor_sim")
class MuJoCoFTSensor(SensorBase):
    def __init__(
        self,
        name: str | dict | None = None,
        *,
        config: dict | None = None,
        mount: SensorMount | None = None,
    ) -> None:
        if isinstance(name, dict) and config is None:
            cfg = dict(name)
            sensor_name = str(cfg.get("name", "ft_sensor"))
        else:
            cfg = dict(config or {})
            sensor_name = str(name or cfg.get("name", "ft_sensor"))
        if mount is None and isinstance(cfg.get("mount"), dict):
            mount = SensorMount(**cfg["mount"])
        super().__init__(name=sensor_name, is_real=False, config=cfg, mount=mount)

    def _init_impl(self, backend) -> None:
        if not hasattr(backend, "_mujoco") or not hasattr(backend, "_model") or not hasattr(backend, "_data"):
            raise RuntimeError("MuJoCoFTSensor requires an initialized MuJoCoBackend.")
        self._mujoco = backend._mujoco
        self._model = backend._model
        self._data = backend._data
        self._force_sensor = str(self.config.get("force_sensor", f"{self.name}_force"))
        self._torque_sensor = str(self.config.get("torque_sensor", f"{self.name}_torque"))
        self._force_id = self._mujoco.mj_name2id(self._model, self._mujoco.mjtObj.mjOBJ_SENSOR, self._force_sensor)
        self._torque_id = self._mujoco.mj_name2id(self._model, self._mujoco.mjtObj.mjOBJ_SENSOR, self._torque_sensor)
        self._allow_missing = bool(self.config.get("allow_missing", False))
        if not self._allow_missing and (self._force_id < 0 or self._torque_id < 0):
            raise KeyError(
                f"MuJoCo FT sensors '{self._force_sensor}' / '{self._torque_sensor}' not found."
            )

    def _read_impl(self) -> SensorData:
        force = self._read_sensor_vec(self._force_id)
        torque = self._read_sensor_vec(self._torque_id)
        sim_time = float(self._data.time)
        return SensorData(
            ft_force=force,
            ft_torque=torque,
            timestamp=sim_time,
            timestamp_hw=sim_time,
            timestamp_recv=time.monotonic(),
            timestamp_source="sim",
        )

    def _close_impl(self) -> None:
        pass

    def _read_sensor_vec(self, sensor_id: int):
        if sensor_id < 0:
            if self._allow_missing:
                return np.zeros(3, dtype=np.float32)
            raise KeyError(f"MuJoCo sensor id {sensor_id} is invalid.")
        adr = int(self._model.sensor_adr[sensor_id])
        dim = int(self._model.sensor_dim[sensor_id])
        out = np.zeros(3, dtype=np.float32)
        n = min(3, dim)
        out[:n] = np.asarray(self._data.sensordata[adr : adr + n], dtype=np.float32)
        return out


@register_sensor_pair(
    "wrist_ft",
    sim=MuJoCoFTSensor,
    by_backend={
        "mujoco": MuJoCoFTSensor,
        "isaacsim": IsaacSimFTSensor,
        "ros2_rviz": None,
        "gazebo": None,
    },
)
class WristFTPair:
    pass

