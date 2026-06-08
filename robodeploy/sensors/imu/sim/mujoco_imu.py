"""MuJoCo IMU sensor (accelerometer + gyro)."""

from __future__ import annotations

import time

import numpy as np

from robodeploy.backends.real.ros2.sensors.imu import Ros2ImuISensor
from robodeploy.core.registry import register_sensor, register_sensor_pair
from robodeploy.core.types import SensorData, SensorMount
from robodeploy.sensors.base import SensorBase
from robodeploy.sensors.imu.sim.isaacsim_imu import IsaacSimIMUSensor


@register_sensor("imu_sim")
class MuJoCoIMUSensor(SensorBase):
    def __init__(
        self,
        name: str | dict | None = None,
        *,
        config: dict | None = None,
        mount: SensorMount | None = None,
    ) -> None:
        if isinstance(name, dict) and config is None:
            cfg = dict(name)
            sensor_name = str(cfg.get("name", "wrist_imu"))
        else:
            cfg = dict(config or {})
            sensor_name = str(name or cfg.get("name", "wrist_imu"))
        if mount is None and isinstance(cfg.get("mount"), dict):
            mount = SensorMount(**cfg["mount"])
        super().__init__(name=sensor_name, is_real=False, config=cfg, mount=mount)

    def _init_impl(self, backend) -> None:
        if not hasattr(backend, "_mujoco") or not hasattr(backend, "_model") or not hasattr(backend, "_data"):
            raise RuntimeError("MuJoCoIMUSensor requires an initialized MuJoCoBackend.")
        self._mujoco = backend._mujoco
        self._model = backend._model
        self._data = backend._data
        self._accel_sensor = str(self.config.get("accel_sensor", f"{self.name}_accel"))
        self._gyro_sensor = str(self.config.get("gyro_sensor", f"{self.name}_gyro"))
        self._accel_id = self._mujoco.mj_name2id(self._model, self._mujoco.mjtObj.mjOBJ_SENSOR, self._accel_sensor)
        self._gyro_id = self._mujoco.mj_name2id(self._model, self._mujoco.mjtObj.mjOBJ_SENSOR, self._gyro_sensor)
        self._allow_missing = bool(self.config.get("allow_missing", False))
        if not self._allow_missing and (self._accel_id < 0 or self._gyro_id < 0):
            raise KeyError(
                f"MuJoCo IMU sensors '{self._accel_sensor}' / '{self._gyro_sensor}' not found."
            )

    def _read_impl(self) -> SensorData:
        accel = self._read_sensor_vec(self._accel_id, dim=3)
        gyro = self._read_sensor_vec(self._gyro_id, dim=3)
        sim_time = float(self._data.time)
        return SensorData(
            imu_acceleration=accel,
            imu_angular_velocity=gyro,
            timestamp=sim_time,
            timestamp_hw=sim_time,
            timestamp_recv=time.monotonic(),
            timestamp_source="sim",
        )

    def _close_impl(self) -> None:
        return

    def _read_sensor_vec(self, sensor_id: int, *, dim: int):
        if sensor_id < 0:
            if self._allow_missing:
                return np.zeros(dim, dtype=np.float32)
            raise KeyError(f"MuJoCo sensor id {sensor_id} is invalid.")
        adr = int(self._model.sensor_adr[sensor_id])
        size = int(self._model.sensor_dim[sensor_id])
        out = np.zeros(dim, dtype=np.float32)
        n = min(dim, size)
        out[:n] = np.asarray(self._data.sensordata[adr : adr + n], dtype=np.float32)
        return out


@register_sensor_pair(
    "wrist_imu",
    sim=MuJoCoIMUSensor,
    by_backend={
        "mujoco": MuJoCoIMUSensor,
        "isaacsim": IsaacSimIMUSensor,
        "ros2": Ros2ImuISensor,
        "ros2_rviz": Ros2ImuISensor,
        "gazebo": Ros2ImuISensor,
    },
)
class WristIMUPair:
    pass


@register_sensor_pair(
    "base_imu",
    sim=MuJoCoIMUSensor,
    by_backend={
        "mujoco": MuJoCoIMUSensor,
        "isaacsim": IsaacSimIMUSensor,
        "ros2": Ros2ImuISensor,
        "ros2_rviz": Ros2ImuISensor,
        "gazebo": Ros2ImuISensor,
    },
)
class BaseIMUPair:
    pass
