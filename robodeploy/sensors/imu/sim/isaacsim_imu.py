"""Isaac Sim IMU sensor (accelerometer + gyro)."""

from __future__ import annotations

import time

import numpy as np

from robodeploy.core.types import SensorData, SensorMount
from robodeploy.sensors.base import SensorBase


class IsaacSimIMUSensor(SensorBase):
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
        if not hasattr(backend, "_robot"):
            raise RuntimeError("IsaacSimIMUSensor requires an initialized IsaacSimBackend.")
        self._backend = backend
        self._mount_link = str(self.config.get("mount_link") or getattr(self.mount, "parent_link", None) or "ee_link")
        self._imu_sensor = None
        prim_path = str(self.config.get("prim_path", f"/World/Robot/{self._mount_link}/IMU"))
        try:
            from isaacsim.sensors.physics import IMUSensor  # type: ignore[import-not-found]

            self._imu_sensor = IMUSensor(
                prim_path=prim_path,
                frequency=int(self.config.get("frequency", 200)),
                translation=np.zeros(3, dtype=np.float64),
                orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            )
        except Exception:
            self._imu_sensor = None

    def _read_impl(self) -> SensorData:
        accel = np.zeros(3, dtype=np.float32)
        gyro = np.zeros(3, dtype=np.float32)
        if self._imu_sensor is not None:
            frame = self._imu_sensor.get_current_frame()
            if isinstance(frame, dict):
                accel = np.asarray(frame.get("lin_acc", accel), dtype=np.float32).reshape(3)
                gyro = np.asarray(frame.get("ang_vel", gyro), dtype=np.float32).reshape(3)
        else:
            robot = self._backend._robot
            link_idx = self._resolve_link_index(robot)
            if link_idx is not None:
                lin_vel, ang_vel = self._read_link_velocities(robot, link_idx)
                gyro = np.asarray(ang_vel, dtype=np.float32).reshape(3)
                gravity = np.array([0.0, 0.0, -9.81], dtype=np.float32)
                accel = gravity

        sim_time = float(getattr(self._backend, "_sim_time", 0.0))
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

    def _resolve_link_index(self, robot):
        for attr in ("get_link_index", "get_body_index"):
            fn = getattr(robot, attr, None)
            if callable(fn):
                try:
                    return int(fn(self._mount_link))
                except Exception:
                    continue
        return None

    def _read_link_velocities(self, robot, link_idx: int):
        fn = getattr(robot, "get_link_velocities", None)
        if not callable(fn):
            return np.zeros(3), np.zeros(3)
        out = fn(indices=[link_idx])
        if isinstance(out, tuple) and len(out) == 2:
            lin, ang = out
            return self._to_numpy(lin, 3), self._to_numpy(ang, 3)
        arr = self._to_numpy(out, 6)
        return arr[:3], arr[3:6]

    @staticmethod
    def _to_numpy(value, dim: int):
        if hasattr(value, "cpu"):
            value = value.cpu().numpy()
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        out = np.zeros(dim, dtype=np.float32)
        n = min(dim, arr.shape[0])
        out[:n] = arr[:n]
        return out
