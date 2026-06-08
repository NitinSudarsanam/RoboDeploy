from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from robodeploy.backends.real.ros2.sensors.imu import Ros2ImuISensor, Ros2ImuSensor
from robodeploy.backends.real.ros2.sensors.interfaces import Ros2SensorConfig
from robodeploy.core.registry import resolve_sensor_class
from robodeploy.core.types import SensorMount
from robodeploy.sensors.imu.sim.isaacsim_imu import IsaacSimIMUSensor
from robodeploy.sensors.imu.sim.isaacsim_imu import IsaacSimIMUSensor
from robodeploy.sensors.imu.sim.mujoco_imu import MuJoCoIMUSensor
from robodeploy.sensors.imu.real.xsens import XsensIMUSensor


class ImuSensorTests(unittest.TestCase):
    def test_resolve_wrist_imu_mujoco(self):
        self.assertIs(
            resolve_sensor_class("wrist_imu", is_real=False, backend_name="mujoco"),
            MuJoCoIMUSensor,
        )

    def test_resolve_wrist_imu_ros2(self):
        self.assertIs(
            resolve_sensor_class("wrist_imu", is_real=False, backend_name="ros2"),
            Ros2ImuISensor,
        )

    def test_resolve_wrist_imu_isaacsim(self):
        self.assertIs(
            resolve_sensor_class("wrist_imu", is_real=False, backend_name="isaacsim"),
            IsaacSimIMUSensor,
        )

    def test_xsens_stub_reads_without_port(self):
        sensor = XsensIMUSensor(config={"name": "wrist_imu"})
        sensor.initialize(mock.Mock())
        reading = sensor.read()
        self.assertIsNotNone(reading.imu_acceleration)
        self.assertIsNotNone(reading.imu_angular_velocity)
        np.testing.assert_allclose(reading.imu_acceleration, [0.0, 0.0, 9.81], rtol=1e-3)

    def test_resolve_wrist_imu_isaacsim(self):
        self.assertIs(
            resolve_sensor_class("wrist_imu", is_real=False, backend_name="isaacsim"),
            IsaacSimIMUSensor,
        )

    def test_mujoco_imu_reads_accel_and_gyro(self):
        sensor = MuJoCoIMUSensor(
            "wrist_imu",
            mount=SensorMount(parent_link="robot0/ee_link"),
        )
        backend = mock.Mock()
        backend._mujoco = mock.Mock()
        backend._mujoco.mjtObj.mjOBJ_SENSOR = 3
        class _Model:
            sensor_adr = np.asarray([0, 3], dtype=np.int32)
            sensor_dim = np.asarray([3, 3], dtype=np.int32)

        backend._model = _Model()
        backend._data = mock.Mock()
        backend._data.time = 1.5
        backend._data.sensordata = np.asarray(
            [0.0, 0.0, -9.81, 0.1, 0.0, 0.0],
            dtype=np.float64,
        )
        backend._mujoco.mj_name2id = mock.Mock(side_effect=[0, 1])
        sensor.initialize(backend)
        reading = sensor.read()
        self.assertIsNotNone(reading.imu_acceleration)
        self.assertIsNotNone(reading.imu_angular_velocity)
        np.testing.assert_allclose(reading.imu_acceleration, [0.0, 0.0, -9.81], rtol=1e-4)
        np.testing.assert_allclose(reading.imu_angular_velocity, [0.1, 0.0, 0.0], rtol=1e-4)

    def test_ros2_imu_sensor_read_from_cache(self):
        impl = Ros2ImuSensor(
            Ros2SensorConfig(robot_id="robot0", name="wrist_imu", namespace="/wrist_imu", topics={"imu": "imu"}),
            {},
        )
        accel = np.asarray([0.0, 0.0, 9.8], dtype=np.float32)
        gyro = np.asarray([0.01, 0.0, 0.0], dtype=np.float32)
        impl._cache.write((accel, gyro), hw_time_s=0.5)
        reading = impl.read()
        np.testing.assert_allclose(reading.imu_acceleration, accel)
        np.testing.assert_allclose(reading.imu_angular_velocity, gyro)


if __name__ == "__main__":
    unittest.main()
