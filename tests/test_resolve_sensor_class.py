from __future__ import annotations

import unittest

from robodeploy.backends.real.ros2.sensors.camera_rgbd import Ros2RgbdCameraISensor
from robodeploy.backends.real.ros2.sensors.wrench import Ros2WrenchISensor
from robodeploy.core.registry import (
    SensorPairSpec,
    _SENSOR_PAIRS,
    normalize_sensor_backend_name,
    resolve_sensor_class,
)
from robodeploy.sensors.camera.sim.mujoco_camera import MuJoCoCameraRenderer
from robodeploy.sensors.ft_sensor.sim.mujoco_ft import MuJoCoFTSensor


class ResolveSensorClassTests(unittest.TestCase):
    def test_wrist_camera_backend_matrix(self):
        self.assertIs(
            resolve_sensor_class("wrist_camera", is_real=False, backend_name="mujoco"),
            MuJoCoCameraRenderer,
        )
        self.assertIs(
            resolve_sensor_class("wrist_camera", is_real=False, backend_name="gazebo"),
            Ros2RgbdCameraISensor,
        )
        self.assertIs(
            resolve_sensor_class("wrist_camera", is_real=False, backend_name="ros2_gazebo"),
            Ros2RgbdCameraISensor,
        )

    def test_overhead_camera_mujoco(self):
        from robodeploy.sensors.camera.sim.mujoco_camera import MuJoCoOverheadCameraRenderer

        self.assertIs(
            resolve_sensor_class("overhead_camera", is_real=False, backend_name="mujoco"),
            MuJoCoOverheadCameraRenderer,
        )

    def test_wrist_imu_mujoco_and_ros2(self):
        from robodeploy.backends.real.ros2.sensors.imu import Ros2ImuISensor
        from robodeploy.sensors.imu.sim.isaacsim_imu import IsaacSimIMUSensor
        from robodeploy.sensors.imu.sim.mujoco_imu import MuJoCoIMUSensor

        self.assertIs(
            resolve_sensor_class("wrist_imu", is_real=False, backend_name="mujoco"),
            MuJoCoIMUSensor,
        )
        self.assertIs(
            resolve_sensor_class("wrist_imu", is_real=False, backend_name="isaacsim"),
            IsaacSimIMUSensor,
        )
        self.assertIs(
            resolve_sensor_class("base_imu", is_real=False, backend_name="ros2"),
            Ros2ImuISensor,
        )

    def test_wrist_ft_ros2_uses_wrench_not_ati_udp(self):
        self.assertIs(
            resolve_sensor_class("wrist_ft", is_real=False, backend_name="ros2"),
            Ros2WrenchISensor,
        )
        self.assertIs(
            resolve_sensor_class("wrist_ft", is_real=True, backend_name="ros2"),
            Ros2WrenchISensor,
        )
        self.assertIs(
            resolve_sensor_class("wrist_ft", is_real=False, backend_name="mujoco"),
            MuJoCoFTSensor,
        )

    def test_normalize_ros2_gazebo_alias(self):
        self.assertEqual(normalize_sensor_backend_name("ros2_gazebo"), "gazebo")

    def test_typeerror_on_corrupt_pair_entry(self):
        name = "_test_bad_sensor_pair_xyz"
        _SENSOR_PAIRS[name] = SensorPairSpec(sim="not_a_class")
        try:
            with self.assertRaises(TypeError):
                resolve_sensor_class(name, is_real=False)
        finally:
            _SENSOR_PAIRS.pop(name, None)

    def test_missing_backend_raises_keyerror(self):
        with self.assertRaises(KeyError):
            resolve_sensor_class("does_not_exist_sensor", is_real=False)


if __name__ == "__main__":
    unittest.main()
