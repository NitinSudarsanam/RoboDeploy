from __future__ import annotations

import unittest

import numpy as np

from robodeploy.teleop.base import ITeleopDevice
from robodeploy.teleop.ros2_bridge import Ros2JoyTeleop, Ros2TwistTeleop


class Ros2TwistTeleopTests(unittest.TestCase):
    def setUp(self) -> None:
        self.device = Ros2TwistTeleop(scale_position=0.1, scale_orientation=0.2, node=object())
        self.device.start()

    def tearDown(self) -> None:
        self.device.stop()

    def test_implements_interface(self) -> None:
        self.assertIsInstance(self.device, ITeleopDevice)

    def test_twist_maps_to_cartesian_delta(self) -> None:
        self.device.inject_twist(linear=[1.0, 0.0, 0.0], angular=[0.0, 0.5, 0.0])
        cmd = self.device.poll()
        self.assertIsNotNone(cmd)
        assert cmd is not None
        self.assertAlmostEqual(float(cmd.delta_position[0]), 0.1, places=5)
        self.assertAlmostEqual(float(cmd.delta_orientation_rpy[1]), 0.1, places=5)


class Ros2JoyTeleopTests(unittest.TestCase):
    def setUp(self) -> None:
        self.device = Ros2JoyTeleop(scale_position=0.01, scale_orientation=0.05, node=object())
        self.device.start()

    def tearDown(self) -> None:
        self.device.stop()

    def test_implements_interface(self) -> None:
        self.assertIsInstance(self.device, ITeleopDevice)

    def test_joy_axes_map_to_motion(self) -> None:
        axes = [0.0] * 6
        axes[1] = 1.0
        axes[0] = -0.5
        self.device.inject_joy(axes=axes, buttons=[0, 0, 0])
        cmd = self.device.poll()
        self.assertIsNotNone(cmd)
        assert cmd is not None
        self.assertGreater(float(cmd.delta_position[0]), 0.0)
        self.assertLess(float(cmd.delta_position[1]), 0.0)

    def test_joy_button_record_toggle(self) -> None:
        self.device.inject_joy(axes=[0.0] * 6, buttons=[0, 0, 1])
        cmd1 = self.device.poll()
        self.assertIsNotNone(cmd1)
        assert cmd1 is not None
        self.assertTrue(cmd1.record_toggle)


if __name__ == "__main__":
    unittest.main()
