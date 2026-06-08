from __future__ import annotations

import unittest

from robodeploy.teleop.base import ITeleopDevice
from robodeploy.teleop.gamepad import GamepadTeleop, _apply_deadzone


class GamepadTeleopTests(unittest.TestCase):
    def setUp(self) -> None:
        self.device = GamepadTeleop(
            deadzone=0.1,
            scale_position=0.01,
            scale_orientation=0.05,
            backend=object(),
        )
        self.device.start()

    def tearDown(self) -> None:
        self.device.stop()

    def test_implements_interface(self) -> None:
        self.assertIsInstance(self.device, ITeleopDevice)

    def test_deadzone(self) -> None:
        self.assertEqual(_apply_deadzone(0.05, 0.1), 0.0)
        self.assertGreater(_apply_deadzone(0.5, 0.1), 0.0)

    def test_left_stick_translation(self) -> None:
        self.device.inject_axis(0, 1.0)
        self.device.inject_axis(1, -1.0)
        cmd = self.device.poll()
        self.assertIsNotNone(cmd)
        assert cmd is not None
        self.assertAlmostEqual(float(cmd.delta_position[0]), -0.01, places=5)
        self.assertAlmostEqual(float(cmd.delta_position[1]), 0.01, places=5)

    def test_shoulder_buttons_z_axis(self) -> None:
        self.device.inject_button(5, pressed=True)
        cmd = self.device.poll()
        self.assertIsNotNone(cmd)
        assert cmd is not None
        self.assertGreater(float(cmd.delta_position[2]), 0.0)

    def test_face_button_edges(self) -> None:
        self.device.inject_button(2, pressed=True)
        cmd = self.device.poll()
        self.assertIsNotNone(cmd)
        assert cmd is not None
        self.assertTrue(cmd.record_toggle)


if __name__ == "__main__":
    unittest.main()
