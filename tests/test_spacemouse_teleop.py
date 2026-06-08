from __future__ import annotations

import unittest

import numpy as np

from robodeploy.teleop.base import ITeleopDevice
from robodeploy.teleop.spacemouse import SpaceMouseTeleop, _apply_deadzone


class SpaceMouseTeleopTests(unittest.TestCase):
    def setUp(self) -> None:
        self.device = SpaceMouseTeleop(
            deadzone=0.1,
            scale_position=0.01,
            scale_orientation=0.02,
            driver=object(),
        )
        self.device.start()

    def tearDown(self) -> None:
        self.device.stop()

    def test_implements_interface(self) -> None:
        self.assertIsInstance(self.device, ITeleopDevice)

    def test_deadzone_zeros_small_inputs(self) -> None:
        out = _apply_deadzone(np.array([0.05, 0.5, -0.5], dtype=np.float32), 0.1)
        self.assertAlmostEqual(float(out[0]), 0.0, places=5)
        self.assertGreater(abs(float(out[1])), 0.0)

    def test_translation_maps_to_delta_position(self) -> None:
        self.device.inject_state(translation=[0.0, 0.0, 1.0])
        cmd = self.device.poll()
        self.assertIsNotNone(cmd)
        assert cmd is not None
        self.assertIsNotNone(cmd.delta_position)
        self.assertAlmostEqual(float(cmd.delta_position[2]), 0.01, places=5)

    def test_rotation_maps_to_delta_rpy(self) -> None:
        self.device.inject_state(rotation=[0.0, 1.0, 0.0])
        cmd = self.device.poll()
        self.assertIsNotNone(cmd)
        assert cmd is not None
        self.assertIsNotNone(cmd.delta_orientation_rpy)
        self.assertAlmostEqual(float(cmd.delta_orientation_rpy[1]), 0.02, places=5)

    def test_button_record_toggle(self) -> None:
        self.device.inject_state(button=1)
        cmd = self.device.poll()
        self.assertIsNotNone(cmd)
        assert cmd is not None
        self.assertTrue(cmd.record_toggle)


if __name__ == "__main__":
    unittest.main()
