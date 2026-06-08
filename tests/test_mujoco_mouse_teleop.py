from __future__ import annotations

import unittest

import numpy as np

from robodeploy.teleop.base import ITeleopDevice
from robodeploy.teleop.mujoco_mouse import MuJoCoMouseIKTeleop


class _FakeBackend:
    def get_observation(self):
        return type(
            "Obs",
            (),
            {
                "ee_position": np.array([0.5, 0.0, 0.4], dtype=np.float32),
                "ee_orientation": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            },
        )()


class MuJoCoMouseTeleopTests(unittest.TestCase):
    def setUp(self) -> None:
        self.device = MuJoCoMouseIKTeleop(
            backend=_FakeBackend(),
            viewer=object(),
        )
        self.device.start()

    def tearDown(self) -> None:
        self.device.stop()

    def test_implements_interface(self) -> None:
        self.assertIsInstance(self.device, ITeleopDevice)

    def test_idle_returns_none(self) -> None:
        self.assertIsNone(self.device.poll())

    def test_injected_target_yields_delta(self) -> None:
        self.device.inject_target(position=[0.52, 0.01, 0.41])
        cmd = self.device.poll()
        self.assertIsNotNone(cmd)
        assert cmd is not None
        self.assertIsNotNone(cmd.delta_position)
        self.assertAlmostEqual(float(cmd.delta_position[0]), 0.02, places=5)
        self.assertAlmostEqual(float(cmd.delta_position[1]), 0.01, places=5)
        self.assertAlmostEqual(float(cmd.delta_position[2]), 0.01, places=5)


if __name__ == "__main__":
    unittest.main()
