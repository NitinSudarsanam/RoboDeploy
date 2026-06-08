from __future__ import annotations

import unittest

from robodeploy.teleop.keyboard import KeyboardTeleop


class KeyboardTeleopTests(unittest.TestCase):
    def setUp(self) -> None:
        self.device = KeyboardTeleop(
            step_position=0.02,
            step_orientation=0.1,
            use_listener=False,
        )
        self.device.start()

    def tearDown(self) -> None:
        self.device.stop()

    def test_wasdq_translation(self) -> None:
        self.device.inject_key("w", pressed=True)
        self.device.inject_key("d", pressed=True)
        cmd = self.device.poll()
        self.assertIsNotNone(cmd)
        assert cmd is not None
        self.assertAlmostEqual(float(cmd.delta_position[0]), 0.02, places=5)
        self.assertAlmostEqual(float(cmd.delta_position[1]), 0.02, places=5)

    def test_record_toggle_edge_triggered_once(self) -> None:
        self.device.inject_key("tab", pressed=True)
        cmd1 = self.device.poll()
        cmd2 = self.device.poll()
        self.assertIsNotNone(cmd1)
        assert cmd1 is not None
        self.assertTrue(cmd1.record_toggle)
        self.assertIsNone(cmd2)

    def test_reset_and_estop_edges(self) -> None:
        self.device.inject_key("r", pressed=True)
        cmd = self.device.poll()
        self.assertIsNotNone(cmd)
        assert cmd is not None
        self.assertTrue(cmd.reset_episode)

        self.device.inject_key("esc", pressed=True)
        cmd2 = self.device.poll()
        self.assertIsNotNone(cmd2)
        assert cmd2 is not None
        self.assertTrue(cmd2.e_stop)

    def test_step_size_brackets(self) -> None:
        before = self.device._step_position
        self.device.inject_key("[", pressed=True)
        self.device.poll()
        self.assertLess(self.device._step_position, before)


if __name__ == "__main__":
    unittest.main()
