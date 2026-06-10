from __future__ import annotations

import unittest

import numpy as np

from robodeploy.teleop.base import ITeleopDevice, TeleopCommand
from robodeploy.teleop.gamepad import GamepadTeleop
from robodeploy.teleop.keyboard import KeyboardTeleop
from robodeploy.teleop.ros2_bridge import Ros2TwistTeleop
from robodeploy.teleop.spacemouse import SpaceMouseTeleop
from robodeploy.teleop.vr import VRTeleop


class _StubDevice(ITeleopDevice):
    def start(self) -> None:
        return

    def poll(self) -> TeleopCommand | None:
        return TeleopCommand(delta_joint_positions=np.ones(2, dtype=np.float32) * 0.01)

    def stop(self) -> None:
        return


class TeleopContractTests(unittest.TestCase):
    def test_stub_device_implements_interface(self) -> None:
        device = _StubDevice()
        device.start()
        cmd = device.poll()
        device.stop()
        self.assertIsInstance(cmd, TeleopCommand)
        self.assertTrue(cmd.has_motion())

    def test_keyboard_teleop_implements_interface(self) -> None:
        device = KeyboardTeleop(use_listener=False)
        self.assertIsInstance(device, ITeleopDevice)
        device.start()
        device.inject_key("w", pressed=True)
        cmd = device.poll()
        device.inject_key("w", pressed=False)
        device.stop()
        self.assertIsNotNone(cmd)
        assert cmd is not None
        self.assertIsNotNone(cmd.delta_position)
        self.assertGreater(float(cmd.delta_position[0]), 0.0)

    def test_spacemouse_and_gamepad_implement_interface(self) -> None:
        for cls, kwargs in (
            (SpaceMouseTeleop, {"driver": object()}),
            (GamepadTeleop, {"backend": object()}),
            (Ros2TwistTeleop, {"node": object()}),
            (VRTeleop, {"session": object()}),
        ):
            device = cls(**kwargs)
            self.assertIsInstance(device, ITeleopDevice)
            device.start()
            device.stop()

    def test_vr_teleop_requires_pyopenxr_without_session(self) -> None:
        device = VRTeleop()
        with self.assertRaises(ImportError):
            device.start()

    def test_keyboard_stub_example_implements_contract(self) -> None:
        from examples.teleop_keyboard_stub import KeyboardStubTeleop, run_stub_demo

        device = KeyboardStubTeleop()
        self.assertIsInstance(device, ITeleopDevice)
        commands = run_stub_demo(keys=["w"])
        self.assertGreaterEqual(len(commands), 1)
        self.assertTrue(any(c.has_motion() for c in commands))


if __name__ == "__main__":
    unittest.main()
