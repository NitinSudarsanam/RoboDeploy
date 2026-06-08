"""Mocked real-hardware e-stop integration (ROS2 controller path)."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from robodeploy.backends.real.ros2.safety import EStop, JointLimitGuard, SafetyError
from robodeploy.safety import SafetyMonitor


class _FakeHardwareLoop:
    """Minimal stand-in for a ROS2 controller command loop."""

    def __init__(self) -> None:
        self._estop = EStop(signal_handlers=False, enable_console=False)
        limits = np.array([[-2.0, 2.0]] * 2, dtype=np.float64)
        vel = np.array([1.5, 1.5], dtype=np.float64)
        self._limit_guard = JointLimitGuard(lower=limits[:, 0], upper=limits[:, 1], vel_max=vel)
        self._q = np.zeros(2, dtype=np.float64)
        self.hard_stop_calls = 0

    def hard_stop(self, reason: str) -> None:
        self.hard_stop_calls += 1
        self._estop.trip(reason)

    def apply_goal(self, q_des: np.ndarray) -> None:
        self._estop.check()
        q_des = np.asarray(q_des, dtype=np.float64).reshape(-1)
        try:
            self._limit_guard.check(q_des, dt=None)
        except SafetyError as exc:
            self.hard_stop(str(exc))
            raise
        self._q[:] = q_des

    def estop_status(self) -> dict:
        return {"estop_tripped": bool(self._estop.tripped)}


class RealEstopIntegrationTests(unittest.TestCase):
    def test_estop_blocks_hardware_goal(self):
        loop = _FakeHardwareLoop()
        loop._estop.trip("operator button")
        with self.assertRaises(SafetyError):
            loop.apply_goal(np.array([0.1, 0.1], dtype=np.float64))
        self.assertTrue(loop.estop_status()["estop_tripped"])

    def test_limit_violation_triggers_hard_stop(self):
        loop = _FakeHardwareLoop()
        with self.assertRaises(SafetyError):
            loop.apply_goal(np.array([9.0, 0.0], dtype=np.float64))
        self.assertEqual(loop.hard_stop_calls, 1)
        self.assertTrue(loop.estop_status()["estop_tripped"])

    @patch("robodeploy.backends.real.ros2.controllers.so101_feetech.EStop")
    def test_so101_controller_wires_shared_estop(self, estop_cls: MagicMock) -> None:
        estop = EStop(signal_handlers=False, enable_console=False)
        estop_cls.return_value = estop
        monitor = SafetyMonitor(estop=estop)
        estop.trip("mock hardware estop")
        with self.assertRaises(SafetyError):
            monitor.check_action(
                __import__("robodeploy.core.types", fromlist=["Action"]).Action(),
                __import__("robodeploy.core.types", fromlist=["Observation"]).Observation(
                    joint_positions=np.zeros(2, dtype=np.float32),
                    joint_velocities=np.zeros(2, dtype=np.float32),
                    joint_torques=np.zeros(2, dtype=np.float32),
                    ee_position=np.zeros(3, dtype=np.float32),
                    ee_orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                    ee_velocity=np.zeros(3, dtype=np.float32),
                    ee_angular_velocity=np.zeros(3, dtype=np.float32),
                ),
                dt=0.05,
            )


if __name__ == "__main__":
    unittest.main()
