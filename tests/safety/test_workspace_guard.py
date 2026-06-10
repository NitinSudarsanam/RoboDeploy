from __future__ import annotations

import unittest

import numpy as np

from robodeploy.core.types import Action, Observation
from robodeploy.safety import SafetyMonitor, WorkspaceGuard


def _obs(ee_xyz: tuple[float, float, float]) -> Observation:
    return Observation(
        joint_positions=np.zeros(7, dtype=np.float32),
        joint_velocities=np.zeros(7, dtype=np.float32),
        joint_torques=np.zeros(7, dtype=np.float32),
        ee_position=np.asarray(ee_xyz, dtype=np.float32),
        ee_orientation=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        ee_velocity=np.zeros(3, dtype=np.float32),
        ee_angular_velocity=np.zeros(3, dtype=np.float32),
    )


class WorkspaceGuardIntegrationTests(unittest.TestCase):
    def test_monitor_clamps_cartesian_action(self):
        low = np.array([0.2, -0.5, 0.0], dtype=np.float64)
        high = np.array([0.9, 0.5, 0.8], dtype=np.float64)
        monitor = SafetyMonitor(
            guards=[WorkspaceGuard(low_xyz=low, high_xyz=high, on_violation="clamp")]
        )
        action = Action(ee_position=np.asarray([1.5, 0.0, 0.4], dtype=np.float32))
        out = monitor.check_action(action, _obs((0.5, 0.0, 0.4)), dt=0.05)
        self.assertIsNotNone(out.ee_position)
        assert out.ee_position is not None
        self.assertAlmostEqual(float(out.ee_position[0]), 0.9, places=4)

    def test_monitor_observation_violation_outside_box(self):
        low = np.array([0.2, -0.5, 0.0], dtype=np.float64)
        high = np.array([0.9, 0.5, 0.8], dtype=np.float64)
        monitor = SafetyMonitor(
            guards=[WorkspaceGuard(low_xyz=low, high_xyz=high, on_violation="clamp")]
        )
        monitor.check_observation(_obs((0.1, 0.0, 0.4)))
        self.assertGreater(len(monitor.violations()), 0)


if __name__ == "__main__":
    unittest.main()
