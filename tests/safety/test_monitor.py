from __future__ import annotations

import unittest

try:
    import jax.numpy as jnp
except Exception:
    import numpy as jnp  # type: ignore[assignment]

from robodeploy.core.types import Action, Observation
from robodeploy.safety import ForceLimitGuard, SafetyError, SafetyMonitor, Severity


def make_safety_obs(ft_force=None) -> Observation:
    return Observation(
        joint_positions=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        joint_velocities=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        joint_torques=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        ee_position=jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float32),
        ee_orientation=jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
        ee_velocity=jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float32),
        ee_angular_velocity=jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float32),
        ft_force=ft_force,
    )


class SafetyMonitorTests(unittest.TestCase):
    def test_force_limit_warning_then_critical(self):
        monitor = SafetyMonitor(guards=[ForceLimitGuard(max_force_N=10.0, over_limit_strikes=3)])
        spike = jnp.asarray([15.0, 0.0, 0.0], dtype=jnp.float32)
        monitor.check_observation(make_safety_obs(ft_force=spike))
        self.assertEqual(monitor.violations()[-1].severity, Severity.WARNING)
        monitor.check_observation(make_safety_obs(ft_force=spike))
        with self.assertRaises(SafetyError):
            monitor.check_observation(make_safety_obs(ft_force=spike))

    def test_status_reports_tripped(self):
        monitor = SafetyMonitor(guards=[ForceLimitGuard(max_force_N=1.0, over_limit_strikes=1)])
        with self.assertRaises(SafetyError):
            monitor.check_observation(make_safety_obs(ft_force=jnp.asarray([5.0, 0.0, 0.0], dtype=jnp.float32)))
        status = monitor.status()
        self.assertTrue(status.tripped)
        self.assertGreater(status.history_count, 0)

    def test_reset_clears_tripped(self):
        monitor = SafetyMonitor(guards=[ForceLimitGuard(max_force_N=1.0, over_limit_strikes=1)])
        with self.assertRaises(SafetyError):
            monitor.check_observation(make_safety_obs(ft_force=jnp.asarray([5.0, 0.0, 0.0], dtype=jnp.float32)))
        monitor.reset()
        self.assertFalse(monitor.tripped)

    def test_check_action_passthrough(self):
        monitor = SafetyMonitor(guards=[ForceLimitGuard(max_force_N=50.0)])
        action = Action(joint_positions=jnp.asarray([0.1, 0.2], dtype=jnp.float32))
        out = monitor.check_action(action, make_safety_obs(), dt=0.05)
        self.assertIsNotNone(out.joint_positions)


if __name__ == "__main__":
    unittest.main()
