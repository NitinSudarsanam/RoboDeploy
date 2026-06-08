from __future__ import annotations

import unittest

try:
    import jax.numpy as jnp
except Exception:
    import numpy as jnp  # type: ignore[assignment]

from robodeploy.core.types import Observation
from robodeploy.safety import ForceLimitGuard, Hazard, Severity


def _obs(ft_force=None) -> Observation:
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


class ForceLimitGuardTests(unittest.TestCase):
    def test_no_ft_sensor_is_ok(self):
        guard = ForceLimitGuard(max_force_N=10.0)
        violations = guard.check_observation(_obs())
        self.assertEqual(violations, [])

    def test_force_spike_warning(self):
        guard = ForceLimitGuard(max_force_N=10.0, over_limit_strikes=3)
        violations = guard.check_observation(_obs(ft_force=jnp.asarray([12.0, 0.0, 0.0], dtype=jnp.float32)))
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].hazard, Hazard.FORCE_LIMIT)
        self.assertEqual(violations[0].severity, Severity.WARNING)


if __name__ == "__main__":
    unittest.main()
