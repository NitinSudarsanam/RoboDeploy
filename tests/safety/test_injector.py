from __future__ import annotations

import unittest

try:
    import jax.numpy as jnp
except Exception:
    import numpy as jnp  # type: ignore[assignment]

from robodeploy.core.types import Observation
from robodeploy.safety import SafetyViolationInjector


def _obs() -> Observation:
    return Observation(
        joint_positions=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        joint_velocities=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        joint_torques=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        ee_position=jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float32),
        ee_orientation=jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
        ee_velocity=jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float32),
        ee_angular_velocity=jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float32),
    )


class SafetyInjectorTests(unittest.TestCase):
    def test_force_spike_injection(self):
        injector = SafetyViolationInjector()
        injector.force_spike(80.0, duration_steps=2)
        obs = injector.apply(_obs())
        self.assertIsNotNone(obs.ft_force)
        self.assertAlmostEqual(float(obs.ft_force[0]), 80.0, places=3)
        obs2 = injector.apply(_obs())
        self.assertAlmostEqual(float(obs2.ft_force[0]), 80.0, places=3)
        obs3 = injector.apply(_obs())
        self.assertIsNone(obs3.ft_force)

    def test_collision_synthetic_violations(self):
        injector = SafetyViolationInjector()
        injector.collision("arm", "table")
        violations = injector.synthetic_violations()
        self.assertEqual(len(violations), 1)

    def test_joint_limit_excursion_injection(self):
        injector = SafetyViolationInjector()
        injector.joint_limit_excursion(joint_idx=0, magnitude_rad=9.0, duration_steps=1)
        obs = injector.apply(_obs())
        self.assertAlmostEqual(float(obs.joint_positions[0]), 9.0, places=3)

    def test_state_timeout_stales_hw_timestamp(self):
        injector = SafetyViolationInjector()
        injector.state_timeout(duration_s=60.0)
        base = _obs()
        base.timestamp = 100.0
        base.timestamp_hw = 100.0
        obs = injector.apply(base)
        self.assertLess(float(obs.timestamp_hw), float(obs.timestamp))


if __name__ == "__main__":
    unittest.main()
