from __future__ import annotations

import unittest

try:
    import jax.numpy as jnp
except Exception:
    import numpy as jnp  # type: ignore[assignment]

from robodeploy.core.spaces import ActionSpace
from robodeploy.core.types import Action, Observation
from robodeploy.safety import SafetyFilterGuard
from robodeploy.testing import DummyRobot


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


class SafetyFilterGuardTests(unittest.TestCase):
    def test_joint_clamp_at_limits(self):
        desc = DummyRobot()
        guard = SafetyFilterGuard(
            safety_filter=desc.get_safety_filter(),
            action_space=ActionSpace.JOINT_POS,
        )
        raw = Action(joint_positions=jnp.asarray([99.0, -99.0], dtype=jnp.float32))
        filtered, violations = guard.check_action(raw, _obs(), dt=0.05)
        clamped = jnp.asarray(filtered.joint_positions, dtype=jnp.float32)
        self.assertAlmostEqual(float(clamped[0]), 3.14, places=3)
        self.assertAlmostEqual(float(clamped[1]), -3.14, places=3)
        self.assertTrue(violations)


if __name__ == "__main__":
    unittest.main()
