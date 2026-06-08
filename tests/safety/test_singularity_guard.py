from __future__ import annotations

import unittest

import numpy as np

try:
    import jax.numpy as jnp
except Exception:
    import numpy as jnp  # type: ignore[assignment]

from robodeploy.core.types import Action, Observation
from robodeploy.safety import Hazard, SingularityGuard, Severity


def _obs(q, dq) -> Observation:
    return Observation(
        joint_positions=jnp.asarray(q, dtype=jnp.float32),
        joint_velocities=jnp.asarray(dq, dtype=jnp.float32),
        joint_torques=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        ee_position=jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float32),
        ee_orientation=jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
        ee_velocity=jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float32),
        ee_angular_velocity=jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float32),
    )


class SingularityGuardTests(unittest.TestCase):
    def test_near_limit_high_velocity_flags_singularity(self):
        limits = np.array([[-3.0, 3.0], [-0.2, 2.0]], dtype=np.float64)
        guard = SingularityGuard(
            joint_position_limits=limits,
            margin_rad=0.1,
            velocity_threshold_rad_s=1.0,
            over_limit_strikes=1,
        )
        obs = _obs([2.95, 0.0], [0.0, 0.0])
        action = Action(joint_velocities=jnp.asarray([2.5, 0.0], dtype=jnp.float32))
        _, violations = guard.check_action(action, obs, dt=0.05)
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].hazard, Hazard.SINGULARITY_IMMINENT)
        self.assertEqual(violations[0].severity, Severity.CRITICAL)


if __name__ == "__main__":
    unittest.main()
