from __future__ import annotations

import unittest

try:
    import jax.numpy as jnp
except Exception:
    import numpy as jnp  # type: ignore[assignment]

from robodeploy.core.types import Observation
from robodeploy.safety import Hazard, HumanProximityGuard, Severity


def _obs(**kwargs) -> Observation:
    base = dict(
        joint_positions=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        joint_velocities=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        joint_torques=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        ee_position=jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float32),
        ee_orientation=jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
        ee_velocity=jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float32),
        ee_angular_velocity=jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float32),
    )
    base.update(kwargs)
    return Observation(**base)


class HumanProximityGuardTests(unittest.TestCase):
    def test_metadata_proximity_critical(self):
        guard = HumanProximityGuard(min_distance_m=0.25, over_limit_strikes=1)
        obs = _obs(metadata={"proximity_m": 0.1})
        violations = guard.check_observation(obs)
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].hazard, Hazard.HUMAN_PROXIMITY)
        self.assertEqual(violations[0].severity, Severity.CRITICAL)

    def test_human_object_pose_warning(self):
        guard = HumanProximityGuard(min_distance_m=0.5, over_limit_strikes=2)
        obs = _obs(
            ee_position=jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float32),
            objects={"operator_hand": ((0.2, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))},
        )
        violations = guard.check_observation(obs)
        self.assertEqual(violations[0].severity, Severity.WARNING)


if __name__ == "__main__":
    unittest.main()
