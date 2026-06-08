from __future__ import annotations

import unittest

try:
    import jax.numpy as jnp
except Exception:
    import numpy as jnp  # type: ignore[assignment]

from robodeploy.core.types import Observation
from robodeploy.safety import CollisionGuard, Hazard


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


class _ContactBackend:
    def has_prop_contact(self, prop_name: str, *, other_body: str | None = None) -> bool:
        del other_body
        return prop_name == "cube"


class CollisionGuardTests(unittest.TestCase):
    def test_disallowed_pair_triggers_violation(self):
        guard = CollisionGuard(
            backend=_ContactBackend(),
            disallowed_pairs=[("cube", "ee_link")],
        )
        violations = guard.check_observation(_obs())
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].hazard, Hazard.COLLISION_IMMINENT)

    def test_no_contact_is_clean(self):
        guard = CollisionGuard(
            backend=type("B", (), {"has_prop_contact": staticmethod(lambda *_a, **_k: False)})(),
            disallowed_pairs=[("cube", "ee_link")],
        )
        violations = guard.check_observation(_obs())
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
