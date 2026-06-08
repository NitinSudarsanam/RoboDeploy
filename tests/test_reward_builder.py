from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class RewardBuilderTests(unittest.TestCase):
    def test_distance_reward_negative(self):
        from robodeploy.core.types import Action, Observation
        from robodeploy.tasks.reward_builder import RewardBuilder

        try:
            import jax.numpy as jnp
        except Exception:
            import numpy as jnp  # type: ignore[assignment]

        obs = Observation(
            joint_positions=jnp.zeros(7),
            joint_velocities=jnp.zeros(7),
            joint_torques=jnp.zeros(7),
            ee_position=jnp.array([0.0, 0.0, 0.5]),
            ee_orientation=jnp.array([1.0, 0.0, 0.0, 0.0]),
            ee_velocity=jnp.zeros(3),
            ee_angular_velocity=jnp.zeros(3),
            objects={"source": ((0.5, 0.0, 0.4), (1.0, 0.0, 0.0, 0.0))},
        )
        reward_fn = (
            RewardBuilder()
            .distance("ee", "source", scale=1.0, name="reach")
            .build()
        )
        value = reward_fn(obs, Action())
        self.assertLess(value, 0.0)

    def test_build_components_named(self):
        from robodeploy.core.types import Action, Observation
        from robodeploy.tasks.reward_builder import RewardBuilder

        try:
            import jax.numpy as jnp
        except Exception:
            import numpy as jnp  # type: ignore[assignment]

        obs = Observation(
            joint_positions=jnp.zeros(2),
            joint_velocities=jnp.zeros(2),
            joint_torques=jnp.zeros(2),
            ee_position=jnp.array([0.0, 0.0, 0.0]),
            ee_orientation=jnp.array([1.0, 0.0, 0.0, 0.0]),
            ee_velocity=jnp.zeros(3),
            ee_angular_velocity=jnp.zeros(3),
        )
        components = RewardBuilder().penalty_action_norm(scale=0.01).build_components()
        out = components(obs, Action(joint_positions=jnp.ones(2)))
        self.assertIn("action_norm", out)
