from __future__ import annotations

import unittest

import numpy as np

try:
    import jax.numpy as jnp
except Exception:
    import numpy as jnp  # type: ignore[assignment]

from robodeploy.core.types import Action, Observation
from robodeploy.tasks.reward_builder import RewardBuilder
from robodeploy.tasks.success_predicates import get_success_predicate


def _obs(**kwargs) -> Observation:
    defaults = dict(
        joint_positions=jnp.zeros((7,), dtype=jnp.float32),
        joint_velocities=jnp.zeros((7,), dtype=jnp.float32),
        joint_torques=jnp.zeros((7,), dtype=jnp.float32),
        ee_position=jnp.asarray([0.6, 0.2, 0.41], dtype=jnp.float32),
        ee_orientation=jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
        ee_velocity=jnp.zeros((3,), dtype=jnp.float32),
        ee_angular_velocity=jnp.zeros((3,), dtype=jnp.float32),
        contact_state={},
    )
    defaults.update(kwargs)
    return Observation(**defaults)


class SensorTaskIntegrationTests(unittest.TestCase):
    def test_grasp_force_min_predicate(self):
        fn = get_success_predicate("grasp_force_min")
        weak = _obs(ft_force=jnp.asarray([0.5, 0.0, 0.0], dtype=jnp.float32))
        strong = _obs(ft_force=jnp.asarray([3.0, 0.0, 0.0], dtype=jnp.float32))
        self.assertFalse(fn(weak, threshold_N=2.0))
        self.assertTrue(fn(strong, threshold_N=2.0))

    def test_contact_held_predicate(self):
        fn = get_success_predicate("contact_held")
        self.assertFalse(fn(_obs()))
        self.assertTrue(fn(_obs(contact_state={"wrist_contact": True})))

    def test_imu_stable_distinguishes_motion(self):
        fn = get_success_predicate("imu_stable")
        settled = _obs(
            imu_angular_velocity=jnp.asarray([0.05, 0.0, 0.0], dtype=jnp.float32),
            imu_acceleration=jnp.asarray([0.0, 0.0, 9.81], dtype=jnp.float32),
        )
        swinging = _obs(
            imu_angular_velocity=jnp.asarray([1.0, 0.5, 0.0], dtype=jnp.float32),
            imu_acceleration=jnp.asarray([2.0, 1.0, 9.0], dtype=jnp.float32),
        )
        self.assertTrue(fn(settled))
        self.assertFalse(fn(swinging))

    def test_penalty_excessive_force_reward(self):
        obs_high = _obs(ft_force=jnp.asarray([25.0, 0.0, 0.0], dtype=jnp.float32))
        obs_low = _obs(ft_force=jnp.asarray([5.0, 0.0, 0.0], dtype=jnp.float32))
        reward_fn = RewardBuilder().penalty_excessive_force(threshold_N=20.0, scale=0.1).build()
        self.assertLess(reward_fn(obs_high, Action()), 0.0)
        self.assertEqual(reward_fn(obs_low, Action()), 0.0)

    def test_bonus_grasp_force_in_band(self):
        obs = _obs(ft_force=jnp.asarray([3.0, 0.0, 0.0], dtype=jnp.float32))
        reward_fn = RewardBuilder().bonus_grasp_force(min_N=1.0, max_N=5.0, scale=0.1).build()
        self.assertGreater(reward_fn(obs, Action()), 0.0)

    def test_bonus_visual_alignment(self):
        rgb = np.zeros((32, 32, 3), dtype=np.uint8)
        rgb[8:20, 8:20] = (255, 0, 0)
        obs = _obs(rgb=rgb)
        reward_fn = (
            RewardBuilder()
            .bonus_visual_alignment(
                target_hsv_range=((0.0, 80.0, 80.0), (10.0, 255.0, 255.0)),
                scale=0.2,
                min_pixels=20,
            )
            .build()
        )
        self.assertGreater(reward_fn(obs, Action()), 0.0)


if __name__ == "__main__":
    unittest.main()
