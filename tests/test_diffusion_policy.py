from __future__ import annotations

import unittest

try:
    import jax.numpy as jnp
except Exception:
    import numpy as jnp  # type: ignore[assignment]

from robodeploy.core.types import Action, Observation
from robodeploy.policies.learned.diffusion import DiffusionPolicy


def make_obs(*, instruction: str | None = None) -> Observation:
    return Observation(
        joint_positions=jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float32),
        joint_velocities=jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float32),
        joint_torques=jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float32),
        ee_position=jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float32),
        ee_orientation=jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
        ee_velocity=jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float32),
        ee_angular_velocity=jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float32),
        language_instruction=instruction,
    )


class DiffusionPolicyTests(unittest.TestCase):
    def test_predict_plan_rolls_out_multiple_actions(self):
        calls = {"count": 0}

        def predict_plan(packet):
            calls["count"] += 1
            self.assertEqual(packet["instruction"], "move left")
            return {
                "actions": [
                    {"ee_position": [0.1, 0.0, 0.0]},
                    {"ee_position": [0.05, 0.0, 0.0]},
                ]
            }

        policy = DiffusionPolicy(
            config={
                "predict_plan_fn": predict_plan,
                "action_space": "delta_ee",
                "replan_interval": 2,
            }
        )
        obs = make_obs(instruction="move left")

        first = policy.get_action(obs)
        second = policy.get_action(obs)

        self.assertEqual(calls["count"], 1)
        self.assertAlmostEqual(float(first.ee_position[0]), 0.1)
        self.assertAlmostEqual(float(second.ee_position[0]), 0.05)

    def test_notify_rejected_forces_replan(self):
        calls = {"count": 0}

        def predict_plan(packet):
            del packet
            calls["count"] += 1
            return [{"ee_position": [0.1 * calls["count"], 0.0, 0.0]}]

        policy = DiffusionPolicy(
            config={
                "predict_plan_fn": predict_plan,
                "action_space": "delta_ee",
                "replan_interval": 4,
            }
        )
        obs = make_obs(instruction="move forward")

        first = policy.get_action(obs)
        policy.notify_rejected(obs, first)
        second = policy.get_action(obs)

        self.assertEqual(calls["count"], 2)
        self.assertNotEqual(float(first.ee_position[0]), float(second.ee_position[0]))

    def test_batch_predictor_returns_first_action_from_each_plan(self):
        def predict_batch_plan(packets):
            self.assertEqual([packet["instruction"] for packet in packets], ["one", "two"])
            return [
                [{"joint_positions": [1.0, 2.0, 3.0]}],
                [{"joint_positions": [4.0, 5.0, 6.0]}],
            ]

        policy = DiffusionPolicy(
            config={
                "predict_batch_plan_fn": predict_batch_plan,
                "action_space": "joint_pos",
            }
        )
        actions = policy.get_action_batch([make_obs(instruction="one"), make_obs(instruction="two")])

        self.assertEqual(float(actions[0].joint_positions[0]), 1.0)
        self.assertEqual(float(actions[1].joint_positions[0]), 4.0)

    def test_fallback_plan_uses_instruction_direction(self):
        policy = DiffusionPolicy(config={"action_space": "delta_ee", "plan_horizon": 3, "max_delta": 0.1})
        action = policy.get_action(make_obs(instruction="move right"))
        self.assertTrue(action.is_delta_ee)
        self.assertLess(float(action.ee_position[1]), 0.0)


if __name__ == "__main__":
    unittest.main()
