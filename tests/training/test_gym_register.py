from __future__ import annotations

import unittest

import pytest

pytest.importorskip("gymnasium")


class GymRegisterTests(unittest.TestCase):
    def test_gym_make_dummy_env(self):
        import gymnasium as gym

        from robodeploy.training.gym_register import register_robodeploy_envs

        register_robodeploy_envs()
        spec = gym.spec("robodeploy/Dummy-v0")
        self.assertIsNotNone(spec)
        env = gym.make("robodeploy/Dummy-v0")
        try:
            obs, info = env.reset()
            self.assertIn("proprio", obs)
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            self.assertIsInstance(reward, float)
            self.assertIsInstance(info, dict)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
