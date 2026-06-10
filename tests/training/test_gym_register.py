from __future__ import annotations

import unittest

import pytest

pytest.importorskip("gymnasium")


class GymRegisterTests(unittest.TestCase):
    def test_kuka_pick_mujoco_env_registered(self):
        pytest.importorskip("mujoco", reason="mujoco not installed")
        import gymnasium as gym

        from robodeploy.training.gym_register import register_robodeploy_envs

        register_robodeploy_envs()
        env = gym.make("robodeploy/kuka_pick_mujoco-v0")
        try:
            obs, info = env.reset()
            self.assertIn("proprio", obs)
            self.assertIsInstance(info, dict)
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            self.assertIn("proprio", obs)
            self.assertIsInstance(reward, float)
            self.assertIsInstance(terminated, bool)
            self.assertIsInstance(truncated, bool)
            self.assertIsInstance(info, dict)
        finally:
            env.close()

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
