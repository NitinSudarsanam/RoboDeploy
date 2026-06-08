from __future__ import annotations

import unittest

import pytest

pytest.importorskip("gymnasium")
sb3 = pytest.importorskip("stable_baselines3")


class SB3SmokeTests(unittest.TestCase):
    def test_gym_make_and_ppo_learn(self):
        import gymnasium as gym

        from robodeploy.training.gym_register import register_robodeploy_envs

        register_robodeploy_envs()
        env = gym.make("robodeploy/Dummy-v0")
        try:
            model = sb3.PPO(
                "MultiInputPolicy",
                env,
                n_steps=32,
                batch_size=32,
                verbose=0,
            )
            model.learn(total_timesteps=128)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
