"""PPO integration on the reach_target dummy benchmark (GOAL 02)."""

from __future__ import annotations

import unittest
from functools import partial

import pytest

torch = pytest.importorskip("torch")

from robodeploy.training.gym_register import reach_target_dummy_gym_env_factory
from robodeploy.training.parallel_vec_env import SubprocVecEnv
from robodeploy.training.ppo import (
    ActorCritic,
    PPOConfig,
    PPOTrainer,
    evaluate_actor_critic,
)


class PPOReachTargetTests(unittest.TestCase):
    @pytest.mark.slow
    def test_ppo_reach_target_dummy_success_rate(self):
        env_fn = partial(reach_target_dummy_gym_env_factory, max_episode_steps=300)
        probe = env_fn()
        obs, _ = probe.reset()
        obs_dim = int(obs["proprio"].shape[0])
        action_dim = int(probe.action_space.shape[0])
        probe.close()

        vec = SubprocVecEnv([env_fn for _ in range(4)])
        try:
            model = ActorCritic(obs_dim, action_dim, hidden=(32, 32))
            cfg = PPOConfig(
                n_envs=4,
                rollout_steps=128,
                total_steps=10_000,
                minibatch_size=32,
                n_epochs=3,
                lr=5e-3,
                seed=0,
            )
            trainer = PPOTrainer(env=vec, model=model, config=cfg)
            trainer.fit()
        finally:
            vec.close()

        metrics = evaluate_actor_critic(
            model,
            env_fn,
            n_episodes=20,
            deterministic=True,
        )
        self.assertGreaterEqual(metrics["eval/success_rate"], 0.8)


if __name__ == "__main__":
    unittest.main()
