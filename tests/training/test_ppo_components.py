from __future__ import annotations

import unittest

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from robodeploy.training.ppo import ActorCritic, PPOConfig, PPOTrainer, compute_gae, ppo_clip_loss


class PPOComponentTests(unittest.TestCase):
    def test_gae_matches_manual_bootstrap(self):
        rewards = np.array([1.0, 0.0, 2.0], dtype=np.float32)
        values = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        dones = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        advantages, returns = compute_gae(
            rewards,
            values,
            dones,
            gamma=0.99,
            gae_lambda=0.95,
            last_value=0.0,
        )
        self.assertEqual(advantages.shape, rewards.shape)
        self.assertEqual(returns.shape, rewards.shape)
        self.assertTrue(np.isfinite(advantages).all())

    def test_ppo_clip_prefers_bounded_ratio(self):
        log_probs = torch.tensor([0.0, 0.0], dtype=torch.float32)
        old_log_probs = torch.tensor([0.0, 0.0], dtype=torch.float32)
        advantages = torch.tensor([1.0, -1.0], dtype=torch.float32)
        loss = ppo_clip_loss(log_probs, old_log_probs, advantages, clip_range=0.2)
        self.assertGreater(float(loss.item()), -2.0)

    def test_actor_critic_value_head_shape(self):
        model = ActorCritic(obs_dim=6, action_dim=2)
        obs = torch.zeros(4, 6)
        dist, value = model.forward(obs)
        self.assertEqual(value.shape, (4,))
        sample = dist.sample()
        self.assertEqual(sample.shape, (4, 2))

    def test_ppo_trainer_short_fit(self):
        from functools import partial

        from robodeploy.training.gym_register import robodeploy_dummy_gym_env_factory
        from robodeploy.training.parallel_vec_env import SubprocVecEnv

        env_fn = partial(robodeploy_dummy_gym_env_factory, max_episode_steps=20)
        vec = SubprocVecEnv([env_fn for _ in range(2)])
        try:
            model = ActorCritic(obs_dim=6, action_dim=2)
            cfg = PPOConfig(
                n_envs=2,
                rollout_steps=32,
                total_steps=64,
                minibatch_size=16,
                n_epochs=2,
            )
            trainer = PPOTrainer(env=vec, model=model, config=cfg, obs_key="proprio")
            metrics = trainer.fit()
            self.assertIn("policy_loss", metrics)
        finally:
            vec.close()


if __name__ == "__main__":
    unittest.main()
