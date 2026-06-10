"""50k-step PPO proxy for production 500k reach_target training (WAVE2_04)."""

from __future__ import annotations

import unittest
from functools import partial

import numpy as np
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


class _LossTracker:
    def __init__(self) -> None:
        self.policy_losses: list[float] = []

    def on_step_end(self, trainer: PPOTrainer, metrics: dict[str, float]) -> None:
        del trainer
        self.policy_losses.append(float(metrics.get("policy_loss", 0.0)))


class PPOReach500kProxyTests(unittest.TestCase):
    @pytest.mark.slow
    @pytest.mark.optional_nightly
    @pytest.mark.flaky(reruns=2, reruns_delay=2)
    def test_ppo_reach_50k_proxy_loss_and_success_rate(self):
        env_fn = partial(reach_target_dummy_gym_env_factory, max_episode_steps=300)
        probe = env_fn()
        try:
            obs_dim = int(probe.observation_space["proprio"].shape[0])
            action_dim = int(probe.action_space.shape[0])
        finally:
            probe.close()

        torch.manual_seed(0)
        np.random.seed(0)
        loss_tracker = _LossTracker()
        vec = SubprocVecEnv([env_fn for _ in range(4)])
        try:
            model = ActorCritic(obs_dim, action_dim, hidden=(32, 32))
            cfg = PPOConfig(
                n_envs=4,
                rollout_steps=256,
                total_steps=50_000,
                minibatch_size=64,
                n_epochs=3,
                lr=5e-3,
                seed=0,
            )
            trainer = PPOTrainer(env=vec, model=model, config=cfg, callbacks=[loss_tracker])
            trainer.fit()
        finally:
            vec.close()

        self.assertGreaterEqual(len(loss_tracker.policy_losses), 2)
        for loss in loss_tracker.policy_losses:
            self.assertTrue(np.isfinite(loss))

        metrics = evaluate_actor_critic(
            model,
            env_fn,
            n_episodes=20,
            deterministic=True,
        )
        self.assertGreaterEqual(metrics["eval/success_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
