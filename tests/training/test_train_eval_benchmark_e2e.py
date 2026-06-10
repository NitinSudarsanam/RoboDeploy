"""Train policies, then evaluate checkpoints on manipulation_v1/reach_target."""

from __future__ import annotations

import json
import tempfile
import unittest
from functools import partial
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[2]


class TrainEvalBenchmarkE2ETests(unittest.TestCase):
    def test_train_bc_then_eval_reach_target_checkpoint(self):
        from robodeploy.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset = tmp_path / "demos.jsonl"
            log_dir = tmp_path / "runs"
            ckpt = log_dir / "bc_final.pt"
            report = tmp_path / "reach_eval.json"

            code = main(
                [
                    "train",
                    "bc",
                    "--dataset",
                    str(dataset),
                    "--dummy",
                    "--epochs",
                    "2",
                    "--batch-size",
                    "4",
                    "--log-dir",
                    str(log_dir),
                ]
            )
            self.assertEqual(code, 0)
            self.assertTrue(ckpt.is_file(), "BC training should write bc_final.pt")

            code = main(
                [
                    "eval",
                    "--benchmark",
                    "manipulation_v1/reach_target",
                    "--policy",
                    str(ckpt),
                    "--backend",
                    "dummy",
                    "--episodes",
                    "3",
                    "--output",
                    str(report),
                    "--benchmarks-root",
                    str(REPO_ROOT / "benchmarks"),
                ]
            )
            self.assertEqual(code, 0)
            self.assertTrue(report.is_file())

            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(payload["benchmark_name"], "manipulation_v1/reach_target")
            self.assertEqual(payload["aggregate"]["n_episodes"], 3)
            self.assertIn("success_rate", payload["aggregate"])
            self.assertEqual(len(payload["episodes"]), 3)

    @unittest.skipUnless(
        __import__("platform").system() != "Windows",
        "MuJoCo PPO train/eval smoke skipped on Windows (Linux CI / sensor-e2e)",
    )
    def test_train_ppo_checkpoint_eval_mujoco_reach_target(self):
        try:
            import mujoco  # noqa: F401
        except ImportError:
            self.skipTest("mujoco not installed")

        from robodeploy.training.parallel_vec_env import SubprocVecEnv
        from robodeploy.training.ppo import ActorCritic, PPOConfig, PPOTrainer, evaluate_actor_critic

        import sys

        examples_dir = REPO_ROOT / "examples"
        if str(examples_dir) not in sys.path:
            sys.path.insert(0, str(examples_dir))
        from train_ppo_reach import _reach_target_mujoco_env_factory

        env_fn = partial(_reach_target_mujoco_env_factory, max_episode_steps=300, seed=0)

        probe = env_fn()
        try:
            obs_dim = int(probe.observation_space["proprio"].shape[0])
            action_dim = int(probe.action_space.shape[0])
        finally:
            probe.close()

        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = Path(tmp) / "ppo_smoke.pt"
            vec = SubprocVecEnv([env_fn for _ in range(2)])
            try:
                model = ActorCritic(obs_dim, action_dim, hidden=(32, 32))
                cfg = PPOConfig(
                    n_envs=2,
                    rollout_steps=64,
                    total_steps=256,
                    minibatch_size=32,
                    n_epochs=2,
                    lr=5e-3,
                    seed=0,
                )
                PPOTrainer(env=vec, model=model, config=cfg).fit()
                torch.save({"policy": model.state_dict(), "config": cfg}, ckpt_path)
            finally:
                vec.close()

            payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            loaded = ActorCritic(obs_dim, action_dim, hidden=(32, 32))
            loaded.load_state_dict(payload["policy"])

            metrics = evaluate_actor_critic(
                loaded,
                env_fn,
                n_episodes=2,
                deterministic=True,
            )
            self.assertIn("eval/success_rate", metrics)
            self.assertIn("eval/mean_reward", metrics)
            self.assertGreaterEqual(metrics["eval/success_rate"], 0.0)
            self.assertLessEqual(metrics["eval/success_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
