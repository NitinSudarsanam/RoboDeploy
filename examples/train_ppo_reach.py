"""Production-scale PPO on the reach_target benchmark (GOAL 02 / WAVE2_04).

Usage:
    python examples/train_ppo_reach.py --backend dummy --total-steps 500000
    python examples/train_ppo_reach.py --backend mujoco --total-steps 500000 --n-envs 8
"""

from __future__ import annotations

import argparse
import json
import tempfile
from functools import partial
from pathlib import Path
from typing import Any, Callable

import torch

from robodeploy.training.parallel_vec_env import SubprocVecEnv
from robodeploy.training.ppo import (
    ActorCritic,
    PPOConfig,
    PPOTrainer,
    evaluate_actor_critic,
)


def _reach_target_mujoco_env_factory(*, max_episode_steps: int = 300, seed: int = 0):
    import yaml

    from robodeploy.evaluation.env_builder import build_env_from_preset
    from robodeploy.training.gym_adapter import GymRoboEnv

    preset_path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks/manipulation_v1/reach_target/preset_mujoco.yaml"
    )
    preset = yaml.safe_load(preset_path.read_text(encoding="utf-8"))
    robo = build_env_from_preset(preset, seed=int(seed))
    return GymRoboEnv(robo, max_episode_steps=max_episode_steps)


def _make_env_factory(
    backend: str,
    *,
    max_episode_steps: int,
    seed: int,
) -> Callable[[], Any]:
    if backend == "dummy":
        from robodeploy.training.gym_register import reach_target_dummy_gym_env_factory

        return partial(reach_target_dummy_gym_env_factory, max_episode_steps=max_episode_steps)
    if backend == "mujoco":
        return partial(
            _reach_target_mujoco_env_factory,
            max_episode_steps=max_episode_steps,
            seed=seed,
        )
    raise ValueError(f"unknown backend: {backend!r} (expected dummy or mujoco)")


class EvalCheckpointCallback:
    """Periodic eval + best-checkpoint save for PPOTrainer."""

    def __init__(
        self,
        *,
        env_factory: Callable[[], Any],
        eval_every: int,
        checkpoint_out: Path,
        n_episodes: int = 20,
    ) -> None:
        self.env_factory = env_factory
        self.eval_every = int(eval_every)
        self.checkpoint_out = Path(checkpoint_out)
        self.n_episodes = int(n_episodes)
        self.best_sr = -1.0
        self.last_eval: dict[str, float] = {}

    def on_train_begin(self, trainer: PPOTrainer) -> None:
        self.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)

    def on_step_end(self, trainer: PPOTrainer, metrics: dict[str, float]) -> None:
        del metrics
        if trainer.global_step == 0 or trainer.global_step % self.eval_every != 0:
            return
        self.last_eval = evaluate_actor_critic(
            trainer.model,
            self.env_factory,
            n_episodes=self.n_episodes,
            deterministic=True,
        )
        sr = float(self.last_eval.get("eval/success_rate", 0.0))
        print(
            f"[eval] step={trainer.global_step} "
            f"success_rate={sr:.3f} "
            f"mean_reward={self.last_eval.get('eval/mean_reward', 0.0):.3f}"
        )
        if sr >= self.best_sr:
            self.best_sr = sr
            torch.save(
                {
                    "policy": trainer.model.state_dict(),
                    "config": trainer.config,
                    "global_step": trainer.global_step,
                    "eval": self.last_eval,
                },
                self.checkpoint_out,
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train PPO on reach_target (dummy or MuJoCo).")
    parser.add_argument("--backend", choices=("dummy", "mujoco"), default="dummy")
    parser.add_argument("--total-steps", type=int, default=500_000)
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--eval-every", type=int, default=50_000)
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--checkpoint-out", default=None)
    args = parser.parse_args()

    log_dir = Path(args.log_dir or tempfile.mkdtemp(prefix="robodeploy_ppo_reach_"))
    checkpoint_out = Path(args.checkpoint_out or log_dir / "ppo_reach_best.pt")
    env_fn = _make_env_factory(
        args.backend,
        max_episode_steps=int(args.max_episode_steps),
        seed=int(args.seed),
    )

    probe = env_fn()
    try:
        obs_dim = int(probe.observation_space["proprio"].shape[0])
        action_dim = int(probe.action_space.shape[0])
    finally:
        probe.close()

    cfg = PPOConfig(
        n_envs=int(args.n_envs),
        total_steps=int(args.total_steps),
        rollout_steps=int(args.rollout_steps),
        log_dir=str(log_dir),
        seed=int(args.seed),
    )
    eval_cb = EvalCheckpointCallback(
        env_factory=env_fn,
        eval_every=int(args.eval_every),
        checkpoint_out=checkpoint_out,
    )
    vec = SubprocVecEnv([env_fn for _ in range(int(args.n_envs))])
    try:
        model = ActorCritic(obs_dim, action_dim)
        trainer = PPOTrainer(env=vec, model=model, config=cfg, callbacks=[eval_cb])
        train_metrics = trainer.fit()
        final_eval = evaluate_actor_critic(model, env_fn, n_episodes=20, deterministic=True)
    finally:
        vec.close()

    final_ckpt = log_dir / "ppo_reach_final.pt"
    torch.save(
        {
            "policy": model.state_dict(),
            "config": cfg,
            "train_metrics": train_metrics,
            "eval": final_eval,
        },
        final_ckpt,
    )

    summary = {
        "backend": args.backend,
        "total_steps": int(args.total_steps),
        "n_envs": int(args.n_envs),
        "best_checkpoint": str(checkpoint_out),
        "final_checkpoint": str(final_ckpt),
        "best_eval_success_rate": eval_cb.best_sr,
        "final_eval": final_eval,
        "train_metrics": train_metrics,
    }
    summary_path = log_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
