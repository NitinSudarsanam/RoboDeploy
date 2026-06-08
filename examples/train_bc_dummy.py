"""Minimal BC training example on dummy RoboEnv demos.

Usage:
    python examples/train_bc_dummy.py --epochs 20 --out /tmp/bc_dummy.pt
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from robodeploy.cli_helpers import action_fn_for_mode, close_quietly
from robodeploy.cli import _make_dummy_env
from robodeploy.dataset_export import export_recorded_episode
from robodeploy.training.bc import train_bc
from robodeploy.training.dataset import DemoDataset
from robodeploy.training.gym_adapter import GymRoboEnv
from robodeploy.training.trainer import TrainerConfig


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    log_dir = Path(args.log_dir or tempfile.mkdtemp(prefix="robodeploy_bc_"))
    dataset_path = log_dir / "demos.jsonl"

    env = _make_dummy_env()
    try:
        export_recorded_episode(
            env,
            steps=80,
            path=dataset_path,
            action_fn=action_fn_for_mode("sinusoid", env),
        )
        demo = DemoDataset.from_jsonl(dataset_path)
        eval_env = GymRoboEnv(_make_dummy_env(), max_episode_steps=40)
        cfg = TrainerConfig(
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            log_dir=str(log_dir),
            checkpoint_interval=10_000,
        )
        train_bc(dataset=demo, config=cfg, eval_env=eval_env)
        out = Path(args.out or log_dir / "bc_final.pt")
        print(out)
        close_quietly(eval_env.robo_env)
    finally:
        close_quietly(env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
