"""Replay a recorded JSONL/JSON demo through a preset env.

Run from repo root::

    python -m examples.replay_demo demos/episode_001.jsonl --preset kuka_pick_mujoco
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.env_from_preset import env_from_preset
from robodeploy.teleop.session import replay_recording


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay a RoboDeploy demo recording.")
    parser.add_argument("demo", help="Path to .jsonl or .json demo file.")
    parser.add_argument("--preset", default="kuka_pick_mujoco", help="YAML preset name.")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier.")
    parser.add_argument(
        "--pause-at-step",
        type=int,
        default=-1,
        help="Stop after this step index (-1 = play all).",
    )
    args = parser.parse_args(argv)

    env = env_from_preset(str(args.preset))
    try:
        steps = replay_recording(
            env,
            args.demo,
            speed=float(args.speed),
            pause_at_step=None if int(args.pause_at_step) < 0 else int(args.pause_at_step),
        )
        print(f"replayed {steps} steps from {args.demo}")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
