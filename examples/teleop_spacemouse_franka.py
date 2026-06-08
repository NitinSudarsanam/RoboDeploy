"""SpaceMouse teleop on Franka MuJoCo preset with optional recording.

Run from repo root::

    python -m examples.teleop_spacemouse_franka --record demos/franka_spacemouse.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.env_from_preset import env_from_preset
from robodeploy.teleop.session import run_teleop_session


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SpaceMouse teleop on franka_pick_mujoco.")
    parser.add_argument("--preset", default="franka_pick_mujoco", help="YAML preset name.")
    parser.add_argument("--record", default="", help="Output .jsonl path or directory.")
    parser.add_argument("--max-steps", type=int, default=0, help="Stop after N steps (0 = unlimited).")
    args = parser.parse_args(argv)

    env = env_from_preset(str(args.preset))
    try:
        saved = run_teleop_session(
            env,
            device="spacemouse",
            record_path=str(args.record) if args.record else None,
            max_steps=int(args.max_steps) if int(args.max_steps) > 0 else None,
        )
        for path in saved:
            print(path)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
