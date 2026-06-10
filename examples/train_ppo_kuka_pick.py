"""Train PPO on kuka_pick_mujoco (reach-to-cube baseline for WAVE2_04).

Thin wrapper around ``train_ppo_reach.py --preset kuka_pick_mujoco``.

Usage:
    python examples/train_ppo_kuka_pick.py --total-steps 100000 --n-envs 8
"""

from __future__ import annotations

import sys
from pathlib import Path

_EXAMPLES_DIR = Path(__file__).resolve().parent

if __name__ == "__main__":
    if "--preset" not in sys.argv:
        sys.argv[1:1] = ["--preset", "kuka_pick_mujoco"]
    if str(_EXAMPLES_DIR) not in sys.path:
        sys.path.insert(0, str(_EXAMPLES_DIR))
    from train_ppo_reach import main

    raise SystemExit(main())
