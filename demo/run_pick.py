"""Kuka pick-and-place demo.

Edit SIMULATOR below, then from repo root:

    python demo/run_pick.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

# --- edit here ---
SIMULATOR = "mujoco"  # mujoco | rviz | gazebo
SEED = 0
# -----------------

_DEMO_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _DEMO_DIR.parent
_CONFIG = _DEMO_DIR / "config" / "kuka_pick.yaml"

_LABELS = {"mujoco": "MuJoCo", "rviz": "RViz", "gazebo": "Gazebo"}


def _ensure_repo_on_path() -> None:
    root = str(_REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _load_config(simulator: str) -> dict:
    raw = yaml.safe_load(_CONFIG.read_text(encoding="utf-8"))
    key = str(simulator).strip().lower()
    if key not in raw:
        raise ValueError(f"Unknown simulator {key!r}; use mujoco, rviz, or gazebo.")
    return dict(raw[key])


def main() -> int:
    _ensure_repo_on_path()
    from robodeploy.env import RoboEnv

    cfg = _load_config(SIMULATOR)
    steps = int(cfg.get("max_episode_steps", 2000))
    cfg["max_episode_steps"] = steps
    task_kwargs = dict(cfg.get("task_kwargs") or {})
    task_kwargs["max_steps"] = steps
    cfg["task_kwargs"] = task_kwargs

    label = _LABELS.get(SIMULATOR, SIMULATOR)
    print(f"Kuka pick & place ({label})")

    env = RoboEnv.from_config(cfg)
    try:
        env.reset(seed=SEED)
        info = None
        for step in range(steps):
            _, _, done, info = env.step()
            if step % 100 == 0:
                print(f"step {step}: success={info.success}")
            if done:
                break
        if info and info.success:
            print("Success.")
            return 0
        print("Done.")
        return 1
    except KeyboardInterrupt:
        print("Stopped.")
        return 130
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
