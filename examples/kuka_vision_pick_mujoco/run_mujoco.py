"""Hybrid vision pick: color blob estimates source, prop_pose sensor reads target.

Requires: pip install -e ".[sim]"
Skipped on Windows in CI when MuJoCo Renderer is unstable.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path


def _ensure_repo_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


_ensure_repo_on_path()

from examples.env_from_preset import env_from_preset  # noqa: E402


def main() -> None:
    if sys.platform == "win32":
        print("Vision pick demo uses MuJoCo Renderer; skip on Windows headless hosts.")
        return
    try:
        env = env_from_preset("kuka_vision_pick_mujoco", max_episode_steps=1500)
    except ImportError as exc:
        print(exc)
        print('\nInstall MuJoCo support:\n  pip install -e ".[sim]"')
        return

    try:
        obs, info = env.reset()
        print("reset", info.episode_id, "objects", list(getattr(obs, "objects", {}).keys()))
        for i in range(1500):
            obs, reward, done, info = env.step()
            if i % 100 == 0:
                print(f"step {i} reward={reward:.3f} success={info.success} objects={list(obs.objects.keys())}")
            if done:
                print("done", i, "success=", info.success)
                break
            time.sleep(0.003)
    finally:
        env.close()


if __name__ == "__main__":
    main()
