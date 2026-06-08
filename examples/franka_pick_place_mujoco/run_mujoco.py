"""Franka pick-place demo via ``franka_pick_mujoco`` preset.

Requires: pip install -e ".[sim]"
"""

from __future__ import annotations

import sys
import time

from examples._bootstrap import ensure_repo_on_path

ensure_repo_on_path()

from examples.env_from_preset import env_from_preset  # noqa: E402


def main() -> None:
    try:
        env = env_from_preset("franka_pick_mujoco", max_episode_steps=1500)
    except ImportError as exc:
        print(exc)
        print('\nInstall MuJoCo support:\n  pip install -e ".[sim]"')
        return

    try:
        obs, info = env.reset()
        print("reset episode", info.episode_id, "objects", list(getattr(obs, "objects", {}).keys()))
        for i in range(1500):
            obs, reward, done, info = env.step()
            if i % 100 == 0:
                objs = getattr(obs, "objects", {})
                src = objs.get("source", ((0, 0, 0), (1, 0, 0, 0)))[0]
                print(
                    f"step {i:4d} reward={reward:7.3f} success={info.success} "
                    f"source_z={src[2]:.3f}"
                )
            if done:
                print("done at step", i, "success=", info.success)
                break
            time.sleep(0.003)
    finally:
        env.close()


if __name__ == "__main__":
    main()
