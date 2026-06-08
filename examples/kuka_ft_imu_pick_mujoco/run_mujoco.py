"""Kuka FT + IMU sensor-driven pick-place demo (MuJoCo).

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
        env = env_from_preset("kuka_ft_imu_pick_mujoco", max_episode_steps=1500)
    except ImportError as exc:
        print(exc)
        print('\nInstall MuJoCo support:\n  pip install -e ".[sim]"')
        return

    try:
        obs, info = env.reset()
        print(
            "reset episode",
            info.episode_id,
            "sensor_health=",
            info.extra.get("sensor_health"),
        )
        for i in range(1500):
            obs, reward, done, info = env.step()
            if i % 100 == 0:
                ft = getattr(obs, "ft_force", None)
                imu = getattr(obs, "imu_angular_velocity", None)
                contact = getattr(obs, "contact_state", {})
                print(
                    f"step {i:4d} reward={reward:7.3f} success={info.success} "
                    f"ft_norm={float((ft ** 2).sum() ** 0.5) if ft is not None else 0:.3f} "
                    f"imu_omega={float((imu ** 2).sum() ** 0.5) if imu is not None else 0:.3f} "
                    f"contact={contact} health={info.extra.get('sensor_health', {}).get('overall')}"
                )
            if done:
                print("done at step", i, "success=", info.success)
                break
            time.sleep(0.003)
    finally:
        env.close()


if __name__ == "__main__":
    main()
