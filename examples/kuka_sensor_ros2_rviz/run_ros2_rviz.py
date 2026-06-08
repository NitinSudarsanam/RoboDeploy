"""Run kuka_sensor_ros2_rviz preset and print wrist camera / FT observations."""

from __future__ import annotations

from examples._bootstrap import ensure_repo_on_path

ensure_repo_on_path()
import time
from pathlib import Path



from examples.env_from_preset import env_from_preset  # noqa: E402


def main() -> None:
    try:
        env = env_from_preset("kuka_sensor_ros2_rviz", max_episode_steps=200)
    except ImportError as exc:
        print(exc)
        print("\nRequires ROS2 Python (rclpy) and pip install -e \".[dev,sim]\"")
        return

    try:
        obs, info = env.reset()
        print("reset episode", info.episode_id)
        for i in range(200):
            obs, reward, done, info = env.step()
            if i % 20 == 0:
                img = obs.images.get("wrist_camera")
                ft = obs.ft_forces.get("wrist_ft")
                status = getattr(obs, "sensor_status", {})
                print(
                    f"step {i:3d} reward={reward:6.3f} "
                    f"img={'yes' if img is not None else 'no'} "
                    f"ft={None if ft is None else [float(x) for x in ft]} "
                    f"status={status}"
                )
            if done:
                break
            time.sleep(0.02)
    finally:
        env.close()


if __name__ == "__main__":
    main()
