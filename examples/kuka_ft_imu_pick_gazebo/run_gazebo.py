"""Kuka multimodal pick-place demo via ``kuka_ft_imu_pick_gazebo`` preset.

Requires Linux: ROS 2 Jazzy, Gazebo Harmonic (``gz``), ``ros_gz_bridge``, ``gz_ros2_control``.
Optional: ``pip install pin`` for Pinocchio reach IK on URDF.
"""

from __future__ import annotations

import time

from examples._bootstrap import ensure_repo_on_path

ensure_repo_on_path()

from examples.env_from_preset import env_from_preset  # noqa: E402


def main() -> None:
    try:
        env = env_from_preset("kuka_ft_imu_pick_gazebo", max_episode_steps=1500)
    except Exception as exc:
        print("Failed to start Gazebo demo:", exc)
        print("\nRequires gz + ROS2 Jazzy + ros_gz_bridge + gz_ros2_control on Linux.")
        return

    try:
        obs, info = env.reset()
        print("reset episode", info.episode_id, "objects", list(getattr(obs, "objects", {}).keys()))
        for i in range(1500):
            obs, reward, done, info = env.step()
            if i % 100 == 0:
                objs = getattr(obs, "objects", {})
                src = objs.get("source", ((0, 0, 0), (1, 0, 0, 0)))[0]
                imu = getattr(obs, "imu_angular_velocity", None)
                contact = getattr(obs, "contact_state", {})
                print(
                    f"step {i:4d} reward={reward:7.3f} success={info.success} "
                    f"source_z={src[2]:.3f} imu={'yes' if imu is not None else 'no'} "
                    f"contact={contact.get('wrist_contact', False)}"
                )
            if done:
                print("done at step", i, "success=", info.success)
                break
            time.sleep(0.01)
    finally:
        env.close()


if __name__ == "__main__":
    main()
