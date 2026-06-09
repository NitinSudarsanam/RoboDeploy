"""Kuka multimodal pick-place demo via ``kuka_ft_imu_pick_gazebo`` preset.

Requires Linux: ROS 2 Jazzy, Gazebo Harmonic (``gz``), ``ros_gz_bridge``, ``gz_ros2_control``.
Optional: ``pip install -e ".[kinematics]"`` for Pinocchio reach IK on URDF.
"""

from __future__ import annotations

import time

import numpy as np

from examples._bootstrap import ensure_repo_on_path

ensure_repo_on_path()

from examples.env_from_preset import env_from_preset  # noqa: E402


def _ft_norm(obs) -> float:
    ft = getattr(obs, "ft_forces", {}) or {}
    wrench = ft.get("wrist_ft")
    if wrench is None:
        return 0.0
    return float(np.linalg.norm(np.asarray(wrench, dtype=np.float32)))


def main() -> None:
    try:
        env = env_from_preset("kuka_ft_imu_pick_gazebo", max_episode_steps=1500)
    except Exception as exc:
        print("Failed to start Gazebo demo:", exc)
        print('\nRequires Linux: gz + ROS2 Jazzy + ros_gz_bridge + gz_ros2_control.')
        print('Optional reach IK: pip install -e ".[kinematics]"')
        print("See docs/BACKEND_SETUP.md#gazebo-harmonic-gz-sim-simulator-path")
        return

    try:
        obs, info = env.reset()
        objs = getattr(obs, "objects", {})
        print(
            "reset episode",
            info.episode_id,
            "objects",
            list(objs.keys()),
            "sensor_health=",
            info.extra.get("sensor_health"),
        )
        print(
            "obs keys:",
            "images" if obs.images else "images=empty",
            "ft_forces" if obs.ft_forces else "ft_forces=empty",
            "imu" if getattr(obs, "imu_angular_velocity", None) is not None else "imu=empty",
            "contact" if getattr(obs, "contact_state", None) else "contact=empty",
        )
        for i in range(1500):
            obs, reward, done, info = env.step()
            if i % 100 == 0:
                src = objs.get("source", ((0, 0, 0), (1, 0, 0, 0)))[0]
                objs = getattr(obs, "objects", {})
                if "source" in objs:
                    src = objs["source"][0]
                imu = getattr(obs, "imu_angular_velocity", None)
                contact = getattr(obs, "contact_state", {})
                has_img = obs.images.get("wrist_camera") is not None
                print(
                    f"step {i:4d} reward={reward:7.3f} success={info.success} "
                    f"source_z={src[2]:.3f} ft_norm={_ft_norm(obs):.3f} "
                    f"imu_omega={float((imu ** 2).sum() ** 0.5) if imu is not None else 0:.3f} "
                    f"contact={contact.get('wrist_contact', False)} "
                    f"camera={'yes' if has_img else 'no'} "
                    f"health={info.extra.get('sensor_health', {}).get('overall')}"
                )
            if done:
                print("done at step", i, "success=", info.success)
                break
            time.sleep(0.01)
    finally:
        env.close()


if __name__ == "__main__":
    main()
