"""Kuka multimodal pick-place demo via ``kuka_ft_imu_pick_gazebo`` preset.

Requires Linux: ROS 2 Jazzy, Gazebo Harmonic (``gz``), ``ros_gz_bridge``, ``gz_ros2_control``.
Optional: ``pip install -e ".[kinematics]"`` for Pinocchio reach IK on URDF.
"""

from __future__ import annotations

import sys
import time

from examples._bootstrap import ensure_repo_on_path

ensure_repo_on_path()

from examples.kuka_ft_imu_pick_gazebo.pick_episode import (  # noqa: E402
    _ft_norm,
    run_pick_episode,
)


def main() -> int:
    try:
        result = run_pick_episode(
            seed=None,
            max_steps=1500,
            on_step=_print_progress,
        )
    except Exception as exc:
        print("Failed to start Gazebo demo:", exc)
        print('\nRequires Linux: gz + ROS2 Jazzy + ros_gz_bridge + gz_ros2_control.')
        print('Optional reach IK: pip install -e ".[kinematics]"')
        print("See docs/BACKEND_SETUP.md#gazebo-harmonic-gz-sim-simulator-path")
        return 2

    info = result.final_info
    if info is not None:
        print("done at step", result.steps, "success=", result.success)
        if result.source_to_goal_distance is not None:
            print(f"source_to_goal_distance={result.source_to_goal_distance:.4f}")

    if not result.sensor_health_ok:
        health = (getattr(info, "extra", None) or {}).get("sensor_health", {})
        print("Sensor health failed:", health)
        return 2
    if not result.success:
        print("Episode did not succeed (info.success or placement tolerance).")
        return 1
    return 0


def _print_progress(step: int, obs, info, _env) -> None:
    if step == 0:
        objs = getattr(obs, "objects", {})
        print(
            "reset episode",
            getattr(info, "episode_id", None),
            "objects",
            list(objs.keys()),
            "sensor_health=",
            (getattr(info, "extra", None) or {}).get("sensor_health"),
        )
        print(
            "obs keys:",
            "images" if obs.images else "images=empty",
            "ft_forces" if obs.ft_forces else "ft_forces=empty",
            "imu" if getattr(obs, "imu_angular_velocity", None) is not None else "imu=empty",
            "contact" if getattr(obs, "contact_state", None) else "contact=empty",
        )
    if step % 100 != 0:
        return
    objs = getattr(obs, "objects", {}) or {}
    src = objs.get("source", ((0, 0, 0), (1, 0, 0, 0)))[0]
    imu = getattr(obs, "imu_angular_velocity", None)
    contact = getattr(obs, "contact_state", {}) or {}
    has_img = obs.images.get("wrist_camera") is not None
    print(
        f"step {step:4d} success={getattr(info, 'success', False)} "
        f"source_z={src[2]:.3f} ft_norm={_ft_norm(obs):.3f} "
        f"imu_omega={float((imu ** 2).sum() ** 0.5) if imu is not None else 0:.3f} "
        f"contact={contact.get('wrist_contact', False)} "
        f"camera={'yes' if has_img else 'no'} "
        f"health={(getattr(info, 'extra', None) or {}).get('sensor_health', {}).get('overall')}"
    )
    time.sleep(0.01)


if __name__ == "__main__":
    sys.exit(main())
