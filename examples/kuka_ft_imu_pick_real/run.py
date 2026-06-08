"""Kuka FT + IMU multi-modal real-hardware demo.

Requires ROS2 bridge or ATI NetFT + Xsens serial. See README.md.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from examples._bootstrap import ensure_repo_on_path

ensure_repo_on_path()

from examples.env_from_preset import env_from_preset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        default="kuka_ft_imu_multimodal_ros2",
        help="Preset name (kuka_ft_imu_multimodal_ros2 or kuka_ft_imu_multimodal_real)",
    )
    args = parser.parse_args()

    if args.preset.endswith("_real"):
        host = os.environ.get("ATI_NETFT_HOST", "").strip()
        port = os.environ.get("ROBODEPLOY_XSENS_PORT", "").strip()
        if not host:
            print("Hardware blocker: set ATI_NETFT_HOST for native FT sensor.")
        if not port:
            print("Hardware blocker: set ROBODEPLOY_XSENS_PORT for native IMU sensor.")

    try:
        env = env_from_preset(args.preset, max_episode_steps=500)
    except ImportError as exc:
        print(exc)
        print('\nInstall ROS2 support:\n  pip install -e ".[ros2]"')
        return

    try:
        obs, info = env.reset()
        print("reset", info.episode_id, "health=", info.extra.get("sensor_health"))
        for i in range(500):
            obs, reward, done, info = env.step()
            if i % 50 == 0:
                ft = getattr(obs, "ft_force", None)
                imu = getattr(obs, "imu_angular_velocity", None)
                contact = getattr(obs, "contact_state", {})
                health = info.extra.get("sensor_health", {})
                print(
                    f"step {i:4d} ft={float((ft ** 2).sum() ** 0.5) if ft is not None else 0:.2f} "
                    f"imu_omega={float((imu ** 2).sum() ** 0.5) if imu is not None else 0:.3f} "
                    f"contact={contact} sensor_status={getattr(obs, 'sensor_status', {})} "
                    f"health={health.get('overall')}"
                )
            if done:
                print("done success=", info.success)
                break
            time.sleep(0.01)
    finally:
        env.close()


if __name__ == "__main__":
    main()
