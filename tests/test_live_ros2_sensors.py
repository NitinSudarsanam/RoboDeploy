"""Live ROS2 sensor integration (run in sensor-live-ros2 CI job)."""

from __future__ import annotations

import os
import sys
import time
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_TESTS_DIR = REPO_ROOT / "tests"
for _p in (str(REPO_ROOT), str(_TESTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from live_sensor_fixtures import LiveRos2SensorPublishers, rclpy_available

LIVE = os.environ.get("ROBODEPLOY_LIVE_ROS2", "").strip() in {"1", "true", "yes"}


def _obs_ready(obs) -> bool:
    return (
        obs.images.get("wrist_camera") is not None
        and obs.ft_forces.get("wrist_ft") is not None
        and bool(obs.camera_intrinsics.get("wrist_camera"))
    )


@unittest.skipUnless(LIVE, "set ROBODEPLOY_LIVE_ROS2=1 to run live ROS2 sensor tests")
@unittest.skipUnless(rclpy_available(), "rclpy not available")
class LiveRos2SensorTests(unittest.TestCase):
    def test_kuka_sensor_ros2_rviz_preset_populates_observation(self):
        from examples.env_from_preset import env_from_preset

        pubs = LiveRos2SensorPublishers(robot_id="robot0")
        time.sleep(0.3)
        env = env_from_preset("kuka_sensor_ros2_rviz", max_episode_steps=20)
        try:
            obs, _ = env.reset()
            deadline = time.monotonic() + 12.0
            while not _obs_ready(obs) and time.monotonic() < deadline:
                obs, _, _, _ = env.step()
            self.assertTrue(_obs_ready(obs), msg=f"sensor_status={getattr(obs, 'sensor_status', {})}")
            extrinsics = obs.camera_extrinsics.get("wrist_camera")
            if extrinsics:
                self.assertEqual(extrinsics.get("source"), "tf")
        finally:
            env.close()
            pubs.close()


if __name__ == "__main__":
    unittest.main()
