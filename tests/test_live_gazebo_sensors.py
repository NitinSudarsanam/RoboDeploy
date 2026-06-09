"""Live Gazebo + ROS2 sensor integration (sensor-live-gazebo CI job)."""

from __future__ import annotations

import os
import sys
import time
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
_TESTS_DIR = REPO_ROOT / "tests"
for _p in (str(REPO_ROOT), str(_TESTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from live_sensor_fixtures import LiveRos2SensorPublishers, gazebo_binary_available, rclpy_available

from robodeploy.backends.real.ros2.sim_launchers.ros_gz_bridge import imu_bridge_rules
from robodeploy.backends.sim.gazebo.urdf_sensors import inject_sensors_into_urdf
from robodeploy.core.types import SensorMount
from robodeploy.sensors.imu.sim.mujoco_imu import MuJoCoIMUSensor

LIVE = os.environ.get("ROBODEPLOY_LIVE_GAZEBO", "").strip() in {"1", "true", "yes"}
EMPTY_WORLD = REPO_ROOT / "tests" / "fixtures" / "gazebo_empty.sdf"
CAMERA_FT_WORLD = REPO_ROOT / "tests" / "fixtures" / "gazebo_camera_ft.sdf"
MINIMAL_URDF = REPO_ROOT / "tests" / "fixtures" / "gazebo_minimal_arm.urdf"


def _gazebo_sim_cfg(
    *,
    world: Path,
    robot_urdf: Path | None = None,
    wait_for_topics: list[str] | None = None,
) -> dict:
    sim: dict = {
        "kind": "gazebo",
        "world": str(world),
        "headless": True,
        "readiness_timeout_s": 60.0,
        "wait_for_topics": wait_for_topics or [],
    }
    if robot_urdf is not None:
        sim["robot_urdf"] = str(robot_urdf)
        sim["robot_name"] = "robot0"
    return sim


class GazeboSensorOfflineTests(unittest.TestCase):
    def test_multimodal_preset_yaml_includes_imu_and_depth(self):
        from examples.config import load_example_preset

        cfg = load_example_preset("kuka_ft_imu_pick_gazebo")
        rig = cfg["sensor_rigs"][0]
        self.assertTrue(rig["wrist_rgbd"]["depth"])
        self.assertIn("wrist_imu", rig)
        self.assertIn("prop_pose", rig)

    def test_imu_urdf_and_bridge_rules_offline(self):
        imu = MuJoCoIMUSensor("wrist_imu", mount=SensorMount(parent_link="ee_link"))
        patched = inject_sensors_into_urdf(
            '<?xml version="1.0"?><robot name="arm"><link name="ee_link"/></robot>',
            [imu],
        )
        self.assertIn('type="imu"', patched)
        rules = imu_bridge_rules("/wrist_imu/imu")
        self.assertTrue(any("sensor_msgs/msg/Imu" in r for r in rules))

    def test_kuka_ft_imu_pick_gazebo_preset_loads(self):
        from examples.config import load_example_preset

        cfg = load_example_preset("kuka_ft_imu_pick_gazebo")
        self.assertEqual(cfg["backend"], "ros2_gazebo")
        self.assertTrue(cfg["backend_kwargs"]["config"]["sim"].get("require_sensors"))
        self.assertEqual(cfg["policy_kwargs"]["config"]["force_threshold"], 0.5)

    def test_kuka_ft_imu_pick_gazebo_env_builds_offline(self):
        from examples.config import load_example_preset
        from robodeploy.backends.real.ros2.sensors.camera_rgbd import Ros2RgbdCameraISensor
        from robodeploy.backends.real.ros2.sensors.wrench import Ros2WrenchISensor
        from robodeploy.env import RoboEnv

        cfg = load_example_preset("kuka_ft_imu_pick_gazebo")
        cfg = {**cfg, "max_episode_steps": 5}
        env = RoboEnv.from_config(cfg)
        try:
            sensors = env.robots[0].sensors
            kinds = {type(s) for s in sensors}
            self.assertIn(Ros2RgbdCameraISensor, kinds)
            self.assertIn(Ros2WrenchISensor, kinds)
            self.assertGreaterEqual(len(env.robots[0].tasks), 1)
            task = next(iter(env.robots[0].tasks.values()))
            self.assertEqual(task.task.__class__.__name__, "PickPlaceTask")
        finally:
            env.close()


@unittest.skipUnless(LIVE, "set ROBODEPLOY_LIVE_GAZEBO=1 to run live Gazebo sensor tests")
@unittest.skipUnless(gazebo_binary_available(), "gz binary not on PATH")
@unittest.skipUnless(rclpy_available(), "rclpy not available")
class LiveGazeboSensorTests(unittest.TestCase):
    def test_ros2_gazebo_sensor_rig_populates_observation(self):
        """Fast fallback: synthetic ROS publishers feed bridged topic names."""
        from examples.config import load_example_preset
        from robodeploy.env import RoboEnv

        pubs = LiveRos2SensorPublishers(robot_id="robot0")
        time.sleep(0.3)

        cfg = load_example_preset("kuka_sensor_gazebo")
        cfg = {
            **cfg,
            "max_episode_steps": 20,
            "backend_kwargs": {"config": {"sim": _gazebo_sim_cfg(world=EMPTY_WORLD)}},
        }
        env = RoboEnv.from_config(cfg)
        try:
            obs, _ = env.reset()
            deadline = time.monotonic() + 20.0
            while time.monotonic() < deadline:
                if obs.images.get("wrist_camera") is not None and obs.ft_forces.get("wrist_ft") is not None:
                    break
                obs, _, _, _ = env.step()
            self.assertIsNotNone(obs.images.get("wrist_camera"))
            self.assertIsNotNone(obs.ft_forces.get("wrist_ft"))
        finally:
            env.close()
            pubs.close()

    def test_gz_rendered_camera_populates_observation(self):
        """End-to-end: gz sim renders camera frames bridged to ROS (no synthetic image pub)."""
        from examples.config import load_example_preset
        from robodeploy.env import RoboEnv

        cfg = load_example_preset("kuka_sensor_gazebo")
        cfg = {
            **cfg,
            "max_episode_steps": 20,
            "backend_kwargs": {
                "config": {
                    "sim": _gazebo_sim_cfg(
                        world=CAMERA_FT_WORLD,
                        robot_urdf=MINIMAL_URDF,
                        wait_for_topics=[
                            "/wrist_camera/image_raw",
                            "/wrist_camera/camera_info",
                        ],
                    )
                }
            },
        }
        env = RoboEnv.from_config(cfg)
        try:
            obs, _ = env.reset()
            deadline = time.monotonic() + 45.0
            while time.monotonic() < deadline:
                img = obs.images.get("wrist_camera")
                intrinsics = obs.camera_intrinsics.get("wrist_camera")
                if img is not None and intrinsics and float(np.sum(img)) > 0.0:
                    break
                obs, _, _, _ = env.step()
            img = obs.images.get("wrist_camera")
            self.assertIsNotNone(img, msg=f"sensor_status={getattr(obs, 'sensor_status', {})}")
            assert img is not None
            self.assertGreater(float(np.sum(img)), 0.0)
            self.assertIn("wrist_camera", obs.camera_intrinsics)
        finally:
            env.close()

    def test_kuka_ft_imu_pick_gazebo_obs_keys(self):
        """Live: multimodal preset populates wrist sensors after reset (no synthetic pubs)."""
        from examples.config import load_example_preset
        from robodeploy.env import RoboEnv

        cfg = load_example_preset("kuka_ft_imu_pick_gazebo")
        cfg = {
            **cfg,
            "max_episode_steps": 30,
            "backend_kwargs": {
                "config": {
                    "sim": _gazebo_sim_cfg(
                        world=EMPTY_WORLD,
                        wait_for_topics=["/joint_states", "/wrist_ft/wrench"],
                    )
                }
            },
        }
        env = RoboEnv.from_config(cfg)
        try:
            obs, _ = env.reset()
            deadline = time.monotonic() + 60.0
            while time.monotonic() < deadline:
                has_ft = obs.ft_forces.get("wrist_ft") is not None
                has_objects = bool(getattr(obs, "objects", {}))
                if has_ft and has_objects:
                    break
                obs, _, _, _ = env.step()
            self.assertIn("source", getattr(obs, "objects", {}))
            self.assertIsNotNone(obs.ft_forces.get("wrist_ft"))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
