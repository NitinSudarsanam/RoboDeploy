"""Live Gazebo + ROS2 sensor integration (sensor-live-gazebo CI job)."""

from __future__ import annotations

import os
import sys
import time
import unittest
from pathlib import Path

import numpy as np
import pytest

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
    headless: bool = True,
    readiness_timeout_s: float = 60.0,
) -> dict:
    sim: dict = {
        "kind": "gazebo",
        "world": str(world),
        "headless": headless,
        "readiness_timeout_s": readiness_timeout_s,
        "wait_for_topics": wait_for_topics or [],
    }
    if robot_urdf is not None:
        sim["robot_urdf"] = str(robot_urdf)
        sim["robot_name"] = "robot0"
    return sim


def _kuka_ft_imu_pick_gazebo_cfg(
    *,
    max_episode_steps: int = 30,
    policy_config: dict | None = None,
    task_kwargs: dict | None = None,
    wait_for_topics: list[str] | None = None,
) -> dict:
    from examples.config import load_example_preset

    cfg = load_example_preset("kuka_ft_imu_pick_gazebo")
    policy_kwargs = dict(cfg.get("policy_kwargs", {}))
    merged_policy = {**dict(policy_kwargs.get("config", {})), **(policy_config or {})}
    policy_kwargs["config"] = merged_policy
    merged_task = {**dict(cfg.get("task_kwargs", {})), **(task_kwargs or {})}
    topics = wait_for_topics or [
        "/joint_states",
        "/wrist_ft/wrench",
        "/wrist_imu/imu",
        "/wrist_camera/image_raw",
    ]
    return {
        **cfg,
        "max_episode_steps": max_episode_steps,
        "policy_kwargs": policy_kwargs,
        "task_kwargs": merged_task,
        "backend_kwargs": {
            "config": {
                "sim": _gazebo_sim_cfg(
                    world=EMPTY_WORLD,
                    wait_for_topics=topics,
                    readiness_timeout_s=90.0,
                )
            }
        },
    }


def _multimodal_obs_ready(obs) -> bool:
    objects = getattr(obs, "objects", {}) or {}
    contact = getattr(obs, "contact_state", {}) or {}
    has_ft = obs.ft_forces.get("wrist_ft") is not None
    has_objects = "source" in objects and "target" in objects
    has_contact_key = "wrist_contact" in contact
    has_imu = getattr(obs, "imu_angular_velocity", None) is not None
    return has_ft and has_objects and has_contact_key and has_imu


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

    @pytest.mark.live_gazebo
    def test_kuka_ft_imu_pick_gazebo_obs_keys(self):
        """Live: multimodal preset populates FT, IMU, contact, prop_pose, and camera obs."""
        from robodeploy.env import RoboEnv

        cfg = _kuka_ft_imu_pick_gazebo_cfg(max_episode_steps=40)
        env = RoboEnv.from_config(cfg)
        try:
            obs, info = env.reset()
            deadline = time.monotonic() + 90.0
            while time.monotonic() < deadline:
                if _multimodal_obs_ready(obs):
                    break
                obs, _, _, info = env.step()
            status = getattr(obs, "sensor_status", {}) or {}
            self.assertIn("source", getattr(obs, "objects", {}))
            self.assertIn("target", getattr(obs, "objects", {}))
            self.assertIsNotNone(obs.ft_forces.get("wrist_ft"))
            self.assertIsNotNone(getattr(obs, "imu_angular_velocity", None))
            self.assertIn("wrist_contact", getattr(obs, "contact_state", {}) or {})
            self.assertIn("wrist_camera", obs.images)
            self.assertTrue(
                "wrist_camera" in (getattr(obs, "depths", {}) or {})
                or getattr(obs, "depth", None) is not None,
                msg="expected wrist_camera depth from bridged RGB-D rig",
            )
            self.assertIn("overall", info.extra.get("sensor_health", {}))
            self.assertIn("wrist_ft", status)
            self.assertIn("wrist_imu", status)
        finally:
            env.close()

    @pytest.mark.live_gazebo
    def test_kuka_ft_imu_pick_gazebo_episode_success_relaxed(self):
        """Live: pick-place with relaxed FT tuning; at least one of three seeds succeeds.

        Skipped unless ROBODEPLOY_LIVE_GAZEBO=1 (sensor-live-gazebo CI on Linux).
        Uses kinematic carry (not weld); success still depends on JTC + IK tuning.
        """
        from robodeploy.env import RoboEnv

        seeds = (0, 1, 2)
        successes = 0
        for seed in seeds:
            cfg = _kuka_ft_imu_pick_gazebo_cfg(
                max_episode_steps=1200,
                policy_config={
                    "force_threshold": 0.3,
                    "grasp_force_window": 2,
                    "imu_omega_max": 0.8,
                    "imu_settle_steps": 2,
                },
                task_kwargs={"grasp_success_force_min": 0.5},
            )
            env = RoboEnv.from_config(cfg)
            try:
                env.reset(seed=seed)
                info = None
                for _ in range(1200):
                    _, _, done, info = env.step()
                    if done:
                        break
                if info is not None and bool(info.success):
                    successes += 1
            finally:
                env.close()

        self.assertGreaterEqual(
            successes,
            1,
            msg=(
                f"relaxed Gazebo pick-place: {successes}/{len(seeds)} seeds succeeded; "
                "check JTC, IK (.[kinematics]), and FT thresholds"
            ),
        )


if __name__ == "__main__":
    unittest.main()
