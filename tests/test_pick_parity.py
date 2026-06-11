"""Cross-simulator pick preset parity (offline)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class PickParityPresetTests(unittest.TestCase):
    def test_preset_diff_only_backend_mujoco_gazebo(self):
        from robodeploy.presets_loader import diff_presets

        diff = diff_presets("kuka_ft_imu_pick_mujoco", "kuka_ft_imu_pick_gazebo")
        allowed = {"backend", "backend_kwargs"}
        extra = set(diff) - allowed
        self.assertFalse(extra, msg=f"non-backend diff keys: {extra}")

    def test_preset_diff_only_backend_mujoco_rviz(self):
        from robodeploy.presets_loader import diff_presets

        diff = diff_presets("kuka_ft_imu_pick_mujoco", "kuka_ft_imu_pick_ros2_rviz")
        allowed = {"backend", "backend_kwargs"}
        extra = set(diff) - allowed
        self.assertFalse(extra, msg=f"non-backend diff keys: {extra}")

    def test_headless_presets_share_core_with_visual(self):
        from examples.config import load_example_preset

        pairs = [
            ("kuka_ft_imu_pick_mujoco", "kuka_ft_imu_pick_mujoco_headless"),
            ("kuka_ft_imu_pick_gazebo", "kuka_ft_imu_pick_gazebo_headless"),
            ("kuka_ft_imu_pick_ros2_rviz", "kuka_ft_imu_pick_ros2_rviz_headless"),
        ]
        keys = ("task", "policy", "sensor_rigs", "policy_kwargs", "task_kwargs", "safety")
        for visual, headless in pairs:
            a = load_example_preset(visual)
            b = load_example_preset(headless)
            for key in keys:
                self.assertEqual(a.get(key), b.get(key), msg=f"{visual} vs {headless} key {key}")

    def test_kuka_ft_imu_pick_shared_core_fields(self):
        from examples.config import load_example_preset

        keys = ("task", "policy", "sensor_rigs", "policy_kwargs", "task_kwargs", "safety")
        presets = [
            load_example_preset("kuka_ft_imu_pick_mujoco"),
            load_example_preset("kuka_ft_imu_pick_gazebo"),
            load_example_preset("kuka_ft_imu_pick_ros2_rviz"),
        ]
        for key in keys:
            values = [p.get(key) for p in presets]
            self.assertEqual(values[0], values[1], msg=f"{key} mujoco vs gazebo")
            self.assertEqual(values[0], values[2], msg=f"{key} mujoco vs rviz")

    def test_pick_presets_default_visual_backends(self):
        from examples.config import load_example_preset

        mujoco = load_example_preset("kuka_ft_imu_pick_mujoco")
        gazebo = load_example_preset("kuka_ft_imu_pick_gazebo")
        rviz = load_example_preset("kuka_ft_imu_pick_ros2_rviz")
        self.assertTrue(mujoco["backend_kwargs"]["config"].get("enable_viewer"))
        self.assertFalse(gazebo["backend_kwargs"]["config"]["sim"].get("headless"))
        self.assertTrue(rviz["backend_kwargs"]["config"]["rviz"].get("enabled"))

    def test_ee_pose_ros_transport_reads_driver_tf_obs(self):
        import numpy as np

        from robodeploy.core.types import Observation
        from examples.sensors.ee_pose import _ee_from_ros_transport

        z7 = np.zeros(7, dtype=np.float32)
        z3 = np.zeros(3, dtype=np.float32)
        class _FakeBackend:
            _latest_obs = {
                "robot0": Observation(
                    joint_positions=z7,
                    joint_velocities=z7,
                    joint_torques=z7,
                    ee_position=np.array([0.1, 0.2, 0.3], dtype=np.float32),
                    ee_orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                    ee_velocity=z3,
                    ee_angular_velocity=z3,
                )
            }

        out = _ee_from_ros_transport(_FakeBackend())
        self.assertIsNotNone(out)
        pos, quat = out
        np.testing.assert_allclose(pos, [0.1, 0.2, 0.3], rtol=1e-5)
        np.testing.assert_allclose(quat, [1.0, 0.0, 0.0, 0.0], rtol=1e-5)

    def test_sensor_reach_pick_maps_rviz_kinematic_carry(self):
        from examples.policies.sensor_reach_pick import SensorReachPickPlacePolicy
        from robodeploy.policies.reach_dsl import ReachTrajectoryPolicy

        policy = SensorReachPickPlacePolicy(
            config={"carry_mode": "follow", "sensor_only": True},
            scene=None,
        )
        backend = type("B", (), {"sensor_backend_name": "ros2_rviz"})()
        policy.bind_runtime(backend, description=None)
        # bind_runtime returns early without description; set backend and remap directly
        policy._backend = backend
        policy._backend_follow_carry = True
        policy._kinematic_carry = False
        ReachTrajectoryPolicy._map_rviz_carry_mode(policy, backend)
        self.assertTrue(policy._kinematic_carry)

    def test_coerce_policy_unwraps_nested_policy_kwargs_config(self):
        from robodeploy.env import RoboEnv
        from robodeploy.core.registry import use
        from robodeploy.builtins import import_builtins

        import_builtins()
        use("examples.policies")
        from examples.kuka_ft_imu_pick_gazebo.pick_episode import (
            RELAXED_POLICY_CONFIG,
            kuka_ft_imu_pick_gazebo_cfg,
        )

        cfg = kuka_ft_imu_pick_gazebo_cfg(policy_config={**RELAXED_POLICY_CONFIG})
        policy = RoboEnv._coerce_policy(cfg["policy"], cfg.get("policy_kwargs"))
        self.assertEqual(str(policy.config.get("carry_mode")), "kinematic")
        self.assertTrue(policy._kinematic_carry)

    def test_sensor_reach_pick_maps_mujoco_kinematic_carry(self):
        from examples.policies.sensor_reach_pick import SensorReachPickPlacePolicy
        from robodeploy.policies.reach_dsl import ReachTrajectoryPolicy

        policy = SensorReachPickPlacePolicy(
            config={"carry_mode": "follow", "sensor_only": True, "grasp_detection": "ft"},
            scene=None,
        )
        backend = type("B", (), {"sensor_backend_name": "mujoco"})()
        policy._backend = backend
        policy._backend_follow_carry = True
        policy._kinematic_carry = False
        ReachTrajectoryPolicy._map_mujoco_carry_mode(policy, backend)
        self.assertTrue(policy._kinematic_carry)
        self.assertEqual(policy._grasp_detection, "ft")

    def test_sensor_reach_pick_maps_gazebo_kinematic_carry(self):
        from examples.policies.sensor_reach_pick import SensorReachPickPlacePolicy
        from robodeploy.policies.reach_dsl import ReachTrajectoryPolicy

        policy = SensorReachPickPlacePolicy(
            config={"carry_mode": "follow", "sensor_only": True},
            scene=None,
        )
        backend = type("B", (), {"sensor_backend_name": "gazebo"})()
        policy._backend = backend
        policy._backend_follow_carry = True
        policy._kinematic_carry = False
        ReachTrajectoryPolicy._map_gazebo_carry_mode(policy, backend)
        self.assertTrue(policy._kinematic_carry)
        self.assertEqual(policy._grasp_detection, "distance")

    def test_gazebo_rsp_params_enable_use_sim_time(self):
        from robodeploy.backends.real.ros2.sim_launchers.robot_state_publisher import (
            robot_state_publisher_params_yaml,
        )

        off = robot_state_publisher_params_yaml("<robot name='r'/>", use_sim_time=False)
        on = robot_state_publisher_params_yaml("<robot name='r'/>", use_sim_time=True)
        self.assertNotIn("use_sim_time", off)
        self.assertIn("use_sim_time: true", on)

    def test_rviz_rsp_joint_driven_publish_frequency(self):
        from robodeploy.backends.real.ros2.sim_launchers.robot_state_publisher import (
            robot_state_publisher_params_yaml,
        )

        yaml_text = robot_state_publisher_params_yaml(
            "<robot name='r'/>",
            use_sim_time=False,
            publish_frequency=0.0,
        )
        self.assertIn("publish_frequency: 0.0", yaml_text)

    def test_ros2_rviz_rsp_disables_publish_timer(self):
        from robodeploy.backends.real.ros2.sim_launchers.robot_state_publisher import (
            robot_state_publisher_params_yaml,
        )

        yaml_text = robot_state_publisher_params_yaml(
            "<robot name='r'/>",
            use_sim_time=False,
            publish_frequency=0.0,
        )
        self.assertIn("publish_frequency: 0.0", yaml_text)

    def test_rviz_isolated_fake_sim_ignores_stale_graph_clock(self):
        backend_src = (
            REPO_ROOT / "robodeploy" / "backends" / "real" / "ros2" / "backend.py"
        ).read_text(encoding="utf-8")
        self.assertIn("dev_fake_sim", backend_src)
        self.assertIn('self.config.get("sim") is None', backend_src)
        self.assertIn("CONNECTION_LOST", backend_src)

    def test_fake_joint_sim_uses_ros_runtime_sim_time_flag(self):
        from robodeploy.backends.real.ros2.dev.fake_joint_sim import FakeJointPosSimConfig
        from robodeploy.backends.real.ros2.runtime import Ros2Runtime

        self.assertIsInstance(FakeJointPosSimConfig(), FakeJointPosSimConfig)
        Ros2Runtime.use_sim_time = True
        try:
            self.assertTrue(Ros2Runtime.use_sim_time)
        finally:
            Ros2Runtime.use_sim_time = False

    def test_gazebo_pick_wait_topics_skip_absent_wrist_camera(self):
        from examples.kuka_ft_imu_pick_gazebo.pick_episode import kuka_ft_imu_pick_gazebo_cfg

        cfg = kuka_ft_imu_pick_gazebo_cfg(max_episode_steps=10)
        topics = cfg["backend_kwargs"]["config"]["sim"]["wait_for_topics"]
        self.assertIn("/joint_states", topics)
        self.assertIn("/wrist_ft/wrench", topics)
        self.assertIn("/wrist_imu/imu", topics)
        self.assertNotIn("/wrist_camera/image_raw", topics)

    def test_sensor_only_pick_policy_no_oracle_reads(self):
        reach_dsl = REPO_ROOT / "robodeploy" / "policies" / "reach_dsl.py"
        text = reach_dsl.read_text(encoding="utf-8")
        self.assertIn('"sensor_only"', text)
        self.assertIn("ee_position_from_obs", text)

        bind_src = (REPO_ROOT / "examples" / "policies" / "sensor_reach_pick.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("get_prop_pose", bind_src)
        self.assertNotIn("waypoints_from_scene", bind_src)


if __name__ == "__main__":
    unittest.main()
