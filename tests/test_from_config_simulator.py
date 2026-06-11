"""RoboEnv.from_config routes known backends through backend_for_simulator."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _mujoco_available() -> bool:
    try:
        import mujoco  # noqa: F401

        return True
    except ImportError:
        return False


class FromConfigSimulatorPathTests(unittest.TestCase):
    def test_simulator_name_for_backend_aliases(self):
        from robodeploy.backends.simulator import simulator_name_for_backend

        self.assertEqual(simulator_name_for_backend("mujoco"), "mujoco")
        self.assertEqual(simulator_name_for_backend("gazebo"), "gazebo")
        self.assertEqual(simulator_name_for_backend("ros2_gazebo"), "gazebo")
        self.assertEqual(simulator_name_for_backend("ros2_rviz"), "ros2_rviz")
        self.assertEqual(simulator_name_for_backend("ros2"), "real_world")
        self.assertIsNone(simulator_name_for_backend("dummy"))

    def test_from_config_mujoco_matches_backend_for_simulator(self):
        from examples.config import load_example_preset
        from robodeploy.backends.simulator import backend_for_simulator
        from robodeploy.builtins import import_builtins
        from robodeploy.env import RoboEnv

        import_builtins()
        cfg = load_example_preset("kuka_pick_mujoco")
        cfg = {
            **cfg,
            "backend_kwargs": {
                "config": {"allow_actuator_name_fallback": True, "enable_viewer": False}
            },
            "max_episode_steps": 5,
        }
        env = RoboEnv.from_config(cfg)
        try:
            direct = backend_for_simulator(
                "mujoco",
                robots=env.robots,
                config_overrides={"allow_actuator_name_fallback": True, "enable_viewer": False},
            )
            self.assertEqual(env.backend.__class__, direct.__class__)
            self.assertEqual(
                env.backend.config.get("allow_actuator_name_fallback"),
                direct.config.get("allow_actuator_name_fallback"),
            )
            self.assertEqual(env.backend.config.get("enable_viewer"), False)
        finally:
            env.close()

    def test_from_config_ros2_rviz_auto_wires_dev_fake_sim(self):
        from examples.config import load_example_preset
        from robodeploy.builtins import import_builtins
        from robodeploy.env import RoboEnv

        import_builtins()
        preset = load_example_preset("kuka_pick_ros2_rviz")
        env = RoboEnv.from_config({**preset, "max_episode_steps": 5})
        try:
            fake = env.backend.config.get("dev_fake_sim")
            self.assertIsInstance(fake, list)
            self.assertGreaterEqual(len(fake), 1)
            self.assertEqual(fake[0].get("robot_ns"), "/robot0")
            self.assertIn("joint_names", fake[0])
            if preset.get("backend_kwargs", {}).get("config", {}).get("dev_fake_sim"):
                self.assertAlmostEqual(float(fake[0].get("follow_tau_s", 0.15)), 0.05, places=3)
            rviz = env.backend.config.get("rviz")
            self.assertIsInstance(rviz, dict)
            self.assertTrue(rviz.get("enabled"))
        finally:
            env.close()

    def test_from_config_gazebo_without_world_falls_back_to_raw_kwargs(self):
        from robodeploy.backends.sim.gazebo.backend import ROS2GazeboBackend
        from robodeploy.builtins import import_builtins
        from robodeploy.env import RoboEnv

        import_builtins()
        cfg = {
            "robot": "kuka",
            "backend": "ros2_gazebo",
            "task": "pick_place",
            "policy": "example_joint_track",
            "backend_kwargs": {"config": {"sim": {"kind": "gazebo", "headless": True}}},
            "custom_modules": ["examples.tasks", "examples.policies"],
        }
        env = RoboEnv.from_config(cfg)
        try:
            self.assertIsInstance(env.backend, ROS2GazeboBackend)
            self.assertEqual(env.backend.config.get("sim", {}).get("kind"), "gazebo")
            self.assertNotIn("dev_fake_sim", env.backend.config)
        finally:
            env.close()

    def test_from_config_max_episode_steps_truncates_not_failure(self):
        if not _mujoco_available():
            self.skipTest("mujoco not installed")
        from examples.config import load_example_preset
        from robodeploy.builtins import import_builtins
        from robodeploy.env import RoboEnv

        import_builtins()
        cfg = {**load_example_preset("kuka_ft_imu_pick_mujoco"), "max_episode_steps": 5}
        env = RoboEnv.from_config(cfg)
        try:
            env.reset(seed=0)
            info = None
            for _ in range(5):
                _, _, done, info = env.step()
                if done:
                    break
            self.assertIsNotNone(info)
            assert info is not None
            self.assertFalse(bool(info.success))
            self.assertFalse(bool(info.failure))
            self.assertTrue(bool(info.extra.get("truncated")))
            self.assertTrue(bool(info.extra.get("timeout")))
        finally:
            env.close()

    @pytest.mark.ci_pick_gate
    def test_ci_pick_gate_3_seeds_from_config(self):
        """PR pick regression gate (Track B3): 3 seeds via from_config + backend_for_simulator."""
        if not _mujoco_available():
            self.skipTest("mujoco not installed")
        from examples.config import load_example_preset
        from robodeploy.builtins import import_builtins
        from robodeploy.env import RoboEnv

        import_builtins()
        seeds = (0, 1, 2)
        for seed in seeds:
            cfg = {**load_example_preset("kuka_ft_imu_pick_mujoco"), "max_episode_steps": 1500}
            env = RoboEnv.from_config(cfg)
            try:
                env.reset(seed=seed)
                info = None
                for _ in range(1500):
                    _, _, done, info = env.step()
                    if done:
                        break
                self.assertIsNotNone(info, msg=f"seed {seed} produced no info")
                assert info is not None
                self.assertTrue(
                    bool(info.success),
                    msg=f"seed {seed} failed (step={info.step}, failure={info.failure})",
                )
            finally:
                env.close()


if __name__ == "__main__":
    unittest.main()
