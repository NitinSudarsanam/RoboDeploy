"""Smoke tests for SensorRig → Gazebo URDF patch → ros_gz_bridge wiring."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MINIMAL_URDF = """<?xml version="1.0"?>
<robot name="arm">
  <link name="ee_link"/>
</robot>
"""


class GazeboSensorRigSmokeTests(unittest.TestCase):
    def test_from_config_ros2_gazebo_sensor_rigs_materialize_ros2_sensors(self):
        from robodeploy.backends.real.ros2.sensors.camera_rgbd import Ros2RgbdCameraISensor
        from robodeploy.backends.real.ros2.sensors.wrench import Ros2WrenchISensor
        from robodeploy.env import RoboEnv

        cfg = {
            "robot": "kuka",
            "backend": "ros2_gazebo",
            "task": "pick_place",
            "policy": "example_joint_track",
            "backend_kwargs": {"config": {"sim": {"kind": "gazebo"}}},
            "sensor_rigs": [
                {
                    "rig_id": "arm_sensors",
                    "ee_link": "robot0/ee_link",
                    "wrist_rgbd": {"width": 64, "height": 48, "depth": False},
                    "wrist_ft": {},
                }
            ],
            "custom_modules": ["examples.tasks", "examples.policies"],
        }
        env = RoboEnv.from_config(cfg)
        try:
            sensors = env.robots[0].sensors
            self.assertEqual(len(sensors), 2)
            self.assertIsInstance(sensors[0], Ros2RgbdCameraISensor)
            self.assertIsInstance(sensors[1], Ros2WrenchISensor)
            self.assertEqual(sensors[0].config.get("rgb"), "image_raw")
            self.assertEqual(sensors[0].config.get("namespace"), "/wrist_camera")
            self.assertEqual(sensors[1].config.get("wrench_topic"), "wrench")
        finally:
            env.close()

    def test_gazebo_initialize_multi_patches_urdf_and_bridges_sensor_rig(self):
        from robodeploy.backends.real.ros2.sensors.camera_rgbd import Ros2RgbdCameraISensor
        from robodeploy.backends.real.ros2.sensors.wrench import Ros2WrenchISensor
        from robodeploy.backends.sim.gazebo.backend import ROS2GazeboBackend
        from robodeploy.core.robot import Robot, RobotTask
        from robodeploy.core.sensor_rig import SensorRig
        from robodeploy.core.spaces import ActionSpace, AssetFormat
        from robodeploy.core.types import Action, ObsSpec, Observation, SceneSpec
        from robodeploy.description.base import RobotDescription
        from robodeploy.policies.base import PolicyBase
        from robodeploy.tasks.base import TaskBase

        try:
            import jax.numpy as jnp
        except Exception:
            import numpy as jnp  # type: ignore[assignment]

        class _Desc(RobotDescription):
            dof = 2
            display_name = "d"
            ee_link_name = "ee_link"
            joint_names = ["j1", "j2"]
            joint_position_limits = jnp.asarray([[-1, 1], [-1, 1]], dtype=jnp.float32)
            joint_velocity_limits = jnp.asarray([1, 1], dtype=jnp.float32)
            joint_torque_limits = jnp.asarray([1, 1], dtype=jnp.float32)
            home_qpos = jnp.zeros((2,), dtype=jnp.float32)

            def asset_path(self, fmt, variant: str = "default"):
                del variant
                if fmt == AssetFormat.URDF:
                    return self._urdf_path
                return ""

        class _Task(TaskBase):
            def obs_spec(self):
                return ObsSpec()

            def scene_spec(self):
                return SceneSpec()

            def language_instruction(self):
                return ""

            def reset_fn(self, backend):
                del backend

            def reward_fn(self, obs, action):
                del obs, action
                return 0.0

            def success_fn(self, obs):
                del obs
                return False

        class _Pol(PolicyBase):
            def __init__(self) -> None:
                super().__init__(action_space=ActionSpace.JOINT_POS)

            def _reset_impl(self) -> None:
                return

            def get_action(self, obs: Observation) -> Action:
                del obs
                return Action(joint_positions=jnp.asarray([0.0, 0.0], dtype=jnp.float32))

        with tempfile.TemporaryDirectory() as td:
            urdf_path = Path(td) / "arm.urdf"
            urdf_path.write_text(MINIMAL_URDF, encoding="utf-8")

            desc = _Desc()
            desc._urdf_path = urdf_path

            rig = SensorRig.robot_mounted(
                "arm_sensors",
                ee_link="ee_link",
                wrist_rgbd={"width": 64, "height": 48, "depth": False},
                wrist_ft={},
            )
            sensors = rig.materialize(is_real=False, backend_name="gazebo")
            self.assertEqual(len(sensors), 2)
            self.assertIsInstance(sensors[0], Ros2RgbdCameraISensor)
            self.assertIsInstance(sensors[1], Ros2WrenchISensor)

            robot = Robot(
                robot_id="robot0",
                description=desc,
                tasks={"task0": RobotTask(task=_Task(), policies={"p": _Pol()})},
                sensors=sensors,
            )
            backend = ROS2GazeboBackend(config={"sim": {"kind": "gazebo"}})

            with (
                mock.patch("robodeploy.backends.real.ros2.sim_launchers.gazebo.GazeboLauncher") as launcher_cls,
                mock.patch("robodeploy.backends.real.ros2.backend.ROS2RealBackend.initialize_multi", return_value=None),
            ):
                backend.initialize_multi([robot], SceneSpec(), [])

            launch_cfg = launcher_cls.call_args.args[0]
            patched_urdf = Path(str(launch_cfg.robot_urdf))
            self.addCleanup(lambda: patched_urdf.unlink(missing_ok=True))
            self.assertTrue(patched_urdf.exists())
            self.assertNotEqual(patched_urdf, urdf_path)
            xml = patched_urdf.read_text(encoding="utf-8")
            self.assertIn('joint name="wrist_camera_joint"', xml)
            self.assertIn('sensor name="wrist_camera" type="camera"', xml)
            self.assertIn('sensor name="wrist_ft" type="force_torque"', xml)
            self.assertTrue(
                any("/wrist_camera/image_raw" in rule for rule in launch_cfg.bridge_rules),
                launch_cfg.bridge_rules,
            )
            self.assertTrue(
                any("/wrist_ft/wrench" in rule for rule in launch_cfg.bridge_rules),
                launch_cfg.bridge_rules,
            )
            self.assertIn("/wrist_camera/image_raw", launch_cfg.wait_for_topics)
            self.assertIn("/wrist_ft/wrench", launch_cfg.wait_for_topics)


if __name__ == "__main__":
    unittest.main()
