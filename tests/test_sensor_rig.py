from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class SensorRigTests(unittest.TestCase):
    def test_robot_mounted_materializes_prop_pose_sensor(self):
        from robodeploy.core.sensor_rig import SensorRig

        use("examples.sensors")
        rig = SensorRig.robot_mounted("rig0", prop_pose={"prop_names": ["source"]})
        sensors = rig.materialize(is_real=False, backend_name="mujoco")
        self.assertEqual(len(sensors), 1)
        self.assertEqual(sensors[0].name, "prop_pose")

    def test_prop_pose_sensor_merges_into_observation(self):
        from robodeploy.core.robot import Robot, RobotTask
        from robodeploy.core.spaces import ActionSpace
        from robodeploy.core.types import Action, Observation
        from robodeploy.description.base import RobotDescription
        from robodeploy.env import RoboEnv
        from robodeploy.policies.base import PolicyBase
        from robodeploy.tasks.base import TaskBase
        from robodeploy.core.types import ObsSpec, SceneSpec
        from examples.sensors.prop_pose import SimPropPoseSensor

        try:
            import jax.numpy as jnp
        except Exception:
            import numpy as jnp  # type: ignore[assignment]

        class _FakeBackend:
            is_real = False
            control_hz = 20.0
            supported_action_spaces = [ActionSpace.JOINT_POS]

            def __init__(self) -> None:
                self._pending_sensor_reads: list = []
                self._sensor_errors: dict = {}
                self._sensor_error_warned: set = set()
                self.config: dict = {}

            def get_prop_pose(self, name: str):
                return ((0.5, 0.0, 0.4), (1.0, 0.0, 0.0, 0.0))

            def initialize_multi(self, robots, scene, shared_sensors) -> None:
                self._sensors = list(robots[0].sensors)

            def reset_multi(self, robot_ids=None):
                obs = self._make_obs()
                return [self._merge(obs)]

            def step_multi(self, actions):
                return [self._merge(self._make_obs())]

            def get_obs_multi(self):
                return [self._merge(self._make_obs())]

            def _make_obs(self):
                return Observation(
                    joint_positions=jnp.zeros((2,), dtype=jnp.float32),
                    joint_velocities=jnp.zeros((2,), dtype=jnp.float32),
                    joint_torques=jnp.zeros((2,), dtype=jnp.float32),
                    ee_position=jnp.zeros((3,), dtype=jnp.float32),
                    ee_orientation=jnp.asarray([1, 0, 0, 0], dtype=jnp.float32),
                    ee_velocity=jnp.zeros((3,), dtype=jnp.float32),
                    ee_angular_velocity=jnp.zeros((3,), dtype=jnp.float32),
                )

            def _merge(self, obs):
                from robodeploy.backends.base import BackendBase

                return BackendBase._merge_sensor_data(self, obs, self._sensors)

            def close(self):
                return

        class _Desc(RobotDescription):
            dof = 2
            display_name = "d"
            ee_link_name = "ee"
            joint_names = ["j1", "j2"]
            joint_position_limits = jnp.asarray([[-1, 1], [-1, 1]], dtype=jnp.float32)
            joint_velocity_limits = jnp.asarray([1, 1], dtype=jnp.float32)
            joint_torque_limits = jnp.asarray([1, 1], dtype=jnp.float32)
            home_qpos = jnp.zeros((2,), dtype=jnp.float32)

            def asset_path(self, fmt, variant="default"):
                return ""

        class _Task(TaskBase):
            def obs_spec(self):
                return ObsSpec()

            def scene_spec(self):
                return SceneSpec()

            def language_instruction(self):
                return ""

            def reset_fn(self, backend):
                pass

            def reward_fn(self, obs, action):
                return 0.0

            def success_fn(self, obs):
                return False

        class _Pol(PolicyBase):
            def __init__(self):
                super().__init__(action_space=ActionSpace.JOINT_POS)

            def _reset_impl(self):
                pass

            def get_action(self, obs):
                self.last_objects = dict(getattr(obs, "objects", {}))
                return Action(joint_positions=jnp.zeros((2,), dtype=jnp.float32))

        sensor = SimPropPoseSensor(config={"name": "prop_pose", "prop_names": ["source"]})
        policy = _Pol()
        robot = Robot(
            robot_id="r0",
            description=_Desc(),
            sensors=[sensor],
            tasks={"t": RobotTask(task=_Task(), policies={"p": policy})},
        )
        env = RoboEnv(backend=_FakeBackend(), robots=[robot])
        try:
            env.reset()
            env.step()
            self.assertIn("source", policy.last_objects)
        finally:
            env.close()

    def test_gazebo_sensor_rig_applies_ros_topic_defaults(self):
        from robodeploy.builtins import import_builtins
        from robodeploy.core.sensor_rig import SensorRig

        import_builtins()
        rig = SensorRig.robot_mounted(wrist_rgbd={"width": 64}, wrist_ft={})
        sensors = rig.materialize(is_real=True, backend_name="gazebo")
        self.assertEqual(len(sensors), 2)
        cam_cfg = sensors[0].config
        self.assertEqual(cam_cfg.get("rgb"), "image_raw")
        self.assertEqual(cam_cfg.get("namespace"), "/wrist_camera")
        ft_cfg = sensors[1].config
        self.assertEqual(ft_cfg.get("wrench_topic"), "wrench")

    def test_from_config_sensor_rigs_materialize(self):
        from robodeploy.env import RoboEnv

        use("examples.sensors")
        cfg = {
            "robot": "kuka",
            "backend": "mujoco",
            "task": "pick_place",
            "policy": "example_sensor_reach_pick",
            "custom_modules": ["examples.tasks", "examples.sensors", "examples.policies"],
            "sensor_rigs": [
                {"prop_pose": {"prop_names": ["source"]}, "ee_link": "robot0/ee_link"},
            ],
            "backend_kwargs": {"config": {"allow_actuator_name_fallback": True, "enable_viewer": False}},
        }
        try:
            env = RoboEnv.from_config(cfg)
        except ImportError:
            self.skipTest("mujoco not installed")
        try:
            self.assertEqual(len(env.robots[0].sensors), 1)
            self.assertEqual(env.robots[0].sensors[0].name, "prop_pose")
        finally:
            env.close()

    def test_sensor_pick_mujoco_success_when_mujoco_installed(self):
        try:
            import mujoco  # noqa: F401
        except ImportError:
            self.skipTest("mujoco not installed")
        from examples.kuka_sensor_pick_mujoco.run_mujoco import _attach_policy_ik, build_env

        env = build_env(max_steps=1500)
        try:
            env.reset()
            _attach_policy_ik(env)
            info = None
            for _ in range(1500):
                _, _, done, info = env.step()
                if done:
                    break
            self.assertIsNotNone(info)
            assert info is not None
            self.assertTrue(bool(info.success))
        finally:
            env.close()


def use(module: str) -> None:
    from robodeploy.core.registry import use as registry_use

    registry_use(module)


if __name__ == "__main__":
    unittest.main()
