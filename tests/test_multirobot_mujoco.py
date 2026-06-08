from __future__ import annotations

import unittest

import numpy as np


def _require_mujoco():
    try:
        import mujoco  # noqa: F401
    except ImportError:
        raise unittest.SkipTest("mujoco not installed")


class MultiRobotMuJoCoTests(unittest.TestCase):
    def _two_franka_env(self):
        _require_mujoco()
        from robodeploy.backends.sim.mujoco.backend import MuJoCoBackend
        from robodeploy.core.robot import Robot, RobotTask
        from robodeploy.core.types import Action, Pose3D
        from robodeploy.env import RoboEnv
        from examples.franka_pick_place_mujoco.components import ExampleFrankaMujocoDescription
        from examples.policies.joint_track import JointTrackPolicy
        from examples.tasks.pick_place import PickPlaceTask

        home = [float(x) for x in ExampleFrankaMujocoDescription().home_qpos]
        left_target = (np.array(home) + np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])).tolist()
        right_target = (np.array(home) + np.array([-0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])).tolist()
        task = PickPlaceTask()
        robots = [
            Robot(
                robot_id="franka_left",
                description=ExampleFrankaMujocoDescription(),
                base_pose=Pose3D(position=(-0.5, 0.0, 0.4)),
                tasks={
                    "pick": RobotTask(
                        task=task,
                        policies={"track": JointTrackPolicy(home_qpos=home, target_qpos=left_target)},
                    )
                },
            ),
            Robot(
                robot_id="franka_right",
                description=ExampleFrankaMujocoDescription(),
                base_pose=Pose3D(position=(0.5, 0.0, 0.4)),
                tasks={
                    "pick": RobotTask(
                        task=task,
                        policies={"track": JointTrackPolicy(home_qpos=home, target_qpos=right_target)},
                    )
                },
            ),
        ]
        env = RoboEnv(
            backend=MuJoCoBackend(config={"allow_actuator_name_fallback": True, "enable_viewer": False}),
            robots=robots,
        )
        return env

    def test_initialize_multi_returns_per_robot_obs(self):
        env = self._two_franka_env()
        try:
            env.reset()
            obs_map = env.get_processed_obs_by_robot()
            self.assertEqual(set(obs_map), {"franka_left", "franka_right"})
            for obs in obs_map.values():
                self.assertEqual(int(obs.joint_positions.shape[0]), 7)
        finally:
            env.close()

    def test_explicit_actions_hit_correct_dof_slices(self):
        env = self._two_franka_env()
        try:
            env.reset()
            home = np.array([float(x) for x in env.robots[0].description.home_qpos], dtype=np.float32)
            left_cmd = home + np.array([0.1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
            right_cmd = home + np.array([-0.1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
            from robodeploy.core.types import Action

            try:
                import jax.numpy as jnp
            except Exception:
                import numpy as jnp  # type: ignore[assignment]

            env.step(
                {
                    "franka_left": Action(joint_positions=jnp.asarray(left_cmd)),
                    "franka_right": Action(joint_positions=jnp.asarray(right_cmd)),
                }
            )
            obs_map = env.get_processed_obs_by_robot()
            left_q0 = float(obs_map["franka_left"].joint_positions[0])
            right_q0 = float(obs_map["franka_right"].joint_positions[0])
            self.assertGreater(left_q0, right_q0, "left arm should command higher q0 than right arm")
        finally:
            env.close()

    def test_shared_overhead_camera_visible_to_all_robots(self):
        _require_mujoco()
        from robodeploy.backends.sim.mujoco.backend import MuJoCoBackend
        from robodeploy.core.robot import Robot, RobotTask
        from robodeploy.core.types import Pose3D
        from robodeploy.env import RoboEnv
        from robodeploy.sensors.camera.sim.mujoco_camera import MuJoCoOverheadCameraRenderer
        from examples.franka_pick_place_mujoco.components import ExampleFrankaMujocoDescription
        from examples.policies.joint_track import JointTrackPolicy
        from examples.tasks.pick_place import PickPlaceTask

        home = [float(x) for x in ExampleFrankaMujocoDescription().home_qpos]
        task = PickPlaceTask()
        shared_cam = MuJoCoOverheadCameraRenderer(config={"width": 64, "height": 48, "depth": False})
        robots = [
            Robot(
                robot_id="franka_a",
                description=ExampleFrankaMujocoDescription(),
                base_pose=Pose3D(position=(-0.5, 0.0, 0.4)),
                tasks={
                    "pick": RobotTask(
                        task=task,
                        policies={"track": JointTrackPolicy(home_qpos=home, target_qpos=home)},
                    )
                },
            ),
            Robot(
                robot_id="franka_b",
                description=ExampleFrankaMujocoDescription(),
                base_pose=Pose3D(position=(0.5, 0.0, 0.4)),
                tasks={
                    "pick": RobotTask(
                        task=task,
                        policies={"track": JointTrackPolicy(home_qpos=home, target_qpos=home)},
                    )
                },
            ),
        ]
        env = RoboEnv(
            backend=MuJoCoBackend(config={"allow_actuator_name_fallback": True, "enable_viewer": False}),
            robots=robots,
            shared_sensors=[shared_cam],
        )
        try:
            env.reset()
            env.step()
            obs_map = env.get_processed_obs_by_robot()
            for rid in ("franka_a", "franka_b"):
                obs = obs_map[rid]
                images = getattr(obs, "images", {}) or {}
                self.assertIn(
                    "overhead_camera",
                    images,
                    f"{rid} should see shared overhead_camera",
                )
        finally:
            env.close()

    def test_average_joint_resolver_midpoint(self):
        from robodeploy.core.types import Action
        from robodeploy.multirobot.resolvers import average_joint_actions

        try:
            import jax.numpy as jnp
        except Exception:
            import numpy as jnp  # type: ignore[assignment]

        a = Action(joint_positions=jnp.asarray([0.0, 0.0], dtype=jnp.float32))
        b = Action(joint_positions=jnp.asarray([1.0, 1.0], dtype=jnp.float32))
        merged = average_joint_actions("r0", [a, b])
        self.assertAlmostEqual(float(merged.joint_positions[0]), 0.5, places=4)
        self.assertAlmostEqual(float(merged.joint_positions[1]), 0.5, places=4)


if __name__ == "__main__":
    unittest.main()
