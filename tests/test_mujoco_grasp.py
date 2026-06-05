from __future__ import annotations

import unittest


class MuJoCoGraspTests(unittest.TestCase):
    def test_set_grasp_prop_tracks_source_on_step(self):
        try:
            import mujoco  # noqa: F401
        except ImportError:
            self.skipTest("mujoco not installed")

        from robodeploy.backends.sim.mujoco.backend import MuJoCoBackend
        from robodeploy.core.robot import Robot, RobotTask
        from robodeploy.core.types import Action
        from robodeploy.description.kuka import KukaDescription
        from robodeploy.env import RoboEnv
        from examples.tasks.pick_place import PickPlaceTask
        from robodeploy.policies.base import PolicyBase
        from robodeploy.core.spaces import ActionSpace

        try:
            import jax.numpy as jnp
        except Exception:
            import numpy as jnp  # type: ignore[assignment]

        class _Hold(PolicyBase):
            def __init__(self):
                super().__init__(action_space=ActionSpace.JOINT_POS)

            def _reset_impl(self):
                return

            def get_action(self, obs):
                del obs
                return Action(joint_positions=jnp.zeros((7,), dtype=jnp.float32))

        desc = KukaDescription()
        robot = Robot(
            robot_id="robot0",
            description=desc,
            tasks={"pick": RobotTask(task=PickPlaceTask(), policies={"h": _Hold()})},
        )
        env = RoboEnv(
            backend=MuJoCoBackend(config={"allow_actuator_name_fallback": True, "enable_viewer": False}),
            robots=[robot],
        )
        try:
            env.reset()
            backend = env.backend
            backend.set_grasp_prop("source", offset=(0.0, 0.0, 0.03))
            before = backend.get_prop_pose("source")[0]
            env.step()
            after = backend.get_prop_pose("source")[0]
            self.assertNotEqual(before, after)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
