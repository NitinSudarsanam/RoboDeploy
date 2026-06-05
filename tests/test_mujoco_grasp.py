from __future__ import annotations

import unittest

import numpy as np


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


    def test_weld_equality_activates_when_enabled(self):
        try:
            import mujoco  # noqa: F401
        except ImportError:
            self.skipTest("mujoco not installed")

        from robodeploy.backends.sim.mujoco.backend import MuJoCoBackend
        from robodeploy.core.robot import Robot, RobotTask
        from robodeploy.description.kuka import KukaDescription
        from robodeploy.env import RoboEnv
        from examples.tasks.pick_place import PickPlaceTask
        from robodeploy.policies.base import PolicyBase
        from robodeploy.core.spaces import ActionSpace
        from robodeploy.core.types import Action

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
            backend=MuJoCoBackend(
                config={
                    "allow_actuator_name_fallback": True,
                    "enable_viewer": False,
                    "enable_grasp_welds": True,
                }
            ),
            robots=[robot],
        )
        try:
            env.reset()
            backend = env.backend
            self.assertIn("source", backend._grasp_eq_ids)
            backend.set_grasp_prop("source", offset=(0.0, 0.0, 0.03), mode="weld")
            eq_id = backend._grasp_eq_ids["source"]
            self.assertEqual(int(backend._data.eq_active[eq_id]), 1)
            backend.set_grasp_prop(None)
            self.assertEqual(int(backend._data.eq_active[eq_id]), 0)
        finally:
            env.close()

    def test_prop_near_ee_contact_proxy(self):
        try:
            import mujoco  # noqa: F401
        except ImportError:
            self.skipTest("mujoco not installed")

        from robodeploy.backends.sim.mujoco.backend import MuJoCoBackend
        from robodeploy.core.robot import Robot, RobotTask
        from robodeploy.description.kuka import KukaDescription
        from robodeploy.env import RoboEnv
        from examples.tasks.pick_place import PickPlaceTask
        from robodeploy.policies.base import PolicyBase
        from robodeploy.core.spaces import ActionSpace
        from robodeploy.core.types import Action

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
            backend.set_prop_pose("source", backend.get_prop_pose("source")[0], (1.0, 0.0, 0.0, 0.0))
            self.assertFalse(backend.prop_near_ee("source", threshold=0.01))
            ee_pos = backend._data.xpos[backend._ee_body_id]
            backend.set_prop_pose("source", (float(ee_pos[0]), float(ee_pos[1]), float(ee_pos[2])), (1.0, 0.0, 0.0, 0.0))
            self.assertTrue(backend.prop_near_ee("source", threshold=0.02))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
