from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import jax.numpy as jnp
except Exception:
    import numpy as jnp  # type: ignore[assignment]

from robodeploy.backends.sim.mujoco.backend import MuJoCoBackend
from robodeploy.core.robot import Robot, RobotTask
from robodeploy.core.spaces import ActionSpace, AssetFormat
from robodeploy.core.types import Action, Observation
from robodeploy.description.base import RobotDescription
from robodeploy.env import RoboEnv
from robodeploy.policies.base import PolicyBase
from robodeploy.tasks.base import TaskBase
from robodeploy.core.types import ObsSpec, SceneSpec

MJCF = """<mujoco model="tiny">
  <worldbody>
    <body name="arm" pos="0 0 0.05">
      <joint name="joint1" type="hinge" axis="0 0 1"/>
      <geom type="capsule" size="0.01 0.08" fromto="0 0 0 0.12 0 0"/>
      <body name="ee_link" pos="0.12 0 0"><geom type="sphere" size="0.02"/></body>
    </body>
  </worldbody>
  <actuator><position name="joint1" joint="joint1" kp="10"/></actuator>
</mujoco>
"""


class _TinyDesc(RobotDescription):
    dof = 1
    display_name = "tiny"
    ee_link_name = "ee_link"
    joint_names = ["joint1"]
    joint_position_limits = jnp.asarray([[-3.14, 3.14]], dtype=jnp.float32)
    joint_velocity_limits = jnp.asarray([2.0], dtype=jnp.float32)
    joint_torque_limits = jnp.asarray([10.0], dtype=jnp.float32)
    home_qpos = jnp.asarray([0.0], dtype=jnp.float32)

    def __init__(self, mjcf_path: Path) -> None:
        self._mjcf_path = mjcf_path

    def asset_path(self, fmt, variant: str = "default"):
        del variant
        if fmt == AssetFormat.MJCF:
            return self._mjcf_path
        raise FileNotFoundError(fmt)


class _TinyTask(TaskBase):
    def obs_spec(self) -> ObsSpec:
        return ObsSpec()

    def scene_spec(self) -> SceneSpec:
        return SceneSpec()

    def language_instruction(self) -> str:
        return ""

    def reset_fn(self, backend) -> None:
        del backend

    def reward_fn(self, obs: Observation, action: Action) -> float:
        del obs, action
        return 0.0

    def success_fn(self, obs: Observation) -> bool:
        del obs
        return False


class _Hold(PolicyBase):
    def __init__(self) -> None:
        super().__init__(action_space=ActionSpace.JOINT_POS)

    def get_action(self, obs: Observation) -> Action:
        del obs
        return Action(joint_positions=jnp.asarray([0.0], dtype=jnp.float32))


class MuJoCoSmokeTests(unittest.TestCase):
    def test_mujoco_backend_import_and_instantiate(self):
        try:
            import mujoco  # noqa: F401
        except ImportError:
            self.skipTest("mujoco not installed")
        backend = MuJoCoBackend()
        self.assertFalse(backend.is_real)

    def test_mujoco_reset_and_step_smoke(self):
        try:
            import mujoco  # noqa: F401
        except ImportError:
            self.skipTest("mujoco not installed")
        with tempfile.TemporaryDirectory() as td:
            mjcf = Path(td) / "tiny.xml"
            mjcf.write_text(MJCF, encoding="utf-8")
            robot = Robot(
                robot_id="tiny0",
                description=_TinyDesc(mjcf),
                tasks={"t": RobotTask(task=_TinyTask(), policies={"p": _Hold()})},
            )
            env = RoboEnv(backend=MuJoCoBackend(config={"enable_viewer": False}), robots=[robot])
            try:
                obs, _ = env.reset()
                self.assertEqual(len(obs.joint_positions), 1)
                obs2, reward, done, _ = env.step(
                    Action(joint_positions=jnp.asarray([0.1], dtype=jnp.float32))
                )
                self.assertEqual(len(obs2.joint_positions), 1)
                self.assertIsInstance(reward, float)
                self.assertIsInstance(done, bool)
            finally:
                env.close()


if __name__ == "__main__":
    unittest.main()
