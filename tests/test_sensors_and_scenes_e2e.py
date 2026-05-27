from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import jax.numpy as jnp
except Exception:
    import numpy as jnp  # type: ignore[assignment]

from robodeploy.backends.sim.mujoco.backend import MuJoCoBackend
from robodeploy.core.robot import Robot, RobotTask
from robodeploy.core.spaces import ActionSpace, AssetFormat
from robodeploy.core.types import (
    Action,
    GeomSpec,
    ObsSpec,
    Observation,
    PropConfig,
    SceneSpec,
    SensorMount,
    WorldSpec,
)
from robodeploy.description.base import RobotDescription
from robodeploy.env import RoboEnv
from robodeploy.policies.base import PolicyBase
from robodeploy.sensors.camera.sim.mujoco_camera import MuJoCoCameraRenderer
from robodeploy.sensors.ft_sensor.sim.mujoco_ft import MuJoCoFTSensor
from robodeploy.tasks.base import TaskBase
from robodeploy.tasks.randomization import DomainRandomizer, DomainRandomizerConfig, ObjectRandomConfig, RandomLevel


class TinyDescription(RobotDescription):
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


class HoldPolicy(PolicyBase):
    def __init__(self) -> None:
        super().__init__(action_space=ActionSpace.JOINT_POS)

    def _reset_impl(self) -> None:
        return

    def get_action(self, obs: Observation) -> Action:
        del obs
        return Action(joint_positions=jnp.asarray([0.0], dtype=jnp.float32))


class SceneTask(TaskBase):
    def obs_spec(self) -> ObsSpec:
        return ObsSpec()

    def scene_spec(self) -> SceneSpec:
        return SceneSpec(
            world=WorldSpec(
                props=[
                    PropConfig(
                        name="cube",
                        position=(0.2, 0.0, 0.05),
                        geom=GeomSpec(kind="box", size=(0.02, 0.02, 0.02)),
                    )
                ]
            )
        )

    def language_instruction(self) -> str:
        return "hold"

    def reset_fn(self, backend) -> None:
        del backend

    def reward_fn(self, obs: Observation, action: Action) -> float:
        del obs, action
        return 0.0

    def success_fn(self, obs: Observation) -> bool:
        del obs
        return False


MJCF = """<mujoco model="tiny">
  <worldbody>
    <body name="arm" pos="0 0 0.05">
      <joint name="joint1" type="hinge" axis="0 0 1"/>
      <geom type="capsule" size="0.01 0.08" fromto="0 0 0 0.12 0 0" rgba="0.4 0.4 0.8 1"/>
      <body name="ee_link" pos="0.12 0 0">
        <geom type="sphere" size="0.02" rgba="0.8 0.4 0.4 1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="joint1" joint="joint1" kp="10"/>
  </actuator>
</mujoco>
"""


class MuJoCoSensorSceneE2ETests(unittest.TestCase):
    def test_sensor_and_scene_capabilities_on_mujoco(self) -> None:
        try:
            import mujoco  # noqa: F401
        except Exception as exc:
            self.skipTest(f"mujoco unavailable: {exc}")

        with tempfile.TemporaryDirectory() as td:
            mjcf_path = Path(td) / "tiny.xml"
            mjcf_path.write_text(MJCF, encoding="utf-8")
            robot = Robot(
                robot_id="tiny0",
                description=TinyDescription(mjcf_path),
                tasks={"hold": RobotTask(task=SceneTask(), policies={"hold": HoldPolicy()})},
                sensors=[
                    MuJoCoCameraRenderer(
                        "wrist",
                        config={"width": 32, "height": 24, "depth": True},
                        mount=SensorMount(parent_link="ee_link", position=(0.0, -0.2, 0.1)),
                    ),
                    MuJoCoFTSensor("wrist_ft", mount=SensorMount(parent_link="ee_link")),
                ],
            )
            env = RoboEnv(backend=MuJoCoBackend(config={"enable_viewer": False}), robots=[robot])
            try:
                obs, _ = env.reset()
                self.assertIn("wrist", obs.images)
                self.assertEqual(tuple(obs.images["wrist"].shape), (24, 32, 3))
                self.assertIn("wrist", obs.depths)
                self.assertIsNotNone(obs.ft_force)
                self.assertGreaterEqual(obs.timestamp_hw, 0.0)

                backend = env.backend
                self.assertIn("cube", backend.get_prop_names())
                pose0 = backend.get_prop_pose("cube")
                backend.set_prop_pose("cube", (0.25, 0.01, 0.05), pose0[1])
                pose1 = backend.get_prop_pose("cube")
                self.assertFalse(np.allclose(pose0[0], pose1[0]))

                randomizer = DomainRandomizer(
                    DomainRandomizerConfig(
                        level=RandomLevel.LIGHT,
                        seed=7,
                        objects=[
                            ObjectRandomConfig(
                                object_name="cube",
                                position_center=(0.2, 0.0, 0.05),
                                position_range=(0.02, 0.02, 0.0),
                            )
                        ],
                    )
                )
                randomizer.randomize(backend)
                pose2 = backend.get_prop_pose("cube")
                self.assertFalse(np.allclose(pose1[0], pose2[0]))
            finally:
                env.close()


if __name__ == "__main__":
    unittest.main()
