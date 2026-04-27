from __future__ import annotations

import unittest

try:
    import jax.numpy as jnp
except Exception:
    import numpy as jnp  # type: ignore[assignment]

from robodeploy import RoboEnv
from robodeploy.backends.base import BackendBase
from robodeploy.core.robot import Robot, RobotTask
from robodeploy.core.spaces import ActionSpace
from robodeploy.core.types import Action, MultiAgentInfo, ObsSpec, Observation, SceneSpec
from robodeploy.description.base import RobotDescription
from robodeploy.policies.base import PolicyBase
from robodeploy.tasks.base import TaskBase


class DummyRobot(RobotDescription):
    dof = 2
    display_name = "dummy"
    ee_link_name = "ee_link"
    joint_names = ["joint1", "joint2"]
    joint_position_limits = jnp.asarray([[-3.14, 3.14], [-3.14, 3.14]], dtype=jnp.float32)
    joint_velocity_limits = jnp.asarray([2.0, 2.0], dtype=jnp.float32)
    joint_torque_limits = jnp.asarray([10.0, 10.0], dtype=jnp.float32)
    home_qpos = jnp.asarray([0.0, 0.0], dtype=jnp.float32)

    def asset_path(self, fmt, variant: str = "default"):
        del fmt, variant
        return ""


def make_obs(value: float) -> Observation:
    return Observation(
        joint_positions=jnp.asarray([value, value], dtype=jnp.float32),
        joint_velocities=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        joint_torques=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        ee_position=jnp.asarray([value, 0.0, 0.0], dtype=jnp.float32),
        ee_orientation=jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
        ee_velocity=jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float32),
        ee_angular_velocity=jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float32),
        timestamp=value,
        timestamp_hw=value,
        timestamp_recv=value,
    )


class DummyBackend(BackendBase):
    is_real = False
    control_hz = 20.0
    supported_action_spaces = [ActionSpace.JOINT_POS]

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self._latest = {"robot0": make_obs(0.0), "robot1": make_obs(1.0)}
        self._latest_viz_payload = None

    def _load(self, description, scene, sensors) -> None:
        del description, scene, sensors

    def _reset_impl(self) -> Observation:
        return self._latest["robot0"]

    def _step_impl(self, action: Action) -> Observation:
        del action
        return self._latest["robot0"]

    def _get_obs_impl(self) -> Observation:
        return self._latest["robot0"]

    def _close_impl(self) -> None:
        return

    def initialize_multi(self, robots, scene, shared_sensors) -> None:
        del scene, shared_sensors
        self._robot_ids = [r.robot_id for r in robots]
        self._initialized = True

    def reset_multi(self, robot_ids=None) -> list[Observation]:
        ids = robot_ids or self._robot_ids
        return [self._latest[rid] for rid in ids]

    def step_multi(self, actions: list[Action]) -> list[Observation]:
        for rid, action in zip(self._robot_ids, actions):
            if action.joint_positions is not None:
                val = float(action.joint_positions[0])
                self._latest[rid] = make_obs(val)
        return [self._latest[rid] for rid in self._robot_ids]

    def get_obs_multi(self) -> list[Observation]:
        return [self._latest[rid] for rid in self._robot_ids]

    def set_viz_payload(self, payload):
        self._latest_viz_payload = payload

    def get_diagnostics(self) -> dict:
        return {"backend": "dummy", "ok": True}


class DummyPolicy(PolicyBase):
    def __init__(self, value: float):
        super().__init__(action_space=ActionSpace.JOINT_POS)
        self._value = value

    def _reset_impl(self) -> None:
        return

    def get_action(self, obs: Observation) -> Action:
        del obs
        return Action(joint_positions=jnp.asarray([self._value, self._value], dtype=jnp.float32))


class DummyTask(TaskBase):
    def obs_spec(self) -> ObsSpec:
        return ObsSpec()

    def scene_spec(self) -> SceneSpec:
        return SceneSpec()

    def language_instruction(self) -> str:
        return "hold"

    def reset_fn(self, backend) -> None:
        del backend

    def reward_fn(self, obs: Observation, action: Action) -> float:
        del action
        return float(obs.joint_positions[0])

    def success_fn(self, obs: Observation) -> bool:
        return False

    def failure_fn(self, obs: Observation) -> bool:
        return False

    def viz_goals(self, obs=None):
        del obs
        return [{"kind": "point", "position": [0.0, 0.0, 0.0]}]


class EnvRefactorTests(unittest.TestCase):
    def test_multi_agent_routing_and_extra_payloads(self):
        backend = DummyBackend()
        r0 = Robot(
            robot_id="robot0",
            description=DummyRobot(),
            tasks={
                "task0": RobotTask(
                    task=DummyTask(),
                    policies={"p": DummyPolicy(2.0)},
                    mode="sequential",
                )
            },
        )
        r1 = Robot(
            robot_id="robot1",
            description=DummyRobot(),
            tasks={
                "task1": RobotTask(
                    task=DummyTask(),
                    policies={"p": DummyPolicy(3.0)},
                    mode="sequential",
                )
            },
        )
        env = RoboEnv(backend=backend, robots=[r0, r1])

        obs, info = env.reset()
        self.assertIn("multi_agent", info.extra)
        self.assertIn("viz", info.extra)
        self.assertIn("diagnostics", info.extra)
        self.assertIsInstance(info.extra["multi_agent"], MultiAgentInfo)

        obs, reward, done, info = env.step()
        del obs, done
        self.assertEqual(reward, 2.0)
        self.assertIn("task0", info.extra["viz"]["tasks"])


class BackendConfigMergeTests(unittest.TestCase):
    def test_nested_config_key_merges_and_nested_wins(self):
        backend = DummyBackend(
            {"enable_viewer": False, "config": {"enable_viewer": True, "x": 1}}
        )
        self.assertEqual(backend.config["enable_viewer"], True)
        self.assertEqual(backend.config["x"], 1)
        self.assertNotIn("config", backend.config)
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
