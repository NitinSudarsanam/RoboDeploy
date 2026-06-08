from __future__ import annotations

try:
    import jax.numpy as jnp
except Exception:
    import numpy as jnp  # type: ignore[assignment]

from robodeploy.backends.base import BackendBase
from robodeploy.core.spaces import ActionSpace
from robodeploy.core.types import Action, ObsSpec, Observation, SceneSpec
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
        self.last_actions: dict[str, Action] = {}

    def _load(self, description, scene, sensors) -> None:
        del description, scene, sensors

    def _reset_impl(self) -> Observation:
        self._latest["robot0"] = make_obs(0.0)
        if "robot1" in self._latest:
            self._latest["robot1"] = make_obs(1.0)
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
        for rid in ids:
            self._latest[rid] = make_obs(0.0 if rid == "robot0" else 1.0)
        return [self._latest[rid] for rid in ids]

    def step_multi(self, actions: list[Action]) -> list[Observation]:
        for rid, action in zip(self._robot_ids, actions):
            self.last_actions[rid] = action
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

    def get_sim_state(self) -> dict:
        latest_values: dict[str, float] = {}
        for rid, obs in self._latest.items():
            jp = obs.joint_positions
            latest_values[rid] = float(jp[0]) if jp is not None else 0.0
        return {"latest_values": latest_values}

    def set_sim_state(self, state: dict) -> None:
        for rid, val in dict(state.get("latest_values") or {}).items():
            if rid in self._latest:
                self._latest[rid] = make_obs(float(val))


class DummyRealBackend(DummyBackend):
    is_real = True


class DummyPolicy(PolicyBase):
    def __init__(self, value: float):
        super().__init__(action_space=ActionSpace.JOINT_POS)
        self._value = value

    def _reset_impl(self, *, seed: int | None = None) -> None:
        del seed
        return

    def get_action(self, obs: Observation) -> Action:
        del obs
        return Action(joint_positions=jnp.asarray([self._value, self._value], dtype=jnp.float32))


class BatchPolicy(PolicyBase):
    def __init__(self):
        super().__init__(action_space=ActionSpace.JOINT_POS)
        self.single_calls = 0
        self.batch_calls = 0

    def get_action(self, obs: Observation) -> Action:
        self.single_calls += 1
        value = float(obs.joint_positions[0]) + 1.0
        return Action(joint_positions=jnp.asarray([value, value], dtype=jnp.float32))

    def get_action_batch(self, obs_batch: list[Observation]) -> list[Action]:
        self.batch_calls += 1
        return [self.get_action(obs) for obs in obs_batch]


class RejectAwarePolicy(PolicyBase):
    def __init__(self, value: float, *, action_hz: float = 0.0):
        super().__init__(action_space=ActionSpace.JOINT_POS, config={"action_hz": action_hz})
        self._value = value
        self.rejected = 0

    def get_action(self, obs: Observation) -> Action:
        del obs
        return Action(joint_positions=jnp.asarray([self._value, self._value], dtype=jnp.float32))

    def notify_rejected(self, obs: Observation, action: Action) -> None:
        del obs, action
        self.rejected += 1


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

    def reward_components_fn(self, obs: Observation, action: Action) -> dict[str, float]:
        from robodeploy.tasks.reward_builder import RewardBuilder

        return RewardBuilder().penalty_action_norm(scale=0.01).build_components()(obs, action)

    def success_fn(self, obs: Observation) -> bool:
        return False

    def failure_fn(self, obs: Observation) -> bool:
        return False

    def viz_goals(self, obs=None):
        del obs
        return [{"kind": "point", "position": [0.0, 0.0, 0.0]}]

