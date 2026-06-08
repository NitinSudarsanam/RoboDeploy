"""Gymnasium-compatible adapter for RoboEnv."""

from __future__ import annotations

from typing import Any

import numpy as np

from robodeploy.core.interop import to_numpy
from robodeploy.core.spaces import ActionSpace
from robodeploy.core.types import Action, Observation, ObsSpec
from robodeploy.env import RoboEnv

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "GymRoboEnv requires gymnasium. Install with: pip install 'robodeploy[training]'"
    ) from exc


def observation_to_dict(obs: Observation, obs_spec: ObsSpec) -> dict[str, np.ndarray]:
    """Flatten Observation into numpy arrays for gym Dict spaces."""
    out: dict[str, np.ndarray] = {}
    out["proprio"] = np.concatenate(
        [
            to_numpy(obs.joint_positions),
            to_numpy(obs.joint_velocities),
            to_numpy(obs.joint_torques),
        ],
        dtype=np.float32,
    )
    out["ee"] = np.concatenate(
        [to_numpy(obs.ee_position), to_numpy(obs.ee_orientation)],
        dtype=np.float32,
    )
    if obs_spec.rgb:
        rgb = obs.rgb
        if rgb is None and obs.images:
            rgb = next(iter(obs.images.values()))
        if rgb is not None:
            out["rgb"] = np.asarray(rgb, dtype=np.uint8)
    if obs_spec.depth:
        depth = obs.depth
        if depth is None and obs.depths:
            depth = next(iter(obs.depths.values()))
        if depth is not None:
            out["depth"] = np.asarray(depth, dtype=np.float32)
    if obs_spec.ft_sensor and obs.ft_force is not None and obs.ft_torque is not None:
        out["ft"] = np.concatenate(
            [to_numpy(obs.ft_force), to_numpy(obs.ft_torque)],
            dtype=np.float32,
        )
    return out


def action_array_to_action(
    arr: np.ndarray,
    *,
    action_space: ActionSpace,
    dof: int,
) -> Action:
    """Convert a flat action vector into RoboDeploy Action."""
    vec = np.asarray(arr, dtype=np.float32).reshape(-1)
    if action_space == ActionSpace.JOINT_POS:
        return Action(joint_positions=vec[:dof], action_space=action_space)
    if action_space == ActionSpace.JOINT_VEL:
        return Action(joint_velocities=vec[:dof], action_space=action_space)
    if action_space == ActionSpace.JOINT_TORQUE:
        return Action(joint_torques=vec[:dof], action_space=action_space)
    if action_space == ActionSpace.CARTESIAN_POSE:
        return Action(
            ee_position=vec[:3],
            ee_orientation=vec[3:7],
            action_space=action_space,
        )
    if action_space == ActionSpace.DELTA_EE:
        return Action(
            ee_position=vec[:3],
            ee_orientation=vec[3:7],
            action_space=action_space,
            is_delta_ee=True,
        )
    return Action(joint_positions=vec[:dof], action_space=ActionSpace.JOINT_POS)


class GymRoboEnv(gym.Env):
    """Adapts RoboEnv to gymnasium API: 5-tuple step, Box/Dict spaces."""

    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        robo_env: RoboEnv,
        *,
        max_episode_steps: int | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self._env = robo_env
        self._render_mode = render_mode
        self._primary_robot = robo_env.primary_robot
        self._description = self._primary_robot.description
        self._active_task = self._resolve_active_task()
        self._obs_spec = self._active_task.task.obs_spec()
        self._action_space_enum = self._active_task.action_space()
        self._dof = int(self._description.dof)
        self._action_space = self._build_action_space()
        self._observation_space = self._build_observation_space()
        self._max_steps = int(
            max_episode_steps
            or getattr(robo_env, "max_episode_steps", None)
            or robo_env._max_steps
        )
        self._step_count = 0

    def _resolve_active_task(self):
        robot = self._primary_robot
        task_id = robot.active_task_id
        if task_id and task_id in robot.tasks:
            return robot.tasks[task_id]
        return next(iter(robot.tasks.values()))

    @property
    def action_space(self) -> spaces.Space:
        return self._action_space

    @property
    def observation_space(self) -> spaces.Space:
        return self._observation_space

    @property
    def robo_env(self) -> RoboEnv:
        """Underlying RoboDeploy runtime (do not override ``unwrapped`` — breaks gymnasium)."""
        return self._env

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        obs, info = self._env.reset()
        self._step_count = 0
        return observation_to_dict(obs, self._obs_spec), self._info_dict(info)

    def step(self, action_array: np.ndarray):
        action = action_array_to_action(
            action_array,
            action_space=self._action_space_enum,
            dof=self._dof,
        )
        obs, reward, done, info = self._env.step(action)
        self._step_count += 1
        truncated = self._step_count >= self._max_steps
        terminated = bool(done) and not truncated
        return (
            observation_to_dict(obs, self._obs_spec),
            float(reward),
            terminated,
            truncated,
            self._info_dict(info),
        )

    def render(self):
        if self._render_mode == "rgb_array":
            obs, _ = self._env.reset()
            rgb = obs.rgb
            if rgb is None and obs.images:
                rgb = next(iter(obs.images.values()))
            if rgb is not None:
                return np.asarray(rgb, dtype=np.uint8)
            return np.zeros((64, 64, 3), dtype=np.uint8)
        self._env.render()
        return None

    def close(self) -> None:
        self._env.close()

    def _build_action_space(self) -> spaces.Box:
        limits = np.asarray(self._description.joint_position_limits, dtype=np.float32)
        if self._action_space_enum == ActionSpace.JOINT_POS:
            low, high = limits[:, 0], limits[:, 1]
        elif self._action_space_enum == ActionSpace.JOINT_VEL:
            vel = np.asarray(self._description.joint_velocity_limits, dtype=np.float32)
            low, high = -vel, vel
        elif self._action_space_enum == ActionSpace.JOINT_TORQUE:
            torque = np.asarray(self._description.joint_torque_limits, dtype=np.float32)
            low, high = -torque, torque
        elif self._action_space_enum in (ActionSpace.CARTESIAN_POSE, ActionSpace.DELTA_EE):
            low = np.array([-2.0, -2.0, 0.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32)
            high = np.array([2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
            return spaces.Box(low=low, high=high, dtype=np.float32)
        else:
            low, high = limits[:, 0], limits[:, 1]
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def _build_observation_space(self) -> spaces.Dict:
        dof = self._dof
        space_map: dict[str, spaces.Space] = {
            "proprio": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(dof * 3,),
                dtype=np.float32,
            ),
            "ee": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(7,),
                dtype=np.float32,
            ),
        }
        if self._obs_spec.rgb:
            h = int(self._obs_spec.image_height)
            w = int(self._obs_spec.image_width)
            space_map["rgb"] = spaces.Box(low=0, high=255, shape=(h, w, 3), dtype=np.uint8)
        if self._obs_spec.depth:
            h = int(self._obs_spec.image_height)
            w = int(self._obs_spec.image_width)
            space_map["depth"] = spaces.Box(
                low=0.0,
                high=10.0,
                shape=(h, w),
                dtype=np.float32,
            )
        if self._obs_spec.ft_sensor:
            space_map["ft"] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(6,),
                dtype=np.float32,
            )
        return spaces.Dict(space_map)

    def _info_dict(self, info: Any) -> dict[str, Any]:
        extra = dict(getattr(info, "extra", {}) or {})
        out = {
            "episode_id": getattr(info, "episode_id", 0),
            "step": getattr(info, "step", 0),
            "reward": float(getattr(info, "reward", 0.0)),
            "success": bool(getattr(info, "success", False)),
            "failure": bool(getattr(info, "failure", False)),
            "extra": extra,
        }
        if "reward_components" in extra:
            out["reward_components"] = dict(extra["reward_components"])
        return out
