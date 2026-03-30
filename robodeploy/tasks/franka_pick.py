"""Basic scripted pick task for the Franka demo robot."""

from __future__ import annotations

import jax.numpy as jnp

from robodeploy.core.task import BaseTask
from robodeploy.core.types import Action, Observation


class BasicFrankaPickTask(BaseTask):
    """A minimal pick sequence with approach, grasp, and lift phases."""

    def __init__(self, robot_id: int = 0):
        super().__init__(robot_id=robot_id)
        self._step = 0

        self._home = jnp.asarray([0.0, -0.7, 0.0, -2.1, 0.0, 1.6, 0.8], dtype=jnp.float32)
        self._pre_grasp = jnp.asarray([0.15, -0.95, 0.1, -2.2, 0.0, 1.75, 0.7], dtype=jnp.float32)
        self._grasp = jnp.asarray([0.2, -1.05, 0.15, -2.3, 0.0, 1.8, 0.75], dtype=jnp.float32)
        self._lift = jnp.asarray([0.1, -0.8, 0.05, -1.9, 0.0, 1.65, 0.7], dtype=jnp.float32)

    def get_observation_spec(self) -> dict:
        return {"rgb": True, "depth": False, "segmentation": False}

    def get_instruction(self) -> str:
        return "Pick the red cube and lift it above the table."

    def reset(self) -> None:
        self._step = 0

    def next_action(self, obs: Observation) -> Action:
        del obs

        self._step += 1

        if self._step < 120:
            return Action(joint_positions=self._home, gripper=0.0)
        if self._step < 240:
            return Action(joint_positions=self._pre_grasp, gripper=0.0)
        if self._step < 320:
            return Action(joint_positions=self._grasp, gripper=0.0)
        if self._step < 420:
            return Action(joint_positions=self._grasp, gripper=1.0)
        return Action(joint_positions=self._lift, gripper=1.0)

    def is_done(self) -> bool:
        return self._step >= 540
