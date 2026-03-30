"""Basic scripted pick task for the Kuka demo robot."""

from __future__ import annotations

import jax.numpy as jnp

from robodeploy.core.task import BaseTask
from robodeploy.core.types import Action, Observation


class BasicKukaPickTask(BaseTask):
    """A minimal pick sequence with approach, grasp, and lift phases."""

    def __init__(self, robot_id: int = 0):
        super().__init__(robot_id=robot_id)
        self._step = 0

        self._home = jnp.asarray([0.0, -0.6, 0.0, -1.8, 0.0, 1.2, 0.0], dtype=jnp.float32)
        self._pre_grasp = jnp.asarray([0.1, -0.9, 0.05, -2.0, 0.0, 1.35, 0.05], dtype=jnp.float32)
        self._grasp = jnp.asarray([0.15, -1.0, 0.1, -2.1, 0.0, 1.45, 0.1], dtype=jnp.float32)
        self._lift = jnp.asarray([0.05, -0.75, 0.05, -1.85, 0.0, 1.3, 0.05], dtype=jnp.float32)

    def get_observation_spec(self) -> dict:
        return {"rgb": True, "depth": True, "segmentation": False}

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
