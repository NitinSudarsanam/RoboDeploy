"""Smoke tests for SO-101 URDF description and backend_for_simulator wiring."""

from __future__ import annotations

import numpy as np
import pytest

from robodeploy.builtins import import_builtins
from robodeploy.backends.real.ros2.backend import ROS2RealBackend
from robodeploy.backends.simulator import backend_for_simulator
from robodeploy.core.robot import Robot, RobotTask
from robodeploy.core.spaces import ActionSpace
from robodeploy.core.types import Action, ObsSpec, Observation, SceneSpec
from robodeploy.description.so101 import SO101Description
from robodeploy.policies.base import PolicyBase
from robodeploy.tasks.base import TaskBase


class _So101MiniTask(TaskBase):
    def __init__(self) -> None:
        super().__init__(config={"max_steps": 10})

    def obs_spec(self) -> ObsSpec:
        return ObsSpec(rgb=False, depth=False)

    def scene_spec(self) -> SceneSpec:
        return SceneSpec()

    def language_instruction(self) -> str:
        return ""

    def reset_fn(self, backend) -> None:
        return

    def reward_fn(self, obs: Observation, action: Action) -> float:
        return 0.0

    def success_fn(self, obs: Observation) -> bool:
        return False


class _So101MiniPolicy(PolicyBase):
    def __init__(self) -> None:
        super().__init__(action_space=ActionSpace.JOINT_POS, config={"action_hz": 50.0})

    def _reset_impl(self) -> None:
        return

    def get_action(self, obs: Observation) -> Action:
        del obs
        return Action(joint_positions=np.zeros(6, dtype=np.float32))


def _so101_robot() -> Robot:
    desc = SO101Description()
    return Robot(
        robot_id="robot0",
        description=desc,
        sensors=[],
        tasks={
            "demo": RobotTask(
                task=_So101MiniTask(),
                policies={"main": _So101MiniPolicy()},
                mode="sequential",
            )
        },
    )


def test_so101_description_parses_urdf() -> None:
    d = SO101Description()
    assert d.dof == 6
    assert d.joint_names == [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    ]
    assert d.home_qpos.shape == (6,)


def test_backend_for_simulator_real_world_returns_ros2() -> None:
    import_builtins()
    backend = backend_for_simulator("real_world", robots=[_so101_robot()])
    assert isinstance(backend, ROS2RealBackend)
    assert backend.config.get("rviz", {}).get("enabled") is False


def test_backend_for_simulator_mujoco_import_only() -> None:
    """MuJoCo backend constructs without init (SO101 has no MJCF — init would fail)."""
    import_builtins()
    pytest.importorskip("mujoco", reason="mujoco not installed")
    backend = backend_for_simulator("mujoco", robots=[_so101_robot()])
    assert type(backend).__name__ == "MuJoCoBackend"
