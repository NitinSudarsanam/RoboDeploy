"""Tests for simulator-neutral BehaviorProfile and backend_for_simulator wiring."""

from __future__ import annotations

import numpy as np
import pytest

from robodeploy.behavior import BehaviorProfile
from robodeploy.behavior_translators import to_mujoco, to_ros2
from robodeploy.backends import simulator as sim_mod
from robodeploy.backends.simulator import backend_for_simulator
from robodeploy.builtins import import_builtins
from robodeploy.core.robot import Robot, RobotTask
from robodeploy.core.spaces import ActionSpace
from robodeploy.core.types import Action, ObsSpec, Observation, SceneSpec
from robodeploy.description.so101 import SO101Description
from robodeploy.policies.base import PolicyBase
from robodeploy.tasks.base import TaskBase


class _MiniTask(TaskBase):
    def __init__(self) -> None:
        super().__init__(config={"max_steps": 2})

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


class _MiniPolicy(PolicyBase):
    def __init__(self) -> None:
        super().__init__(action_space=ActionSpace.JOINT_POS, config={"action_hz": 50.0})

    def _reset_impl(self) -> None:
        return

    def get_action(self, obs: Observation) -> Action:
        del obs
        return Action(joint_positions=np.zeros(6, dtype=np.float32))


def _minimal_robot() -> Robot:
    desc = SO101Description()
    return Robot(
        robot_id="robot0",
        description=desc,
        sensors=[],
        tasks={
            "demo": RobotTask(
                task=_MiniTask(),
                policies={"main": _MiniPolicy()},
                mode="sequential",
            )
        },
    )


def test_behavior_default_resolved() -> None:
    r = BehaviorProfile().resolved()
    assert r.control_hz == 50.0
    assert r.velocity_scale == 0.5
    assert r.kp == 10.0
    assert r.joint_damping == 1.0
    assert r.physics_timestep == 0.001
    assert r.physics_integrator == "RK4"


def test_behavior_smooth_merge_kp_scale() -> None:
    r = BehaviorProfile(preset="smooth", kp_scale=2.0).resolved()
    assert r.preset == "smooth"
    assert r.kp_scale == 2.0
    # soft baseline kp=5, * 2.0 = 10
    assert abs(r.kp - 10.0) < 1e-6


def test_to_mujoco_keys() -> None:
    robot = _minimal_robot()
    r = BehaviorProfile(preset="default").resolved()
    cfg = to_mujoco(r, robot)
    assert cfg["control_hz"] == 50.0
    assert "urdf_position_kp" in cfg
    assert "urdf_joint_damping" in cfg


def test_to_ros2_max_joint_velocity_shape() -> None:
    robot = _minimal_robot()
    desc = robot.description
    r = BehaviorProfile(preset="default").resolved()
    cfg = to_ros2(r, robot)
    assert cfg["command_hz"] == 50.0
    mv = cfg["robot0.max_joint_velocity"]
    assert isinstance(mv, list)
    assert len(mv) == desc.dof
    lim = np.asarray(desc.joint_velocity_limits, dtype=np.float64)
    for i, x in enumerate(mv):
        assert float(x) == pytest.approx(float(lim[i]) * 0.5)


def test_mujoco_merged_config_fast_preset() -> None:
    robot = _minimal_robot()
    resolved = sim_mod._resolve_behavior_profile([robot], BehaviorProfile(preset="fast"))
    cfg = sim_mod.merge_simulator_config(sim_mod._mujoco_auto_config([robot], resolved), {})
    assert float(cfg["urdf_position_kp"]) == pytest.approx(60.0)
    assert float(cfg["control_hz"]) == pytest.approx(100.0)


def test_mujoco_merged_config_override_wins() -> None:
    robot = _minimal_robot()
    resolved = sim_mod._resolve_behavior_profile([robot], BehaviorProfile(preset="fast"))
    base = sim_mod._mujoco_auto_config([robot], resolved)
    cfg = sim_mod.merge_simulator_config(base, {"urdf_position_kp": 7.0})
    assert float(cfg["urdf_position_kp"]) == pytest.approx(7.0)


def test_mujoco_merged_config_uses_so101_smooth_when_no_behavior() -> None:
    """SO101Description.default_behavior_profile is smooth → soft kp * 0.5 = 2.5."""
    robot = _minimal_robot()
    resolved = sim_mod._resolve_behavior_profile([robot], None)
    cfg = sim_mod.merge_simulator_config(sim_mod._mujoco_auto_config([robot], resolved), {})
    assert float(cfg["urdf_position_kp"]) == pytest.approx(2.5)


def test_mujoco_merged_config_caller_default_overrides_description_smooth() -> None:
    robot = _minimal_robot()
    resolved = sim_mod._resolve_behavior_profile([robot], BehaviorProfile(preset="default"))
    cfg = sim_mod.merge_simulator_config(sim_mod._mujoco_auto_config([robot], resolved), {})
    assert float(cfg["urdf_position_kp"]) == pytest.approx(10.0)


def test_backend_mujoco_fast_sets_kp_and_control_hz_if_mujoco() -> None:
    import_builtins()
    pytest.importorskip("mujoco", reason="mujoco not installed")
    robot = _minimal_robot()
    backend = backend_for_simulator("mujoco", robots=[robot], behavior=BehaviorProfile(preset="fast"))
    assert float(backend.config["urdf_position_kp"]) == pytest.approx(60.0)
    assert float(backend.control_hz) == pytest.approx(100.0)
