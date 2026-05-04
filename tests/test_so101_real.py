"""SO-101 real hardware path: calibration, safety helpers, ROS2 auto-config, slew clamp."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from robodeploy.backends.real.ros2.controllers._clamp import slew_limit_command
from robodeploy.backends.real.ros2.safety import JointLimitGuard, SafetyError, TemperatureGuard, Watchdog
from robodeploy.backends.simulator import _ros2_auto_config
from robodeploy.behavior import BehaviorProfile
from robodeploy.core.robot import Robot, RobotTask
from robodeploy.core.spaces import ActionSpace
from robodeploy.core.types import Action, ObsSpec, Observation, SceneSpec
from robodeploy.description.so101 import SO101Description
from robodeploy.description.so101.calibration import JointCalibration, MissingCalibrationError, SO101Calibration
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


def _so101_robot() -> Robot:
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


def test_so101_calibration_roundtrip(tmp_path: Path) -> None:
    desc = SO101Description()
    lim = desc.joint_position_limits
    joints = tuple(
        JointCalibration(
            name=str(i + 1),
            motor_id=i + 1,
            zero_ticks=2000 + i * 10,
            ticks_per_rad=650.0 + i,
            soft_min_rad=float(lim[i, 0]),
            soft_max_rad=float(lim[i, 1]),
        )
        for i in range(6)
    )
    cal = SO101Calibration(joints=joints)
    p = tmp_path / "cal.json"
    cal.save(p)
    cal2 = SO101Calibration.load(p)
    q = np.array([0.01, -0.02, 0.0, 0.03, -0.01, 0.02], dtype=np.float64)
    t1 = cal.to_ticks(q)
    t2 = cal2.to_ticks(q)
    assert t1 == t2
    q_back = cal2.to_radians({k: float(v) for k, v in t1.items()})
    assert np.allclose(q_back, q, rtol=0, atol=5e-3)


def test_so101_calibration_locate_rejects_template() -> None:
    with pytest.raises(MissingCalibrationError):
        SO101Calibration.locate(explicit_path=str(SO101Calibration.bundled_example_path()), allow_template=False)


def test_slew_limit_command() -> None:
    q_des = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    q_cur = np.zeros(6)
    mv = np.full(6, 0.5)
    out = slew_limit_command(q_des, q_cur, max_joint_velocity=mv, command_hz=50.0)
    assert out[0] == pytest.approx(0.01)


def test_joint_limit_guard_velocity() -> None:
    lower = np.full(6, -3.0)
    upper = np.full(6, 3.0)
    vel = np.full(6, 1.0)
    g = JointLimitGuard(lower, upper, vel)
    g.check(np.zeros(6), dt=None)
    with pytest.raises(SafetyError):
        g.check(np.full(6, 10.0), dt=0.01)


def test_watchdog_fires_once() -> None:
    fired: list[str] = []

    def on_timeout() -> None:
        fired.append("x")

    w = Watchdog(0.15, on_timeout)
    w.arm()
    import time as _t

    _t.sleep(0.35)
    w.disarm()
    assert len(fired) == 1


def test_temperature_guard_calls_violation() -> None:
    calls: list[str] = []

    def on_v(r: str) -> None:
        calls.append(r)

    g = TemperatureGuard(lambda: {"1": 99.0}, max_c=70.0, period_s=0.05, on_violation=on_v)
    g.start()
    import time as _t

    _t.sleep(0.25)
    g.stop()
    assert len(calls) >= 1


def test_ros2_auto_config_feetech_only_for_real_world() -> None:
    robot = _so101_robot()
    r = BehaviorProfile().resolved()
    cfg_hw = _ros2_auto_config([robot], local_ros_graph=False, resolved=r, use_hardware_feetech=True)
    assert cfg_hw["robot0.controller"] == "so101_feetech"
    cfg_sim = _ros2_auto_config([robot], local_ros_graph=True, resolved=r, use_hardware_feetech=False)
    assert cfg_sim["robot0.controller"] == "joint_position"
    assert "dev_fake_sim" in cfg_sim


def test_ros2_auto_config_skips_fake_sim_for_hardware_path() -> None:
    robot = _so101_robot()
    r = BehaviorProfile().resolved()
    cfg = _ros2_auto_config([robot], local_ros_graph=True, resolved=r, use_hardware_feetech=True)
    assert "dev_fake_sim" not in cfg


def test_hardware_smoke_so101_port() -> None:
    port = os.environ.get("ROBODEPLOY_SO101_PORT", "").strip()
    if not port:
        pytest.skip("ROBODEPLOY_SO101_PORT not set")
    pytest.importorskip("rclpy")
    pytest.importorskip("lerobot")
    # Intentionally minimal: only verify FeetechMotorsBus can handshake when deps exist.
    from robodeploy.backends.real.ros2.controllers.so101_feetech import _build_motors_dict, _import_feetech_stack

    FeetechMotorsBus, Motor, MotorNormMode = _import_feetech_stack()
    motors = _build_motors_dict(Motor, MotorNormMode)
    bus = FeetechMotorsBus(port, motors)
    bus.connect(handshake=True)
    bus.disable_torque()
    bus.disconnect(disable_torque=True)
