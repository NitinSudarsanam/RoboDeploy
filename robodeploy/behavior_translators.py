"""Map ``ResolvedBehaviorProfile`` to backend-native config keys."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from robodeploy.behavior import ResolvedBehaviorProfile

if TYPE_CHECKING:
    from robodeploy.core.robot import Robot


def to_mujoco(rp: ResolvedBehaviorProfile, robot: Robot) -> dict[str, Any]:
    del robot
    return {
        "control_hz": rp.control_hz,
        "urdf_position_kp": float(rp.kp),
        "urdf_joint_damping": float(rp.joint_damping),
        "urdf_joint_armature": 0.01,
        "urdf_min_mass": float(rp.min_mass),
        "urdf_min_inertia_diag": float(rp.min_inertia_diag),
        "urdf_timestep": float(rp.physics_timestep),
        "urdf_integrator": str(rp.physics_integrator),
    }


def to_ros2(rp: ResolvedBehaviorProfile, robot: Robot) -> dict[str, Any]:
    rid = str(robot.robot_id or "robot0")
    desc = robot.description
    vmax = np.asarray(desc.joint_velocity_limits, dtype=np.float64) * float(rp.velocity_scale)
    return {
        "control_hz": float(rp.control_hz),
        "command_hz": float(rp.control_hz),
        f"{rid}.command_hz": float(rp.control_hz),
        f"{rid}.max_joint_velocity": vmax.tolist(),
    }


def to_isaacsim(rp: ResolvedBehaviorProfile, robot: Robot) -> dict[str, Any]:
    del robot
    return {
        "control_hz": float(rp.control_hz),
        "behavior_preset": str(rp.preset),
        "behavior_kp": float(rp.kp),
        "behavior_damping": float(rp.joint_damping),
    }


def to_gazebo(rp: ResolvedBehaviorProfile, robot: Robot) -> dict[str, Any]:
    """v1: align command rate only; controller stiffness mapping is TODO."""
    del robot
    return {
        "control_hz": float(rp.control_hz),
        "command_hz": float(rp.control_hz),
    }


__all__ = ["to_mujoco", "to_ros2", "to_isaacsim", "to_gazebo"]
