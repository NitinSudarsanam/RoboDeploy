"""Construct backends from a single ``simulator`` name plus ``Robot`` aggregates.

Library users keep one ``Robot`` graph; switching MuJoCo vs ROS2+RViz vs Gazebo is
``backend_for_simulator(...)`` instead of hand-rolling per-backend config dicts.
Per-robot ROS topics and joint names are derived from each ``RobotDescription``.
"""

from __future__ import annotations

from typing import Any, Literal

from robodeploy.core.interfaces.backend import IBackend
from robodeploy.core.robot import Robot

SimulatorName = Literal["mujoco", "isaacsim", "ros2_rviz", "gazebo", "real_world"]


def merge_simulator_config(base: dict[str, Any], overrides: dict[str, Any] | None) -> dict[str, Any]:
    """Deep-merge ``overrides`` into ``base`` (dict values merge recursively)."""
    if not overrides:
        return dict(base)
    out: dict[str, Any] = dict(base)
    for key, val in overrides.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = merge_simulator_config(out[key], val)
        else:
            out[key] = val
    return out


def _fake_joint_sim_spec(robot: Robot) -> dict[str, Any]:
    rid = robot.robot_id
    desc = robot.description
    return {
        "robot_ns": f"/{rid}",
        "joint_states_topic": "joint_states",
        "joint_pos_cmd_topic": "joint_position_commands",
        "joint_names": desc.ros_transport_joint_names(),
        "base_frame": desc.ros_base_frame_id(),
        "ee_frame": desc.ros_ee_frame_id(),
        "publish_hz": 100.0,
    }


def _ros2_auto_config(robots: list[Robot], *, local_ros_graph: bool) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "rviz": {"enabled": True, "fixed_frame": "world", "publish_hz": 10.0},
        "command_hz": 50.0,
        "controller_by_robot_id": {},
    }
    for r in robots:
        rid = r.robot_id
        desc = r.description
        cfg["controller_by_robot_id"][rid] = "joint_position"
        cfg[f"{rid}.joint_names"] = desc.ros_transport_joint_names()
        cfg[f"{rid}.joint_states_topic"] = "joint_states"
        cfg[f"{rid}.joint_cmd_topic"] = "joint_position_commands"
        cfg[f"{rid}.base_frame"] = desc.ros_base_frame_id()
        cfg[f"{rid}.ee_frame"] = desc.ros_ee_frame_id()
    if local_ros_graph:
        # In dev_fake_sim mode we don't have a simulator publishing a global `world` TF frame.
        # Use the robot base frame so RViz RobotModel appears without extra TF plumbing.
        if robots:
            rviz = cfg.get("rviz")
            if isinstance(rviz, dict):
                rviz = dict(rviz)
                rviz["fixed_frame"] = robots[0].description.ros_base_frame_id()
                cfg["rviz"] = rviz
        cfg["dev_fake_sim"] = [_fake_joint_sim_spec(r) for r in robots]
    return cfg


def _mujoco_auto_config(robots: list[Robot]) -> dict[str, Any]:
    if len(robots) != 1:
        raise ValueError("MuJoCo via backend_for_simulator currently supports exactly one Robot.")
    cfg: dict[str, Any] = {"enable_viewer": True, "allow_actuator_name_fallback": True}
    extra = robots[0].description.mujoco_backend_extra_config()
    if isinstance(extra, dict):
        cfg = merge_simulator_config(cfg, extra)
    return cfg


def _isaacsim_auto_config(robots: list[Robot]) -> dict[str, Any]:
    if len(robots) != 1:
        raise ValueError("Isaac Sim via backend_for_simulator currently supports exactly one Robot.")
    return {
        "headless": True,
        "usd_prefer": True,
        "usd_fallback_to_urdf": True,
    }


def backend_for_simulator(
    simulator: SimulatorName,
    *,
    robots: list[Robot],
    local_ros_graph: bool = False,
    config_overrides: dict[str, Any] | None = None,
) -> IBackend:
    """Return a backend instance wired for ``simulator`` using ``robots`` descriptions.

    Args:
        simulator: ``"mujoco"`` | ``"isaacsim"`` | ``"ros2_rviz"`` | ``"gazebo"`` | ``"real_world"``.
        robots: Non-empty list passed to ``RoboEnv``; used to derive per-robot ROS
            joint names, frames, and optional Gazebo ``sim`` layout.
        local_ros_graph: For ``ros2_rviz`` only: start embedded ``dev_fake_sim`` joint
            publishers so no external robot graph is required.
        config_overrides: Merged last (wins on conflicts). For Gazebo, you may supply
            ``{"sim": {...}}`` when ``RobotDescription.gazebo_sim_launch_config()`` is
            ``None``.

    Raises:
        ValueError: If ``robots`` is empty, MuJoCo gets more than one robot, or Gazebo
            lacks a ``sim`` dict after merging description + overrides.
    """
    if not robots:
        raise ValueError("backend_for_simulator requires at least one Robot.")

    overrides = dict(config_overrides or {})

    if simulator == "mujoco":
        from robodeploy.backends.sim.mujoco.backend import MuJoCoBackend

        cfg = merge_simulator_config(_mujoco_auto_config(robots), overrides)
        return MuJoCoBackend(config=cfg)

    if simulator == "isaacsim":
        from robodeploy.backends.sim.isaacsim.backend import IsaacSimBackend

        cfg = merge_simulator_config(_isaacsim_auto_config(robots), overrides)
        return IsaacSimBackend(config=cfg)

    if simulator == "ros2_rviz":
        from robodeploy.backends.real.ros2.backend import ROS2RealBackend

        cfg = merge_simulator_config(_ros2_auto_config(robots, local_ros_graph=local_ros_graph), overrides)
        return ROS2RealBackend(config=cfg)

    if simulator == "real_world":
        from robodeploy.backends.real.ros2.backend import ROS2RealBackend

        base = _ros2_auto_config(robots, local_ros_graph=False)
        base["rviz"] = {"enabled": False}
        cfg = merge_simulator_config(base, overrides)
        return ROS2RealBackend(config=cfg)

    if simulator == "gazebo":
        from robodeploy.backends.sim.gazebo.backend import ROS2GazeboBackend

        base = _ros2_auto_config(robots, local_ros_graph=False)
        for r in robots:
            extra = r.description.gazebo_ros2_extra_config(r.robot_id)
            if isinstance(extra, dict):
                base = merge_simulator_config(base, extra)
        sim_from_desc = robots[0].description.gazebo_sim_launch_config()
        sim_from_over = overrides.pop("sim", None)
        sim_merged = merge_simulator_config(sim_from_desc or {}, sim_from_over if isinstance(sim_from_over, dict) else {})
        if not isinstance(sim_merged, dict) or not str(sim_merged.get("kind", "")).strip().lower():
            sim_merged = merge_simulator_config({"kind": "gazebo"}, sim_merged)
        if not sim_merged.get("world"):
            raise ValueError(
                "Gazebo requires a sim world path. Implement RobotDescription.gazebo_sim_launch_config() "
                "or pass config_overrides['sim'] including at least 'world'."
            )
        base["sim"] = sim_merged
        rviz = base.get("rviz")
        if isinstance(rviz, dict):
            rviz = dict(rviz)
            rviz.setdefault("enabled", True)
            base["rviz"] = rviz
        cfg = merge_simulator_config(base, overrides)
        return ROS2GazeboBackend(config=cfg)

    raise ValueError(f"Unknown simulator: {simulator!r}")
