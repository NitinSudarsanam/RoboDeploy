"""Construct backends from a single ``simulator`` name plus ``Robot`` aggregates.

Library users keep one ``Robot`` graph; switching MuJoCo vs ROS2+RViz vs Gazebo is
``backend_for_simulator(...)`` instead of hand-rolling per-backend config dicts.
Per-robot ROS topics and joint names are derived from each ``RobotDescription``.

Behavior (speed / stiffness / stability) is normalized via ``BehaviorProfile`` and
translated per backend so users do not tune simulator-specific constants by default.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from robodeploy.core.interfaces.backend import IBackend
from robodeploy.core.robot import Robot

if TYPE_CHECKING:
    from robodeploy.behavior import BehaviorProfile

SimulatorName = Literal["mujoco", "isaacsim", "ros2_rviz", "gazebo", "real_world"]

_BACKEND_TO_SIMULATOR: dict[str, SimulatorName] = {
    "mujoco": "mujoco",
    "isaacsim": "isaacsim",
    "ros2_rviz": "ros2_rviz",
    "ros2_gazebo": "gazebo",
    "ros2": "real_world",
}


def simulator_name_for_backend(backend_name: str) -> SimulatorName | None:
    """Map a registry / preset backend name to ``backend_for_simulator``'s ``simulator`` arg."""
    from robodeploy.core.registry import resolve_backend_name

    return _BACKEND_TO_SIMULATOR.get(resolve_backend_name(backend_name))


def normalize_backend_config_overrides(backend_kwargs: dict[str, Any] | None) -> dict[str, Any]:
    """Flatten preset ``backend_kwargs`` (including nested ``config``) for ``config_overrides``."""
    raw = dict(backend_kwargs or {})
    nested = raw.pop("config", None)
    merged: dict[str, Any] = dict(raw)
    if isinstance(nested, dict):
        merged = merge_simulator_config(merged, nested)
    return merged


def behavior_profile_from_config(cfg: dict[str, Any], backend_kwargs: dict[str, Any]) -> "BehaviorProfile | None":
    from robodeploy.behavior import BehaviorProfile

    spec = cfg.get("behavior") or backend_kwargs.get("behavior")
    if spec is None:
        return None
    if isinstance(spec, BehaviorProfile):
        return spec
    if isinstance(spec, str):
        return BehaviorProfile(preset=spec)  # type: ignore[arg-type]
    if isinstance(spec, dict):
        return BehaviorProfile(**spec)
    return None


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


def _resolve_behavior_profile(robots: list[Robot], behavior: "BehaviorProfile | None"):
    from robodeploy.behavior import BehaviorProfile

    base_pf = robots[0].description.default_behavior_profile()
    return base_pf.merged_with(behavior).resolved()


def _fake_joint_sim_spec(
    robot: Robot,
    *,
    publish_hz: float,
    max_joint_velocity: list[float] | None = None,
) -> dict[str, Any]:
    rid = robot.robot_id
    desc = robot.description
    out: dict[str, Any] = {
        "robot_ns": f"/{rid}",
        "joint_states_topic": "joint_states",
        "joint_pos_cmd_topic": "joint_position_commands",
        "joint_names": desc.ros_transport_joint_names(),
        "base_frame": desc.ros_base_frame_id(),
        "ee_frame": desc.ros_ee_frame_id(),
        "publish_hz": float(publish_hz),
    }
    if max_joint_velocity is not None:
        out["max_joint_velocity"] = tuple(float(x) for x in max_joint_velocity)
    return out


def _ros2_auto_config(
    robots: list[Robot],
    *,
    local_ros_graph: bool,
    resolved,
    use_hardware_feetech: bool = False,
) -> dict[str, Any]:
    from robodeploy.behavior_translators import to_ros2

    cfg: dict[str, Any] = {
        "rviz": {"enabled": True, "fixed_frame": "world", "publish_hz": 10.0},
        "controller_by_robot_id": {},
    }
    for r in robots:
        cfg = merge_simulator_config(cfg, to_ros2(resolved, r))
        rid = r.robot_id
        desc = r.description
        use_feetech = use_hardware_feetech and getattr(desc, "real_controller_name", "joint_position") == "so101_feetech"
        ctl = "so101_feetech" if use_feetech else "joint_position"
        cfg["controller_by_robot_id"][rid] = ctl
        cfg[f"{rid}.controller"] = ctl
        cfg[f"{rid}.joint_names"] = desc.ros_transport_joint_names()
        cfg[f"{rid}.joint_states_topic"] = "joint_states"
        cfg[f"{rid}.joint_cmd_topic"] = "joint_position_commands"
        cfg[f"{rid}.base_frame"] = desc.ros_base_frame_id()
        cfg[f"{rid}.ee_frame"] = desc.ros_ee_frame_id()

    if local_ros_graph:
        if robots:
            rviz = cfg.get("rviz")
            if isinstance(rviz, dict):
                rviz = dict(rviz)
                rviz["fixed_frame"] = robots[0].description.ros_base_frame_id()
                cfg["rviz"] = rviz
        # ``so101_feetech`` publishes ``joint_states`` from hardware; do not spawn ``dev_fake_sim``.
        skip_fake = any(
            use_hardware_feetech
            and getattr(r.description, "real_controller_name", "joint_position") == "so101_feetech"
            for r in robots
        )
        if not skip_fake:
            import numpy as np

            fake_specs: list[dict[str, Any]] = []
            for r in robots:
                vmax = (
                    np.asarray(r.description.joint_velocity_limits, dtype=np.float64) * float(resolved.velocity_scale)
                ).tolist()
                fake_specs.append(_fake_joint_sim_spec(r, publish_hz=float(resolved.control_hz), max_joint_velocity=vmax))
            cfg["dev_fake_sim"] = fake_specs
    return cfg


def _mujoco_auto_config(robots: list[Robot], resolved) -> dict[str, Any]:
    from robodeploy.behavior_translators import to_mujoco

    cfg: dict[str, Any] = {
        "enable_viewer": True,
        "allow_actuator_name_fallback": True,
    }
    cfg = merge_simulator_config(cfg, to_mujoco(resolved, robots[0]))
    if len(robots) == 1:
        extra = robots[0].description.mujoco_backend_extra_config()
        if isinstance(extra, dict):
            cfg = merge_simulator_config(cfg, extra)
    return cfg


def _isaacsim_auto_config(robots: list[Robot], resolved) -> dict[str, Any]:
    if len(robots) != 1:
        raise ValueError("Isaac Sim via backend_for_simulator currently supports exactly one Robot.")
    from robodeploy.behavior_translators import to_isaacsim

    base = {
        "headless": True,
        "usd_prefer": True,
        "usd_fallback_to_urdf": True,
    }
    return merge_simulator_config(base, to_isaacsim(resolved, robots[0]))


def _apply_control_hz(backend: IBackend, cfg: dict[str, Any], resolved) -> None:
    chz = float(cfg.get("control_hz", resolved.control_hz))
    setattr(backend, "control_hz", chz)


def backend_for_simulator(
    simulator: SimulatorName,
    *,
    robots: list[Robot],
    local_ros_graph: bool = False,
    config_overrides: dict[str, Any] | None = None,
    behavior: "BehaviorProfile | None" = None,
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
        behavior: Optional ``BehaviorProfile`` merged on top of
            ``robots[0].description.default_behavior_profile()`` before per-simulator translation.

    Raises:
        ValueError: If ``robots`` is empty or Gazebo lacks a ``sim`` dict after merging
            description + overrides.
    """
    if not robots:
        raise ValueError("backend_for_simulator requires at least one Robot.")

    overrides = dict(config_overrides or {})
    resolved = _resolve_behavior_profile(robots, behavior)

    if simulator == "mujoco":
        from robodeploy.backends.sim.mujoco.backend import MuJoCoBackend

        cfg = merge_simulator_config(_mujoco_auto_config(robots, resolved), overrides)
        backend = MuJoCoBackend(config=cfg)
        _apply_control_hz(backend, cfg, resolved)
        return backend

    if simulator == "isaacsim":
        from robodeploy.backends.sim.isaacsim.backend import IsaacSimBackend

        cfg = merge_simulator_config(_isaacsim_auto_config(robots, resolved), overrides)
        backend = IsaacSimBackend(config=cfg)
        _apply_control_hz(backend, cfg, resolved)
        return backend

    if simulator == "ros2_rviz":
        from robodeploy.backends.real.ros2.backend import ROS2RvizBackend

        cfg = merge_simulator_config(
            _ros2_auto_config(robots, local_ros_graph=local_ros_graph, resolved=resolved, use_hardware_feetech=False),
            overrides,
        )
        backend = ROS2RvizBackend(config=cfg)
        _apply_control_hz(backend, cfg, resolved)
        return backend

    if simulator == "real_world":
        from robodeploy.backends.real.ros2.backend import ROS2RealBackend

        base = _ros2_auto_config(robots, local_ros_graph=False, resolved=resolved, use_hardware_feetech=True)
        base["rviz"] = {"enabled": False}
        cfg = merge_simulator_config(base, overrides)
        backend = ROS2RealBackend(config=cfg)
        _apply_control_hz(backend, cfg, resolved)
        return backend

    if simulator == "gazebo":
        from robodeploy.behavior_translators import to_gazebo
        from robodeploy.backends.sim.gazebo.backend import ROS2GazeboBackend

        base = _ros2_auto_config(robots, local_ros_graph=False, resolved=resolved, use_hardware_feetech=False)
        base = merge_simulator_config(base, to_gazebo(resolved, robots[0]))
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
        backend = ROS2GazeboBackend(config=cfg)
        _apply_control_hz(backend, cfg, resolved)
        return backend

    raise ValueError(f"Unknown simulator: {simulator!r}")
