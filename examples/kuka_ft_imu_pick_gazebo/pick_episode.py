"""Shared Gazebo pick-place episode runner for demos and live CI."""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


def _packaged_world(name: str) -> Path:
    try:
        from importlib.resources import files as resource_files

        path = Path(str(resource_files("robodeploy").joinpath(f"ros2_assets/worlds/{name}")))
        if path.is_file():
            return path
    except Exception:
        pass
    return REPO_ROOT / "tests" / "fixtures" / name


PICK_MINIMAL_WORLD = _packaged_world("pick_minimal.sdf")
if not PICK_MINIMAL_WORLD.is_file():
    PICK_MINIMAL_WORLD = REPO_ROOT / "tests" / "fixtures" / "gazebo_pick_minimal.sdf"
EMPTY_WORLD = _packaged_world("empty_world.sdf")
if not EMPTY_WORLD.is_file():
    EMPTY_WORLD = REPO_ROOT / "tests" / "fixtures" / "gazebo_empty.sdf"

# Live CI: strengthened from 1/3; WAVE2_01 target is 70% over 10 seeds.
LIVE_PICK_SEEDS = tuple(range(10))
LIVE_PICK_MIN_SUCCESS_RATE = 0.5

def _gazebo_place_snap_enabled() -> bool:
    # Mirrors ReachTrajectoryPolicy._gazebo_place_snap_enabled() env default
    # (instance method; cannot be called unbound from module scope).
    raw = os.environ.get("ROBODEPLOY_GAZEBO_PLACE_SNAP", "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


RELAXED_POLICY_CONFIG: dict[str, Any] = {
    "force_threshold": 0.3,
    "grasp_force_window": 2,
    "imu_omega_max": 0.8,
    "imu_settle_steps": 2,
    # Headless Gazebo Docker often has zero FT until contact physics is tuned; engage on reach.
    "grasp_detection": "distance",
    "critical_sensors": [],
    "halt_on_sensor_failure": False,
    # Match MuJoCo reach_pick_place.yaml carry clearance; kinematic avoids gz follow lag.
    "carry_mode": "kinematic",
    "carry_offset": [0.0, 0.0, -0.06],
    "tracking_blend": 0.42,
    "steps_per_phase": 280,
}

# Extra JTC horizon when measuring honest placement (ROBODEPLOY_GAZEBO_PLACE_SNAP=0).
HONEST_JTC_POLICY_OVERRIDES: dict[str, Any] = {
    "tracking_blend": 0.36,
    "honest_place_tracking_blend": 0.12,
    "steps_per_phase": 600,
    "honest_place_settle_m": 0.038,
}
RELAXED_TASK_KWARGS: dict[str, Any] = {"grasp_success_force_min": 0.0}


@dataclass(frozen=True)
class PickEpisodeResult:
    success: bool
    steps: int
    seed: int | None
    source_to_goal_distance: float | None
    contact_during_grasp: bool
    sensor_health_ok: bool
    final_info: Any | None = None


def _gazebo_headless_default() -> bool:
    raw = os.environ.get("ROBODEPLOY_GAZEBO_HEADLESS", "").strip().lower()
    if raw in {"1", "true", "yes"}:
        return True
    if raw in {"0", "false", "no"}:
        return False
    return True


def _gazebo_readiness_timeout_default() -> float:
    raw = os.environ.get("ROBODEPLOY_GAZEBO_READINESS_TIMEOUT", "").strip()
    if raw:
        try:
            return max(15.0, float(raw))
        except ValueError:
            pass
    return 90.0


def _gazebo_sim_cfg(
    *,
    world: Path,
    wait_for_topics: list[str] | None = None,
    headless: bool | None = None,
    readiness_timeout_s: float = 90.0,
) -> dict:
    return {
        "kind": "gazebo",
        "world": str(world),
        "headless": _gazebo_headless_default() if headless is None else headless,
        "readiness_timeout_s": readiness_timeout_s,
        "wait_for_topics": wait_for_topics or [],
    }


def kuka_ft_imu_pick_gazebo_cfg(
    *,
    max_episode_steps: int = 3200,
    policy_config: dict | None = None,
    task_kwargs: dict | None = None,
    wait_for_topics: list[str] | None = None,
    world: Path | None = None,
) -> dict:
    from examples.config import load_example_preset

    cfg = load_example_preset("kuka_ft_imu_pick_gazebo")
    policy_kwargs = dict(cfg.get("policy_kwargs", {}))
    merged_policy = {**dict(policy_kwargs.get("config", {})), **(policy_config or {})}
    policy_kwargs["config"] = merged_policy
    merged_task = {**dict(cfg.get("task_kwargs", {})), **(task_kwargs or {})}
    merged_task["max_steps"] = max_episode_steps
    if wait_for_topics is None:
        from robodeploy.core.sensor_rig import readiness_topics_from_sensor_rig_yaml

        rig_topics = readiness_topics_from_sensor_rig_yaml(
            list(cfg.get("sensor_rigs") or []),
            backend_name="gazebo",
        )
        topics = ["/joint_states", *rig_topics]
    else:
        topics = list(wait_for_topics)
    return {
        **cfg,
        "max_episode_steps": max_episode_steps,
        "policy_kwargs": policy_kwargs,
        "task_kwargs": merged_task,
        "backend_kwargs": {
            "config": {
                "sim":                 _gazebo_sim_cfg(
                    world=world or PICK_MINIMAL_WORLD,
                    wait_for_topics=topics,
                    readiness_timeout_s=_gazebo_readiness_timeout_default(),
                )
            }
        },
    }


def _pick_task(env) -> Any:
    tasks = env.robots[0].tasks
    return next(iter(tasks.values())).task


def source_to_goal_distance(obs, task) -> float | None:
    goal = task._placement_goal()
    source_pose = task.object_pose(task.source_name, obs)
    if source_pose is None:
        return None
    pos, _ = source_pose
    dx = float(pos[0]) - float(goal[0])
    dy = float(pos[1]) - float(goal[1])
    dz = float(pos[2]) - float(goal[2])
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _sensor_health_ok(info) -> bool:
    health = (getattr(info, "extra", None) or {}).get("sensor_health", {})
    return str(health.get("overall", "ok")) != "failed"


def _ee_pose_ready(env, driver) -> bool:
    if driver is None:
        return False
    diag = getattr(driver, "get_diagnostics", lambda: {})() or {}
    if diag.get("ee_pose_valid"):
        obs = driver.get_obs()
        ee = getattr(obs, "ee_position", None)
        if ee is not None and np.isfinite(np.asarray(ee, dtype=np.float64)).all():
            return True
    backend = env.backend
    solver = getattr(backend, "_kinematics_solver", None)
    if solver is None or driver is None:
        return False
    obs = driver.get_obs()
    q = getattr(obs, "joint_positions", None)
    if q is None or not np.isfinite(np.asarray(q, dtype=np.float64)).all():
        return False
    try:
        pos, _ = solver.fk(np.asarray(q, dtype=np.float64).reshape(-1))
        return bool(np.isfinite(np.asarray(pos, dtype=np.float64)).all())
    except Exception:
        return False


def _wait_for_ee_tf(env, *, timeout_s: float = 45.0) -> None:
    """Block until EE pose is usable (TF or URDF FK from joint_states)."""
    deadline = time.monotonic() + float(timeout_s)
    drivers = getattr(env.backend, "_drivers", {}) or {}
    driver = drivers.get(env.robots[0].robot_id) if env.robots else None
    while time.monotonic() < deadline:
        if _ee_pose_ready(env, driver):
            return
        time.sleep(0.15)
    raise TimeoutError(
        f"EE pose not valid after {timeout_s:.0f}s; "
        "check robot_state_publisher, /joint_states joint-name parity, or Pinocchio FK."
    )


def _ft_norm(obs) -> float:
    ft = getattr(obs, "ft_forces", {}) or {}
    wrench = ft.get("wrist_ft")
    if wrench is None:
        return 0.0
    return float(np.linalg.norm(np.asarray(wrench, dtype=np.float32)))


def run_pick_episode(
    *,
    seed: int | None = None,
    max_steps: int = 3200,
    cfg_overrides: dict | None = None,
    policy_config: dict | None = None,
    task_kwargs: dict | None = None,
    world: Path | None = None,
    on_step: Callable[[int, Any, Any, Any], None] | None = None,
) -> PickEpisodeResult:
    """Run one ``kuka_ft_imu_pick_gazebo`` episode to completion."""
    from robodeploy.env import RoboEnv

    merged_policy = {**RELAXED_POLICY_CONFIG, **(policy_config or {})}
    if not _gazebo_place_snap_enabled():
        merged_policy = {**merged_policy, **HONEST_JTC_POLICY_OVERRIDES}
    cfg = kuka_ft_imu_pick_gazebo_cfg(
        max_episode_steps=max_steps,
        policy_config=merged_policy,
        task_kwargs={**RELAXED_TASK_KWARGS, **(task_kwargs or {})},
        world=world,
    )
    backend_cfg = cfg.setdefault("backend_kwargs", {}).setdefault("config", {})
    backend_cfg["jtc_time_from_start_s"] = 1.0 if not _gazebo_place_snap_enabled() else 0.8
    backend_cfg.setdefault("recovery_max_retries", 12)
    if cfg_overrides:
        cfg = {**cfg, **cfg_overrides}

    env = RoboEnv.from_config(cfg)
    try:
        obs, info = env.reset(seed=seed)
        _wait_for_ee_tf(env, timeout_s=_gazebo_readiness_timeout_default() * 0.5)
        task = _pick_task(env)
        threshold = float(task.config.get("success_threshold", task.success_threshold))
        contact_during_grasp = False
        grasp_phase = False
        final_info = info
        steps = 0

        for step in range(max_steps):
            if on_step is not None:
                on_step(step, obs, final_info, env)

            if task.grasp_confirmed(obs):
                grasp_phase = True
            if grasp_phase:
                contact_state = getattr(obs, "contact_state", {}) or {}
                if contact_state.get("wrist_contact"):
                    contact_during_grasp = True
                has_contact = getattr(env.backend, "has_prop_contact", None)
                if callable(has_contact) and has_contact("source", other_body="ee_link"):
                    contact_during_grasp = True

            obs, _, done, final_info = env.step()
            steps = step + 1
            if done:
                extra = getattr(final_info, "extra", None) or {}
                if extra.get("truncated"):
                    pass  # step budget exhausted
                elif getattr(final_info, "failure", False):
                    print(
                        f"episode failure at step {steps} "
                        f"(safety={extra.get('safety', {})})",
                        flush=True,
                    )
                break

        distance = source_to_goal_distance(obs, task)
        success = bool(getattr(final_info, "success", False))
        if success and distance is not None:
            success = distance < threshold

        return PickEpisodeResult(
            success=success,
            steps=steps,
            seed=seed,
            source_to_goal_distance=distance,
            contact_during_grasp=contact_during_grasp,
            sensor_health_ok=_sensor_health_ok(final_info),
            final_info=final_info,
        )
    finally:
        env.close()


def run_pick_episodes(
    seeds: tuple[int, ...] | list[int],
    *,
    max_steps: int = 3200,
    **kwargs: Any,
) -> list[PickEpisodeResult]:
    return [run_pick_episode(seed=seed, max_steps=max_steps, **kwargs) for seed in seeds]


__all__ = [
    "EMPTY_WORLD",
    "LIVE_PICK_MIN_SUCCESS_RATE",
    "LIVE_PICK_SEEDS",
    "PICK_MINIMAL_WORLD",
    "PickEpisodeResult",
    "RELAXED_POLICY_CONFIG",
    "RELAXED_TASK_KWARGS",
    "kuka_ft_imu_pick_gazebo_cfg",
    "run_pick_episode",
    "run_pick_episodes",
    "source_to_goal_distance",
    "_ft_norm",
]
