"""Shared Gazebo pick-place episode runner for demos and live CI."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
PICK_MINIMAL_WORLD = REPO_ROOT / "tests" / "fixtures" / "gazebo_pick_minimal.sdf"
EMPTY_WORLD = REPO_ROOT / "tests" / "fixtures" / "gazebo_empty.sdf"

# Live CI: strengthened from 1/3; WAVE2_01 target is 70% over 10 seeds.
LIVE_PICK_SEEDS = tuple(range(10))
LIVE_PICK_MIN_SUCCESS_RATE = 0.5

RELAXED_POLICY_CONFIG: dict[str, Any] = {
    "force_threshold": 0.3,
    "grasp_force_window": 2,
    "imu_omega_max": 0.8,
    "imu_settle_steps": 2,
}
RELAXED_TASK_KWARGS: dict[str, Any] = {"grasp_success_force_min": 0.5}


@dataclass(frozen=True)
class PickEpisodeResult:
    success: bool
    steps: int
    seed: int | None
    source_to_goal_distance: float | None
    contact_during_grasp: bool
    sensor_health_ok: bool
    final_info: Any | None = None


def _gazebo_sim_cfg(
    *,
    world: Path,
    wait_for_topics: list[str] | None = None,
    headless: bool = True,
    readiness_timeout_s: float = 90.0,
) -> dict:
    return {
        "kind": "gazebo",
        "world": str(world),
        "headless": headless,
        "readiness_timeout_s": readiness_timeout_s,
        "wait_for_topics": wait_for_topics or [],
    }


def kuka_ft_imu_pick_gazebo_cfg(
    *,
    max_episode_steps: int = 1200,
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
    topics = wait_for_topics or [
        "/joint_states",
        "/wrist_ft/wrench",
        "/wrist_imu/imu",
        "/wrist_camera/image_raw",
    ]
    return {
        **cfg,
        "max_episode_steps": max_episode_steps,
        "policy_kwargs": policy_kwargs,
        "task_kwargs": merged_task,
        "backend_kwargs": {
            "config": {
                "sim": _gazebo_sim_cfg(
                    world=world or PICK_MINIMAL_WORLD,
                    wait_for_topics=topics,
                    readiness_timeout_s=90.0,
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


def _ft_norm(obs) -> float:
    ft = getattr(obs, "ft_forces", {}) or {}
    wrench = ft.get("wrist_ft")
    if wrench is None:
        return 0.0
    return float(np.linalg.norm(np.asarray(wrench, dtype=np.float32)))


def run_pick_episode(
    *,
    seed: int | None = None,
    max_steps: int = 1200,
    cfg_overrides: dict | None = None,
    policy_config: dict | None = None,
    task_kwargs: dict | None = None,
    world: Path | None = None,
    on_step: Callable[[int, Any, Any, Any], None] | None = None,
) -> PickEpisodeResult:
    """Run one ``kuka_ft_imu_pick_gazebo`` episode to completion."""
    from robodeploy.env import RoboEnv

    cfg = kuka_ft_imu_pick_gazebo_cfg(
        max_episode_steps=max_steps,
        policy_config={**RELAXED_POLICY_CONFIG, **(policy_config or {})},
        task_kwargs={**RELAXED_TASK_KWARGS, **(task_kwargs or {})},
        world=world,
    )
    if cfg_overrides:
        cfg = {**cfg, **cfg_overrides}

    env = RoboEnv.from_config(cfg)
    try:
        obs, info = env.reset(seed=seed)
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
    max_steps: int = 1200,
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
