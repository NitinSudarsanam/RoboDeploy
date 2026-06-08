"""Standard per-episode and aggregate metrics for benchmark evaluation."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

try:
    import jax.numpy as jnp
except Exception:
    import numpy as jnp  # type: ignore[assignment]

from robodeploy.core.interfaces.task import ITask
from robodeploy.core.types import Action, EpisodeInfo, Observation
from robodeploy.description.base import RobotDescription


def _to_float_array(value) -> jnp.ndarray | None:  # noqa: ANN001
    if value is None:
        return None
    try:
        return jnp.asarray(value, dtype=jnp.float32)
    except Exception:
        return None


def ci95_binomial(values: Sequence[bool | int | float]) -> tuple[float, float]:
    """Normal-approximation 95% CI for a Bernoulli proportion."""
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    p = float(sum(bool(v) for v in values)) / float(n)
    if n == 1:
        return (p, p)
    z = 1.96
    margin = z * math.sqrt(max(p * (1.0 - p), 0.0) / n)
    return (max(0.0, p - margin), min(1.0, p + margin))


@dataclass
class EpisodeMetrics:
    success: bool
    reward_total: float
    reward_per_step: float
    steps: int
    time_to_success_steps: int | None
    time_to_success_seconds: float | None
    smoothness_jerk: float
    smoothness_action_norm: float
    smoothness_velocity: float
    collision_count: int
    max_force_N: float
    workspace_violations: int
    distance_to_goal_final: float
    distance_to_goal_min: float
    constraint_violations: dict[str, int]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AggregateMetrics:
    n_episodes: int
    success_rate: float
    success_rate_ci95: tuple[float, float]
    mean_reward: float
    std_reward: float
    median_time_to_success_steps: float | None
    mean_smoothness_jerk: float
    mean_smoothness_action_norm: float
    mean_collision_count: float
    robo_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["success_rate_ci95"] = list(self.success_rate_ci95)
        return payload


class MetricsCollector:
    """Per-episode metric accumulator."""

    def __init__(self, *, task: ITask, robot_description: RobotDescription) -> None:
        self._task = task
        self._robot_description = robot_description
        self.reset()

    def reset(self) -> None:
        self._reward_total = 0.0
        self._steps = 0
        self._success = False
        self._failure = False
        self._time_to_success_steps: int | None = None
        self._time_to_success_seconds: float | None = None
        self._prev_action: jnp.ndarray | None = None
        self._prev_velocity: jnp.ndarray | None = None
        self._jerk_sq_sum = 0.0
        self._jerk_count = 0
        self._action_norm_sum = 0.0
        self._velocity_sum = 0.0
        self._collision_count = 0
        self._max_force_N = 0.0
        self._workspace_violations = 0
        self._distance_to_goal_final = float("inf")
        self._distance_to_goal_min = float("inf")
        self._constraint_violations: dict[str, int] = {}
        self._metadata: dict[str, Any] = {}

    def observe(
        self,
        obs: Observation,
        action: Action | None,
        reward: float,
        info: EpisodeInfo,
    ) -> None:
        self._steps += 1
        self._reward_total += float(reward)

        if bool(info.success):
            self._success = True
            if self._time_to_success_steps is None:
                self._time_to_success_steps = self._steps
                self._time_to_success_seconds = float(info.sim_time or info.elapsed_time or 0.0)
        if bool(info.failure):
            self._failure = True

        action_vec = _to_float_array(action.joint_positions if action is not None else None)
        if action_vec is not None:
            norm = float(jnp.linalg.norm(action_vec))
            self._action_norm_sum += norm
            if self._prev_action is not None and action_vec.shape == self._prev_action.shape:
                delta = action_vec - self._prev_action
                self._jerk_sq_sum += float(jnp.sum(delta * delta))
                self._jerk_count += int(action_vec.shape[0])
            self._prev_action = action_vec

        vel = _to_float_array(obs.joint_velocities)
        if vel is not None:
            self._velocity_sum += float(jnp.mean(jnp.abs(vel)))
            if self._prev_velocity is not None and vel.shape == self._prev_velocity.shape:
                jerk = vel - self._prev_velocity
                self._jerk_sq_sum += float(jnp.sum(jerk * jerk))
                self._jerk_count += int(vel.shape[0])
            self._prev_velocity = vel

        diag = (getattr(info, "extra", {}) or {}).get("diagnostics") or {}
        if isinstance(diag, dict):
            self._collision_count += int(diag.get("collision_count", 0) or 0)
            self._max_force_N = max(self._max_force_N, float(diag.get("max_force_N", 0.0) or 0.0))
            self._workspace_violations += int(diag.get("workspace_violations", 0) or 0)
            violations = diag.get("constraint_violations")
            if isinstance(violations, dict):
                for key, count in violations.items():
                    self._constraint_violations[str(key)] = self._constraint_violations.get(str(key), 0) + int(
                        count or 0
                    )

        goal_dist = self._goal_distance(obs)
        if goal_dist is not None:
            self._distance_to_goal_final = goal_dist
            self._distance_to_goal_min = min(self._distance_to_goal_min, goal_dist)

    def _goal_distance(self, obs: Observation) -> float | None:
        try:
            if hasattr(self._task, "success_fn"):
                if bool(self._task.success_fn(obs)):
                    return 0.0
        except Exception:
            pass
        ee = _to_float_array(obs.ee_position)
        if ee is None:
            return None
        extra = getattr(obs, "extra", None)
        if isinstance(extra, dict):
            goal = extra.get("goal_position")
            if goal is not None:
                g = _to_float_array(goal)
                if g is not None:
                    return float(jnp.linalg.norm(ee - g))
        return float(jnp.linalg.norm(ee))

    def finalize(self) -> EpisodeMetrics:
        steps = max(self._steps, 1)
        jerk_rms = math.sqrt(self._jerk_sq_sum / max(self._jerk_count, 1)) if self._jerk_count else 0.0
        min_dist = self._distance_to_goal_min if math.isfinite(self._distance_to_goal_min) else 0.0
        final_dist = self._distance_to_goal_final if math.isfinite(self._distance_to_goal_final) else min_dist
        return EpisodeMetrics(
            success=bool(self._success and not self._failure),
            reward_total=self._reward_total,
            reward_per_step=self._reward_total / steps,
            steps=self._steps,
            time_to_success_steps=self._time_to_success_steps,
            time_to_success_seconds=self._time_to_success_seconds,
            smoothness_jerk=jerk_rms,
            smoothness_action_norm=self._action_norm_sum / steps,
            smoothness_velocity=self._velocity_sum / steps,
            collision_count=self._collision_count,
            max_force_N=self._max_force_N,
            workspace_violations=self._workspace_violations,
            distance_to_goal_final=final_dist,
            distance_to_goal_min=min_dist,
            constraint_violations=dict(self._constraint_violations),
            metadata=dict(self._metadata),
        )


def aggregate_episodes(
    metrics: Sequence[EpisodeMetrics],
    *,
    weights: Sequence[float] | None = None,
) -> AggregateMetrics:
    if not metrics:
        return AggregateMetrics(
            n_episodes=0,
            success_rate=0.0,
            success_rate_ci95=(0.0, 0.0),
            mean_reward=0.0,
            std_reward=0.0,
            median_time_to_success_steps=None,
            mean_smoothness_jerk=0.0,
            mean_smoothness_action_norm=0.0,
            mean_collision_count=0.0,
        )

    successes = [m.success for m in metrics]
    rewards = [m.reward_total for m in metrics]
    mean_reward = sum(rewards) / len(rewards)
    var = sum((r - mean_reward) ** 2 for r in rewards) / max(len(rewards) - 1, 1)
    success_times = [m.time_to_success_steps for m in metrics if m.success and m.time_to_success_steps is not None]
    success_times_sorted = sorted(success_times)
    median_tts: float | None = None
    if success_times_sorted:
        mid = len(success_times_sorted) // 2
        if len(success_times_sorted) % 2:
            median_tts = float(success_times_sorted[mid])
        else:
            median_tts = 0.5 * (success_times_sorted[mid - 1] + success_times_sorted[mid])

    robo_score: float | None = None
    if weights is not None and len(weights) == len(metrics):
        total_w = sum(weights)
        if total_w > 0:
            robo_score = sum(w * (1.0 if m.success else 0.0) for w, m in zip(weights, metrics)) / total_w

    return AggregateMetrics(
        n_episodes=len(metrics),
        success_rate=sum(successes) / len(successes),
        success_rate_ci95=ci95_binomial(successes),
        mean_reward=mean_reward,
        std_reward=math.sqrt(max(var, 0.0)),
        median_time_to_success_steps=median_tts,
        mean_smoothness_jerk=sum(m.smoothness_jerk for m in metrics) / len(metrics),
        mean_smoothness_action_norm=sum(m.smoothness_action_norm for m in metrics) / len(metrics),
        mean_collision_count=sum(m.collision_count for m in metrics) / len(metrics),
        robo_score=robo_score,
    )
