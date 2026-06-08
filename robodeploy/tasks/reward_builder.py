"""Composable reward function builder for manipulation tasks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from robodeploy.core.types import Action, Observation

RewardFn = Callable[[Observation, Action], float]
ComponentFn = Callable[[Observation, Action], dict[str, float]]


@dataclass
class _RewardTerm:
    name: str
    fn: Callable[[Observation, Action], float]


def _vec3(values) -> tuple[float, float, float]:  # noqa: ANN001
    return (float(values[0]), float(values[1]), float(values[2]))


def _dist(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


class RewardBuilder:
    def __init__(self) -> None:
        self._terms: list[_RewardTerm] = []
        self._pose_resolver: Callable[[str, Observation], tuple[float, float, float] | None] | None = None

    def with_pose_resolver(
        self,
        resolver: Callable[[str, Observation], tuple[float, float, float] | None],
    ) -> RewardBuilder:
        self._pose_resolver = resolver
        return self

    def distance(
        self,
        source: str,
        target: str | tuple[float, float, float],
        *,
        scale: float = 1.0,
        name: str | None = None,
    ) -> RewardBuilder:
        term_name = name or f"dist_{source}_to_{target}"

        def _term(obs: Observation, _action: Action) -> float:
            src = self._resolve(source, obs)
            if src is None:
                return 0.0
            if isinstance(target, tuple):
                tgt = target
            else:
                tgt_pos = self._resolve(target, obs)
                if tgt_pos is None:
                    return 0.0
                tgt = tgt_pos
            return -float(scale) * _dist(src, tgt)

        self._terms.append(_RewardTerm(term_name, _term))
        return self

    def distance_to_point(
        self,
        source: str,
        point: tuple[float, float, float],
        *,
        scale: float = 1.0,
        name: str | None = None,
    ) -> RewardBuilder:
        return self.distance(
            source,
            point,
            scale=scale,
            name=name or f"dist_{source}_to_point",
        )

    def bonus_lift(
        self,
        source: str,
        *,
        initial_z: float,
        max_bonus: float = 0.1,
        threshold: float = 0.05,
        name: str | None = None,
    ) -> RewardBuilder:
        term_name = name or f"lift_{source}"

        def _term(obs: Observation, _action: Action) -> float:
            pos = self._resolve(source, obs)
            if pos is None:
                return 0.0
            lift = max(0.0, pos[2] - float(initial_z))
            if lift < threshold:
                return 0.0
            return min(lift, max_bonus)

        self._terms.append(_RewardTerm(term_name, _term))
        return self

    def bonus_in_zone(
        self,
        source: str,
        zone_center: tuple[float, float, float],
        zone_radius: float,
        *,
        scale: float = 1.0,
        name: str | None = None,
    ) -> RewardBuilder:
        term_name = name or f"zone_{source}"

        def _term(obs: Observation, _action: Action) -> float:
            pos = self._resolve(source, obs)
            if pos is None:
                return 0.0
            if _dist(pos, zone_center) <= zone_radius:
                return float(scale)
            return 0.0

        self._terms.append(_RewardTerm(term_name, _term))
        return self

    def penalty_action_norm(self, *, scale: float = 0.001, name: str = "action_norm") -> RewardBuilder:
        def _term(_obs: Observation, action: Action) -> float:
            jp = getattr(action, "joint_positions", None)
            if jp is None:
                return 0.0
            try:
                norm = float(sum(float(v) ** 2 for v in jp) ** 0.5)
            except Exception:
                return 0.0
            return -float(scale) * norm

        self._terms.append(_RewardTerm(name, _term))
        return self

    def penalty_force_above(self, threshold_N: float, *, scale: float = 0.1, name: str = "force_penalty") -> RewardBuilder:
        return self.penalty_excessive_force(threshold_N=threshold_N, scale=scale, name=name)

    def penalty_excessive_force(
        self,
        *,
        threshold_N: float = 20.0,
        scale: float = 0.1,
        name: str = "excessive_force",
    ) -> RewardBuilder:
        """Soft penalty if FT force exceeds threshold (collision avoidance)."""

        def _term(obs: Observation, _action: Action) -> float:
            ft = getattr(obs, "ft_force", None)
            if ft is None:
                return 0.0
            try:
                mag = float(sum(float(v) ** 2 for v in ft) ** 0.5)
            except Exception:
                return 0.0
            excess = max(0.0, mag - float(threshold_N))
            return -float(scale) * excess

        self._terms.append(_RewardTerm(name, _term))
        return self

    def bonus_grasp_force(
        self,
        *,
        min_N: float = 1.0,
        max_N: float = 5.0,
        scale: float = 0.05,
        name: str = "grasp_force_bonus",
    ) -> RewardBuilder:
        """Reward FT force in the grasp band."""

        def _term(obs: Observation, _action: Action) -> float:
            ft = getattr(obs, "ft_force", None)
            if ft is None:
                return 0.0
            try:
                mag = float(sum(float(v) ** 2 for v in ft) ** 0.5)
            except Exception:
                return 0.0
            if mag < min_N or mag > max_N:
                return 0.0
            band = max(float(max_N) - float(min_N), 1e-6)
            return float(scale) * (mag - float(min_N)) / band

        self._terms.append(_RewardTerm(name, _term))
        return self

    def penalty_jerk_imu(self, *, scale: float = 0.01, name: str = "imu_jerk") -> RewardBuilder:
        """Penalty on IMU acceleration jerk between steps."""
        state: dict[str, object] = {"prev": None}

        def _term(obs: Observation, _action: Action) -> float:
            accel = getattr(obs, "imu_acceleration", None)
            if accel is None:
                return 0.0
            arr = np.asarray(accel, dtype=np.float32)
            prev = state["prev"]
            state["prev"] = arr.copy()
            if prev is None:
                return 0.0
            jerk = float(np.linalg.norm(arr - np.asarray(prev, dtype=np.float32)))
            return -float(scale) * jerk

        self._terms.append(_RewardTerm(name, _term))
        return self

    def bonus_visual_alignment(
        self,
        *,
        target_hsv_range: tuple[tuple[float, float, float], tuple[float, float, float]],
        scale: float = 0.1,
        min_pixels: int = 50,
        name: str = "visual_alignment",
    ) -> RewardBuilder:
        """Bonus proportional to target blob pixel count in view."""

        def _term(obs: Observation, _action: Action) -> float:
            rgb = getattr(obs, "rgb", None)
            if rgb is None and getattr(obs, "images", None):
                images = obs.images
                if images:
                    rgb = next(iter(images.values()))
            if rgb is None:
                return 0.0
            from robodeploy.perception.vision_predicates import count_hsv_pixels

            lower, upper = target_hsv_range
            pixels = count_hsv_pixels(np.asarray(rgb), lower=lower, upper=upper)
            if pixels < int(min_pixels):
                return 0.0
            return float(scale) * min(1.0, pixels / float(min_pixels))

        self._terms.append(_RewardTerm(name, _term))
        return self

    def add_term(self, name: str, fn: Callable[[Observation, Action], float]) -> RewardBuilder:
        self._terms.append(_RewardTerm(name, fn))
        return self

    def build(self) -> RewardFn:
        components = self.build_components()

        def _reward(obs: Observation, action: Action) -> float:
            return sum(components(obs, action).values())

        return _reward

    def build_components(self) -> ComponentFn:
        terms = list(self._terms)

        def _components(obs: Observation, action: Action) -> dict[str, float]:
            return {t.name: float(t.fn(obs, action)) for t in terms}

        return _components

    def _resolve(self, key: str, obs: Observation) -> tuple[float, float, float] | None:
        if self._pose_resolver is not None:
            return self._pose_resolver(key, obs)
        if key == "ee":
            return _vec3(obs.ee_position)
        objects = getattr(obs, "objects", None) or {}
        if key in objects:
            pos, _ = objects[key]
            return tuple(float(v) for v in pos)
        return None
