"""Failure mode classification for benchmark episodes."""

from __future__ import annotations

from typing import Sequence

from robodeploy.core.types import Observation
from robodeploy.evaluation.metrics import EpisodeMetrics


class FailureClassifier:
    """Tag failed episodes with a heuristic category."""

    CATEGORIES = [
        "dropped",
        "missed_grasp",
        "off_target",
        "out_of_workspace",
        "exceeded_force",
        "timeout",
        "collision",
        "other",
    ]

    def __init__(
        self,
        *,
        force_threshold_N: float = 50.0,
        workspace_limit: int = 1,
        goal_miss_threshold: float = 0.15,
    ) -> None:
        self._force_threshold = float(force_threshold_N)
        self._workspace_limit = int(workspace_limit)
        self._goal_miss_threshold = float(goal_miss_threshold)

    def classify(
        self,
        metrics: EpisodeMetrics,
        trajectory: Sequence[Observation] | None = None,
    ) -> str:
        if metrics.success:
            return "other"

        meta = metrics.metadata or {}
        if meta.get("failure_category"):
            return str(meta["failure_category"])

        max_steps = int(meta.get("max_steps", 0) or 0)
        if metrics.time_to_success_steps is None and max_steps > 0 and metrics.steps >= max_steps:
            return "timeout"

        if metrics.collision_count > 0:
            return "collision"

        if metrics.max_force_N >= self._force_threshold:
            return "exceeded_force"

        if metrics.workspace_violations >= self._workspace_limit:
            return "out_of_workspace"

        if metrics.distance_to_goal_final >= self._goal_miss_threshold:
            if trajectory and self._detect_drop(trajectory):
                return "dropped"
            return "missed_grasp"

        if metrics.distance_to_goal_final > 0.05:
            return "off_target"

        return "other"

    @staticmethod
    def _detect_drop(trajectory: Sequence[Observation]) -> bool:
        try:
            import numpy as np
        except ImportError:
            return False
        zs = []
        for obs in trajectory:
            ee = getattr(obs, "ee_position", None)
            if ee is not None:
                zs.append(float(np.asarray(ee)[2]))
        if len(zs) < 3:
            return False
        peak = max(zs[: max(1, len(zs) // 2)])
        tail = zs[-max(3, len(zs) // 4) :]
        return peak - min(tail) > 0.08


def classify_episode_failures(
    episodes: list[EpisodeMetrics],
    trajectories: dict[int, list[Observation]] | None = None,
) -> dict[str, int]:
    clf = FailureClassifier()
    counts = {cat: 0 for cat in FailureClassifier.CATEGORIES}
    for idx, ep in enumerate(episodes):
        if ep.success:
            continue
        traj = (trajectories or {}).get(idx)
        cat = clf.classify(ep, traj)
        ep.metadata["failure_category"] = cat
        counts[cat] = counts.get(cat, 0) + 1
    return counts
