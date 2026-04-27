"""Task and policy selectors used for per-robot arbitration.

A `Robot` may have multiple tasks (sequential pool) and multiple policies per
task. Selectors decide which task is active and which policy's action is used.

Two stock selectors cover the common case:
  - WeightTaskSelector: picks the task with the highest weight.
  - WeightedPolicySelector: picks the policy with the highest weight.

Users wanting dynamic, observation-conditioned selection implement the
ITaskSelector / IPolicySelector protocols (or pass any callable matching
the same signature).
"""

from __future__ import annotations

from typing import Mapping, Protocol, runtime_checkable

from robodeploy.core.types import Action, Observation


@runtime_checkable
class ITaskSelector(Protocol):
    """Selects which sequential task is active for a robot at a given step."""

    def select(self, *, robot_id: str, obs: Observation, candidates: list[str]) -> str:
        """Return the chosen task_id from the candidate list."""
        ...


@runtime_checkable
class IPolicySelector(Protocol):
    """Combines actions produced by multiple policies bound to one task."""

    def select(
        self,
        *,
        robot_id: str,
        task_id: str,
        obs: Observation,
        candidate_actions: Mapping[str, Action],
    ) -> Action:
        """Return the resolved Action (winner-takes-all, blended, etc.)."""
        ...


class WeightTaskSelector:
    """Static-weight task selector. Picks highest-weight candidate.

    Ties broken by candidate order (stable). Unknown candidates ignored.
    Weights of zero are valid and may still be picked when no other candidate has
    positive weight (the first candidate wins).
    """

    def __init__(self, weights: Mapping[str, float]) -> None:
        self._weights: dict[str, float] = dict(weights)

    def select(self, *, robot_id: str, obs: Observation, candidates: list[str]) -> str:
        if not candidates:
            raise ValueError(f"No candidate tasks for robot '{robot_id}'.")
        best_id = candidates[0]
        best_w = self._weights.get(best_id, 0.0)
        for tid in candidates[1:]:
            w = self._weights.get(tid, 0.0)
            if w > best_w:
                best_id, best_w = tid, w
        return best_id

    def update(self, weights: Mapping[str, float]) -> None:
        self._weights.update(weights)


class WeightedPolicySelector:
    """Static-weight policy selector. Winner-takes-all (highest weight)."""

    def __init__(self, weights: Mapping[str, float]) -> None:
        self._weights: dict[str, float] = dict(weights)

    def select(
        self,
        *,
        robot_id: str,
        task_id: str,
        obs: Observation,
        candidate_actions: Mapping[str, Action],
    ) -> Action:
        if not candidate_actions:
            raise ValueError(
                f"No candidate actions for robot '{robot_id}' task '{task_id}'."
            )
        ids = list(candidate_actions.keys())
        best_id = ids[0]
        best_w = self._weights.get(best_id, 0.0)
        for pid in ids[1:]:
            w = self._weights.get(pid, 0.0)
            if w > best_w:
                best_id, best_w = pid, w
        return candidate_actions[best_id]

    def update(self, weights: Mapping[str, float]) -> None:
        self._weights.update(weights)
