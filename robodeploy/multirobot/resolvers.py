"""Task action resolvers for concurrent multi-task robots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from robodeploy.core.types import Action

ActionResolver = Callable[[str, list[Action]], Action]

_ACTION_RESOLVERS: dict[str, ActionResolver] = {}


def register_action_resolver(name: str):
    """Decorator: register a named task action resolver."""

    def decorator(fn: ActionResolver) -> ActionResolver:
        _ACTION_RESOLVERS[str(name)] = fn
        return fn

    return decorator


def get_action_resolver(name: str) -> ActionResolver:
    if name not in _ACTION_RESOLVERS:
        raise KeyError(
            f"Action resolver '{name}' not found. Registered: {sorted(_ACTION_RESOLVERS)}"
        )
    return _ACTION_RESOLVERS[name]


def list_action_resolvers() -> list[str]:
    return sorted(_ACTION_RESOLVERS)


def _joint_arrays(actions: list[Action]) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for action in actions:
        if action.joint_positions is None:
            continue
        out.append(np.asarray(action.joint_positions, dtype=np.float64).reshape(-1))
    return out


@register_action_resolver("average_joint")
def average_joint_actions(robot_id: str, actions: list[Action]) -> Action:
    del robot_id
    arrays = _joint_arrays(actions)
    if not arrays:
        return Action()
    merged = sum(arrays) / len(arrays)
    return Action(joint_positions=merged.astype(np.float32))


@register_action_resolver("priority")
def priority_select(
    robot_id: str,
    actions: list[Action],
    *,
    priority_order: list[str] | None = None,
    task_ids: list[str] | None = None,
) -> Action:
    del robot_id
    if not actions:
        return Action()
    if task_ids and priority_order:
        for task_id in priority_order:
            if task_id in task_ids:
                idx = task_ids.index(task_id)
                if idx < len(actions):
                    return actions[idx]
    return actions[0]


@register_action_resolver("weighted_blend")
def weighted_blend(
    robot_id: str,
    actions: list[Action],
    *,
    weights: list[float] | None = None,
) -> Action:
    del robot_id
    arrays = _joint_arrays(actions)
    if not arrays:
        return Action()
    w = np.asarray(weights if weights is not None else [1.0] * len(arrays), dtype=np.float64)
    if w.shape[0] != len(arrays):
        w = np.ones(len(arrays), dtype=np.float64)
    w = w / max(float(w.sum()), 1e-9)
    merged = sum(arr * weight for arr, weight in zip(arrays, w))
    return Action(joint_positions=merged.astype(np.float32))


@dataclass
class Box:
    """Axis-aligned safety zone (metres)."""

    xmin: float
    xmax: float
    ymin: float
    ymax: float
    zmin: float
    zmax: float


@register_action_resolver("shared_workspace_safe")
def shared_workspace_safe(
    robot_id: str,
    actions: list[Action],
    *,
    safety_zones: list[Box] | None = None,
) -> Action:
    """Pick the first action whose EE target (if present) lies outside safety zones."""
    del robot_id
    zones = list(safety_zones or [])
    for action in actions:
        if action.ee_position is None:
            return action
        pos = np.asarray(action.ee_position, dtype=np.float64).reshape(3)
        blocked = False
        for zone in zones:
            if (
                zone.xmin <= pos[0] <= zone.xmax
                and zone.ymin <= pos[1] <= zone.ymax
                and zone.zmin <= pos[2] <= zone.zmax
            ):
                blocked = True
                break
        if not blocked:
            return action
    return average_joint_actions("", actions)
