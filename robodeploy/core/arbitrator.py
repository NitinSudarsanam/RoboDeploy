"""Arbitrator — explicit sequential task switching for shared robots."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

from robodeploy.core.task_config import TaskConfig
from robodeploy.core.types import ArbitrationEvent


class Arbitrator:
    """Manages active sequential tasks per robot."""

    def __init__(self, tasks: List[TaskConfig]) -> None:
        self._tasks_by_id: Dict[str, TaskConfig] = {task.task_id: task for task in tasks}
        self._task_ids_by_robot: dict[str, list[str]] = defaultdict(list)
        self._active_task_by_robot: dict[str, str] = {}
        self._events: list[ArbitrationEvent] = []

        for task_cfg in tasks:
            for robot_id in task_cfg.robot_ids:
                self._task_ids_by_robot[robot_id].append(task_cfg.task_id)

        for robot_id, task_ids in self._task_ids_by_robot.items():
            sequential_ids = [
                task_id for task_id in task_ids
                if self._tasks_by_id[task_id].mode == "sequential"
            ]
            if sequential_ids:
                self._active_task_by_robot[robot_id] = sequential_ids[0]

    def active_task_id(self, robot_id: str) -> Optional[str]:
        return self._active_task_by_robot.get(robot_id)

    def active_task(self, robot_id: str) -> Optional[TaskConfig]:
        task_id = self.active_task_id(robot_id)
        return self._tasks_by_id.get(task_id) if task_id else None

    def switch(self, robot_id: str, to_task_id: str, reason: str = "") -> ArbitrationEvent:
        """Switch the active sequential task for a robot."""
        if to_task_id not in self._tasks_by_id:
            raise KeyError(f"Unknown task_id '{to_task_id}'.")
        if robot_id not in self._task_ids_by_robot or to_task_id not in self._task_ids_by_robot[robot_id]:
            raise ValueError(f"Task '{to_task_id}' is not assigned to robot '{robot_id}'.")

        new_cfg = self._tasks_by_id[to_task_id]
        if new_cfg.mode != "sequential":
            raise ValueError("Arbitrator can only switch sequential TaskConfigs.")

        old_task_id = self._active_task_by_robot.get(robot_id, "")
        if old_task_id == to_task_id:
            event = ArbitrationEvent(robot_id=robot_id, from_task_id=old_task_id, to_task_id=to_task_id, reason=reason)
            self._events.append(event)
            return event

        if old_task_id:
            self._tasks_by_id[old_task_id].task.on_deactivate()
            old_policy = self._tasks_by_id[old_task_id].policy
            if old_policy is not None and not self._tasks_by_id[old_task_id].preserve_policy_state_on_deactivate:
                old_policy.reset()

        self._active_task_by_robot[robot_id] = to_task_id
        new_cfg.task.on_activate()
        event = ArbitrationEvent(robot_id=robot_id, from_task_id=old_task_id, to_task_id=to_task_id, reason=reason)
        self._events.append(event)
        return event

    def consume_events(self) -> list[ArbitrationEvent]:
        events = list(self._events)
        self._events.clear()
        return events

