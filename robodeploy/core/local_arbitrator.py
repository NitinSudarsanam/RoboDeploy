"""Per-robot arbitrator. Replaces the env-wide Arbitrator.

Each Robot owns one LocalArbitrator. Responsibilities:
  - Track the active sequential task for this robot.
  - Re-evaluate the active task each step via an ITaskSelector.
  - Drive on_activate / on_deactivate hooks and reset policy state on switch.
  - Emit ArbitrationEvent records for inspection by RoboEnv.

Concurrent-mode tasks are not arbitrated here — they always run.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional, Protocol

from robodeploy.core.selectors import ITaskSelector
from robodeploy.core.types import ArbitrationEvent, Observation


class _TaskHandle(Protocol):
    """Minimum surface a RobotTask must expose to LocalArbitrator."""

    mode: str
    preserve_policy_state_on_deactivate: bool

    def on_activate(self) -> None: ...
    def on_deactivate(self) -> None: ...
    def reset_policies(self) -> None: ...


class LocalArbitrator:
    def __init__(
        self,
        robot_id: str,
        tasks: Mapping[str, _TaskHandle],
        task_selector: Optional[ITaskSelector] = None,
    ) -> None:
        self._robot_id = robot_id
        self._tasks: dict[str, _TaskHandle] = dict(tasks)
        self._selector = task_selector
        self._sequential_ids: list[str] = [
            tid for tid, t in self._tasks.items() if t.mode == "sequential"
        ]
        self._concurrent_ids: list[str] = [
            tid for tid, t in self._tasks.items() if t.mode == "concurrent"
        ]
        self._active_task_id: Optional[str] = self._initial_active_task_id()
        self._events: list[ArbitrationEvent] = []

    @property
    def robot_id(self) -> str:
        return self._robot_id

    @property
    def active_task_id(self) -> Optional[str]:
        return self._active_task_id

    @property
    def concurrent_task_ids(self) -> list[str]:
        return list(self._concurrent_ids)

    @property
    def sequential_task_ids(self) -> list[str]:
        return list(self._sequential_ids)

    def evaluate(self, obs: Observation) -> Optional[str]:
        """Re-pick the active sequential task for this step.

        With a selector configured and >1 sequential candidates, ask it.
        Otherwise the active task is unchanged. Emits an event on switch.
        """
        if self._selector is None:
            return self._active_task_id
        chosen = self._selector.select(
            robot_id=self._robot_id, obs=obs, candidates=list(self._sequential_ids)
        )
        if chosen != self._active_task_id:
            self._do_switch(chosen, reason="selector")
        return self._active_task_id

    def switch(self, to_task_id: str, reason: str = "") -> ArbitrationEvent:
        if to_task_id not in self._tasks:
            raise KeyError(
                f"Robot '{self._robot_id}' has no task '{to_task_id}'."
            )
        if self._tasks[to_task_id].mode != "sequential":
            raise ValueError(
                f"Cannot switch to non-sequential task '{to_task_id}'."
            )
        return self._do_switch(to_task_id, reason=reason)

    def consume_events(self) -> list[ArbitrationEvent]:
        events = list(self._events)
        self._events.clear()
        return events

    def active_and_concurrent(self) -> Iterable[str]:
        if self._active_task_id is not None:
            yield self._active_task_id
        for tid in self._concurrent_ids:
            yield tid

    def _do_switch(self, to_task_id: str, *, reason: str) -> ArbitrationEvent:
        from_id = self._active_task_id or ""
        if from_id == to_task_id:
            event = ArbitrationEvent(
                robot_id=self._robot_id,
                from_task_id=from_id,
                to_task_id=to_task_id,
                reason=reason,
            )
            self._events.append(event)
            return event

        if from_id:
            old = self._tasks[from_id]
            old.on_deactivate()
            if not old.preserve_policy_state_on_deactivate:
                old.reset_policies()

        self._active_task_id = to_task_id
        self._tasks[to_task_id].on_activate()

        event = ArbitrationEvent(
            robot_id=self._robot_id,
            from_task_id=from_id,
            to_task_id=to_task_id,
            reason=reason,
        )
        self._events.append(event)
        return event

    def _initial_active_task_id(self) -> Optional[str]:
        if not self._sequential_ids:
            return None
        if self._selector is None:
            return self._sequential_ids[0]
        try:
            chosen = self._selector.select(
                robot_id=self._robot_id,
                obs=None,  # type: ignore[arg-type]
                candidates=list(self._sequential_ids),
            )
        except Exception:
            return self._sequential_ids[0]
        if chosen not in self._sequential_ids:
            raise KeyError(f"Task selector chose unknown task '{chosen}' for robot '{self._robot_id}'.")
        return chosen
