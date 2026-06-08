"""Environment state snapshots for rollback and replay debugging."""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robodeploy.core.seeding import capture_rng_state, restore_rng_state
from robodeploy.core.types import Action, Observation

if TYPE_CHECKING:
    from robodeploy.env import RoboEnv


@dataclass
class StateSnapshot:
    timestamp: float
    step: int
    episode_id: str
    sim_state: dict[str, Any] | None
    obs: Observation
    last_action: Action | None
    policy_state: dict[str, Any] | None
    rng_state: dict[str, Any]


class SnapshotManager:
    """Save/load environment state for rollback."""

    def __init__(self, *, env: "RoboEnv", max_keep: int = 100) -> None:
        self._env = env
        self._max_keep = max(1, int(max_keep))
        self._snapshots: list[StateSnapshot] = []

    @property
    def snapshots(self) -> list[StateSnapshot]:
        return list(self._snapshots)

    def capture(self, *, last_action: Action | None = None) -> StateSnapshot:
        obs_by_robot = self._env.get_processed_obs_by_robot()
        primary = self._env.primary_robot.robot_id
        obs = obs_by_robot[primary]
        info = getattr(self._env, "_episode_info", None)
        snap = StateSnapshot(
            timestamp=time.time(),
            step=int(getattr(info, "step", 0) or 0),
            episode_id=str(getattr(info, "episode_id", 0)),
            sim_state=self._backend_sim_state(),
            obs=obs,
            last_action=last_action,
            policy_state=self._policy_state(),
            rng_state=capture_rng_state(),
        )
        self._snapshots.append(snap)
        if len(self._snapshots) > self._max_keep:
            self._snapshots = self._snapshots[-self._max_keep :]
        return snap

    def restore(self, snapshot: StateSnapshot) -> None:
        if snapshot.sim_state is not None:
            setter = getattr(self._env.backend, "set_sim_state", None)
            if callable(setter):
                setter(snapshot.sim_state)
        restore_rng_state(snapshot.rng_state)

    def rollback(self, n_steps: int = 1) -> StateSnapshot:
        n = int(n_steps)
        if len(self._snapshots) < n:
            raise ValueError(f"Cannot rollback {n} steps; only {len(self._snapshots)} snapshots stored.")
        snap = self._snapshots[-n]
        self.restore(snap)
        self._snapshots = self._snapshots[:-n]
        return snap

    def save(self, path: str | Path) -> None:
        Path(path).write_bytes(pickle.dumps(self._snapshots))

    def load(self, path: str | Path) -> None:
        self._snapshots = pickle.loads(Path(path).read_bytes())

    def _backend_sim_state(self) -> dict[str, Any] | None:
        getter = getattr(self._env.backend, "get_sim_state", None)
        if not callable(getter):
            return None
        try:
            state = getter()
            return dict(state) if state is not None else None
        except NotImplementedError:
            return None

    def _policy_state(self) -> dict[str, Any] | None:
        out: dict[str, Any] = {}
        for robot in self._env.robots:
            for robot_task in robot.tasks.values():
                for policy_id, policy in robot_task.policies.items():
                    diag = getattr(policy, "diagnostics", None)
                    if callable(diag):
                        out[f"{robot.robot_id}/{policy_id}"] = diag()
        return out or None
