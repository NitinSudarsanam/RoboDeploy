"""Multi-phase task choreography DSL for pour/insertion tasks."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

from robodeploy.core.types import Observation
from robodeploy.tasks.success_predicates import get_success_predicate


@dataclass(frozen=True)
class ChoreographyPhase:
    kind: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskChoreography:
    """YAML-driven multi-step task phases (reach → tilt → verify, etc.)."""

    phases: list[ChoreographyPhase]
    _phase_idx: int = 0
    _hold_steps: int = 0

    @classmethod
    def from_yaml(cls, path: Path | str) -> TaskChoreography:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Choreography YAML must be a mapping: {path}")
        if "phases" in raw:
            data = raw
        else:
            data = next(iter(raw.values()))
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskChoreography:
        phases: list[ChoreographyPhase] = []
        for entry in data.get("phases", []) or []:
            if not isinstance(entry, dict) or len(entry) != 1:
                raise ValueError(f"Each phase must be a single-key mapping: {entry!r}")
            kind, params = next(iter(entry.items()))
            phases.append(ChoreographyPhase(kind=str(kind), params=dict(params or {})))
        if not phases:
            raise ValueError("Choreography requires at least one phase.")
        return cls(phases=phases)

    def reset(self) -> None:
        self._phase_idx = 0
        self._hold_steps = 0

    @property
    def current(self) -> ChoreographyPhase:
        return self.phases[min(self._phase_idx, len(self.phases) - 1)]

    @property
    def complete(self) -> bool:
        return self._phase_idx >= len(self.phases)

    def advance(self, obs: Observation, *, pose_resolver: Callable[[str, Observation], Any]) -> bool:
        """Return True when the active phase is satisfied and index advances."""
        if self.complete:
            return True
        phase = self.current
        if phase.kind == "reach":
            target = str(phase.params.get("target", ""))
            threshold = float(phase.params.get("threshold", 0.05))
            pose = pose_resolver(target, obs)
            if pose is None:
                return False
            ee = tuple(float(v) for v in obs.ee_position)
            if _dist3(ee, tuple(float(v) for v in pose[:3])) <= threshold:
                self._phase_idx += 1
                self._hold_steps = 0
                return True
            return False

        if phase.kind == "tilt":
            hold_steps = int(phase.params.get("hold_steps", 30))
            self._hold_steps += 1
            if self._hold_steps >= hold_steps:
                self._phase_idx += 1
                self._hold_steps = 0
                return True
            return False

        if phase.kind == "align":
            target = str(phase.params.get("target", ""))
            threshold = float(phase.params.get("threshold", 0.04))
            pose = pose_resolver(target, obs)
            if pose is None:
                return False
            obj_name = str(phase.params.get("object", phase.params.get("peg", "")))
            obj_pose = pose_resolver(obj_name, obs)
            if obj_pose is None:
                return False
            if _dist3(tuple(float(v) for v in obj_pose[:3]), tuple(float(v) for v in pose[:3])) <= threshold:
                self._phase_idx += 1
                self._hold_steps = 0
                return True
            return False

        if phase.kind == "verify":
            predicate = str(phase.params.get("predicate", ""))
            fn = get_success_predicate(predicate)
            kwargs = {k: v for k, v in phase.params.items() if k != "predicate"}
            if fn(obs, **kwargs):
                self._phase_idx += 1
                self._hold_steps = 0
                return True
            return False

        if phase.kind == "insert":
            predicate = str(phase.params.get("predicate", "peg_in_hole"))
            fn = get_success_predicate(predicate)
            kwargs = {k: v for k, v in phase.params.items() if k != "predicate"}
            if fn(obs, **kwargs):
                self._phase_idx += 1
                self._hold_steps = 0
                return True
            return False

        self._phase_idx += 1
        return True

    def phase_reward_scale(self) -> float:
        """Encourage progress through choreography phases."""
        if not self.phases:
            return 1.0
        return 0.5 + 0.5 * (self._phase_idx / max(len(self.phases), 1))


def _dist3(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))
