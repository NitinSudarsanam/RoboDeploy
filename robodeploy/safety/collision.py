"""Sim-only collision guard using backend contact queries."""

from __future__ import annotations

from typing import TYPE_CHECKING

from robodeploy.core.types import Action, Observation

from .violation import Hazard, Severity, ViolationRecord

if TYPE_CHECKING:
    from robodeploy.core.interfaces.backend import IBackend


class CollisionGuard:
    """Halts when the backend reports disallowed body contacts."""

    def __init__(
        self,
        *,
        allowed_pairs: list[tuple[str, str]] | None = None,
        disallowed_pairs: list[tuple[str, str]] | None = None,
        backend: IBackend | None = None,
        robot_id: str | None = None,
    ) -> None:
        self.robot_id = robot_id
        self._allowed = {tuple(p) for p in (allowed_pairs or [])}
        self._disallowed = list(disallowed_pairs or [])
        self._backend = backend

    def set_backend(self, backend: IBackend) -> None:
        self._backend = backend

    def check_action(
        self,
        action: Action,
        obs: Observation,
        *,
        dt: float,
    ) -> tuple[Action, list[ViolationRecord]]:
        del obs, dt
        return action, []

    def check_observation(self, obs: Observation) -> list[ViolationRecord]:
        del obs
        if self._backend is None:
            return []

        violations: list[ViolationRecord] = []
        query = getattr(self._backend, "has_prop_contact", None)
        if not callable(query):
            return []

        for body_a, body_b in self._disallowed:
            if (body_a, body_b) in self._allowed or (body_b, body_a) in self._allowed:
                continue
            if self._pair_in_contact(query, body_a, body_b):
                violations.append(
                    ViolationRecord(
                        hazard=Hazard.COLLISION_IMMINENT,
                        severity=Severity.CRITICAL,
                        message=f"contact between {body_a!r} and {body_b!r}",
                    )
                )
        return violations

    @staticmethod
    def _pair_in_contact(query, body_a: str, body_b: str) -> bool:
        try:
            return bool(query(body_a, other_body=body_b))
        except TypeError:
            return bool(query(body_a))
