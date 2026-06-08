"""Gazebo contact monitoring via gz-transport (Garden+)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class GazeboContactMonitor:
    """Track contacts from Gazebo ``contacts`` topic or injected test fixtures."""

    _contacts: list[tuple[str, str]] = field(default_factory=list)
    _subscriber: object | None = field(default=None, repr=False)

    def bind_transport(self, node, *, topic: str = "contacts") -> None:
        """Subscribe to gz-transport contacts when available."""
        subscribe = getattr(node, "subscribe", None)
        if not callable(subscribe):
            return
        try:
            subscribe(topic, self._on_contacts)
            self._subscriber = node
        except Exception:
            return

    def inject_contacts(self, pairs: list[tuple[str, str]]) -> None:
        """Test helper: replace active contacts."""
        self._contacts = [(str(a), str(b)) for a, b in pairs]

    def _on_contacts(self, msg) -> None:
        pairs: list[tuple[str, str]] = []
        for contact in getattr(msg, "contact", ()) or ():
            a1 = self._entity_name(getattr(contact, "collision1", None) or getattr(contact, "entity1", None))
            a2 = self._entity_name(getattr(contact, "collision2", None) or getattr(contact, "entity2", None))
            if a1 and a2:
                pairs.append((a1, a2))
        self._contacts = pairs

    @staticmethod
    def _entity_name(entity) -> str:
        if entity is None:
            return ""
        for attr in ("name", "data", "id"):
            value = getattr(entity, attr, None)
            if isinstance(value, str) and value:
                return value
        return str(entity)

    def has_contact(self, body_a: str, body_b: str | None = None) -> bool:
        a = str(body_a)
        b = str(body_b) if body_b else None
        for left, right in self._contacts:
            if b is None and a in (left, right):
                return True
            if {left, right} == {a, b}:
                return True
        return False
