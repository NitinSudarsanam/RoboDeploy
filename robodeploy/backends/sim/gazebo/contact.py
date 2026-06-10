"""Gazebo contact monitoring via gz-transport (Garden+)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

logger = logging.getLogger(__name__)


@dataclass
class GazeboContactMonitor:
    """Track contacts from Gazebo ``contacts`` topic or injected test fixtures."""

    _contacts: list[tuple[str, str]] = field(default_factory=list)
    _subscriber: object | None = field(default=None, repr=False)

    def bind_transport(self, node, *, topic: str = "contacts") -> None:
        """Subscribe to gz-transport contacts when available."""
        subscribe = getattr(node, "subscribe", None)
        if not callable(subscribe):
            logger.debug("Gazebo contact monitor: transport node has no subscribe() for %s", topic)
            return
        try:
            subscribe(topic, self._on_contacts)
            self._subscriber = node
            logger.debug("Gazebo contact monitor subscribed to %s", topic)
        except Exception as exc:
            logger.debug("Gazebo contact monitor subscribe failed for %s: %s", topic, exc)
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

    @staticmethod
    def _contact_tokens(name: str) -> list[str]:
        normalized = str(name).replace("::", "/")
        parts = [p for p in normalized.split("/") if p]
        tokens: list[str] = []
        for part in parts:
            if part not in tokens:
                tokens.append(part)
        return tokens

    @classmethod
    def _matches(cls, name: str, pattern: str) -> bool:
        if not name or not pattern:
            return False
        if name == pattern:
            return True
        if pattern in name or name in pattern:
            return True
        name_tokens = cls._contact_tokens(name)
        pattern_tokens = cls._contact_tokens(pattern)
        if not pattern_tokens:
            return False
        pattern_tail = pattern_tokens[-1]
        if pattern_tail in name_tokens:
            return True
        if len(pattern_tokens) > 1 and all(tok in name_tokens for tok in pattern_tokens):
            return True
        return False

    def has_contact(self, body_a: str, body_b: str | None = None) -> bool:
        a = str(body_a)
        b = str(body_b) if body_b else None
        for left, right in self._contacts:
            if b is None:
                if self._matches(left, a) or self._matches(right, a):
                    return True
                continue
            if (self._matches(left, a) and self._matches(right, b)) or (
                self._matches(left, b) and self._matches(right, a)
            ):
                return True
        return False
