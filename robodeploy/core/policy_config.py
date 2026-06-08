"""Validated policy configuration schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from robodeploy.core.spaces import ActionSpace

_VALID_CARRY = frozenset({"kinematic", "follow", "contact", "weld", "none"})


@dataclass
class PolicyConfig:
    action_space: ActionSpace = ActionSpace.JOINT_POS
    action_hz: float = 50.0
    carry_mode: str = "kinematic"
    home_qpos: list[float] = field(default_factory=lambda: [0.0, -0.6, 0.0, -1.8, 0.0, 1.2, 0.0])
    tracking_blend: float = 0.22
    settle_threshold: float = 0.025
    steps_per_phase: int = 180
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.action_hz <= 0.0:
            raise ValueError("action_hz must be positive.")
        mode = str(self.carry_mode).lower()
        if mode not in _VALID_CARRY:
            raise ValueError(f"carry_mode must be one of {sorted(_VALID_CARRY)}, got '{self.carry_mode}'.")
        self.carry_mode = mode
        if self.steps_per_phase <= 0:
            raise ValueError("steps_per_phase must be positive.")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PolicyConfig":
        space_name = str(data.get("action_space", "JOINT_POS"))
        return cls(
            action_space=ActionSpace[space_name] if space_name in ActionSpace.__members__ else ActionSpace.JOINT_POS,
            action_hz=float(data.get("action_hz", 50.0)),
            carry_mode=str(data.get("carry_mode", "kinematic")),
            home_qpos=list(data.get("home_qpos", data.get("home", [])) or [0.0, -0.6, 0.0, -1.8, 0.0, 1.2, 0.0]),
            tracking_blend=float(data.get("tracking_blend", 0.22)),
            settle_threshold=float(data.get("settle_threshold", 0.025)),
            steps_per_phase=int(data.get("steps_per_phase", 180)),
            extra={k: v for k, v in data.items() if k not in cls.__dataclass_fields__},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_space": self.action_space.name,
            "action_hz": self.action_hz,
            "carry_mode": self.carry_mode,
            "home_qpos": list(self.home_qpos),
            "tracking_blend": self.tracking_blend,
            "settle_threshold": self.settle_threshold,
            "steps_per_phase": self.steps_per_phase,
            **self.extra,
        }
