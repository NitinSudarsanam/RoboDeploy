"""Validated task configuration schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from robodeploy.core.types import ObsSpec


@dataclass
class TaskConfig:
    scene: dict[str, Any] | None = None
    obs_spec: ObsSpec | None = None
    domain_randomization: dict[str, Any] | bool | None = None
    reward_weights: dict[str, float] = field(default_factory=dict)
    success_threshold: float = 0.04
    language_instruction: str | None = None
    require_objects: bool = False
    max_steps: int = 1000
    obs_spec_policy: Literal["warn", "raise", "off"] = "warn"
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.success_threshold <= 0.0:
            raise ValueError("success_threshold must be positive.")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive.")
        policy = str(self.obs_spec_policy).lower()
        if policy not in ("warn", "raise", "off"):
            raise ValueError("obs_spec_policy must be 'warn', 'raise', or 'off'.")
        self.obs_spec_policy = policy  # type: ignore[assignment]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskConfig":
        obs = data.get("obs_spec")
        obs_spec = None
        if isinstance(obs, ObsSpec):
            obs_spec = obs
        elif isinstance(obs, dict):
            obs_spec = ObsSpec(**obs)
        return cls(
            scene=data.get("scene"),
            obs_spec=obs_spec,
            domain_randomization=data.get("domain_randomization"),
            reward_weights=dict(data.get("reward_weights", {})),
            success_threshold=float(data.get("success_threshold", 0.04)),
            language_instruction=data.get("language_instruction"),
            require_objects=bool(data.get("require_objects", False)),
            max_steps=int(data.get("max_steps", 1000)),
            obs_spec_policy=data.get("obs_spec_policy", "warn"),
            extra={k: v for k, v in data.items() if k not in cls.__dataclass_fields__},
        )

    def to_task_kwargs(self) -> dict[str, Any]:
        out = dict(self.extra)
        out.update(
            {
                "require_objects": self.require_objects,
                "success_threshold": self.success_threshold,
                "max_steps": self.max_steps,
                "reward_weights": dict(self.reward_weights),
            }
        )
        if self.domain_randomization is not None:
            out["domain_randomization"] = self.domain_randomization
        return out
