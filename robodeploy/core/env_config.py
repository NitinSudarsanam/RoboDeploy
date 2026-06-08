"""Validated environment configuration schema."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

from robodeploy.core.task_config import TaskConfig


@dataclass
class EnvConfig:
    robot: str | Any
    backend: str | Any
    task: str | Any
    policy: str | Any
    sensors: list[str | Any] = field(default_factory=list)
    sensor_rigs: list[Any] | None = None
    obs_pipeline: Any | None = None
    backend_kwargs: dict[str, Any] | None = None
    task_kwargs: dict[str, Any] | None = None
    policy_kwargs: dict[str, Any] | None = None
    robot_kwargs: dict[str, Any] | None = None
    sensor_kwargs: dict[str, Any] | None = None
    custom_modules: list[str] = field(default_factory=list)
    obs_spec_policy: Literal["warn", "raise", "off"] = "warn"
    max_episode_steps: int | None = None
    robot_id: str = "robot0"
    task_id: str = "task0"
    policy_id: str = "policy0"

    def __post_init__(self) -> None:
        policy = str(self.obs_spec_policy).lower()
        if policy not in ("warn", "raise", "off"):
            raise ValueError("obs_spec_policy must be 'warn', 'raise', or 'off'.")
        self.obs_spec_policy = policy  # type: ignore[assignment]
        if self.max_episode_steps is not None and self.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive when set.")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EnvConfig":
        task_cfg = data.get("task_config")
        task_kwargs = dict(data.get("task_kwargs") or {})
        if isinstance(task_cfg, dict):
            task_kwargs.update(TaskConfig.from_dict(task_cfg).to_task_kwargs())
        elif isinstance(task_cfg, TaskConfig):
            task_kwargs.update(task_cfg.to_task_kwargs())
        return cls(
            robot=data["robot"],
            backend=data["backend"],
            task=data["task"],
            policy=data["policy"],
            sensors=list(data.get("sensors") or []),
            sensor_rigs=data.get("sensor_rigs"),
            obs_pipeline=data.get("obs_pipeline"),
            backend_kwargs=data.get("backend_kwargs"),
            task_kwargs=task_kwargs or None,
            policy_kwargs=data.get("policy_kwargs"),
            robot_kwargs=data.get("robot_kwargs"),
            sensor_kwargs=data.get("sensor_kwargs"),
            custom_modules=list(data.get("custom_modules") or []),
            obs_spec_policy=data.get("obs_spec_policy", "warn"),
            max_episode_steps=data.get("max_episode_steps"),
            robot_id=str(data.get("robot_id", "robot0")),
            task_id=str(data.get("task_id", "task0")),
            policy_id=str(data.get("policy_id", "policy0")),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EnvConfig":
        if yaml is None:
            raise ImportError("PyYAML is required for EnvConfig.from_yaml().")
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid env config YAML: {path}")
        return cls.from_dict(raw)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        if yaml is None:
            raise ImportError("PyYAML is required for EnvConfig.to_yaml().")
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")

    def resolve(self) -> dict[str, Any]:
        """Return fully materialized config dict (preset inheritance applied)."""
        return dict(self.to_dict())
