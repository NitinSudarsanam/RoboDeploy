"""Fluent builder for EnvConfig."""

from __future__ import annotations

from typing import Any

from robodeploy.core.env_config import EnvConfig
from robodeploy.core.task_config import TaskConfig


class EnvConfigBuilder:
    def __init__(self) -> None:
        self._robot: str | Any | None = None
        self._backend: str | Any | None = None
        self._task: str | Any | None = None
        self._policy: str | Any | None = None
        self._sensors: list[str] = []
        self._sensor_kwargs: dict[str, dict[str, Any]] = {}
        self._sensor_rigs: list[Any] = []
        self._backend_kwargs: dict[str, Any] = {}
        self._task_kwargs: dict[str, Any] = {}
        self._policy_kwargs: dict[str, Any] = {}
        self._custom_modules: list[str] = []
        self._obs_spec_policy: str = "warn"
        self._max_episode_steps: int | None = None
        self._obs_pipeline = None

    def with_robot(self, robot: str, **kwargs: Any) -> EnvConfigBuilder:
        self._robot = robot
        if kwargs:
            self._task_kwargs.setdefault("robot_kwargs", {})
        return self

    def with_backend(self, backend: str, **kwargs: Any) -> EnvConfigBuilder:
        self._backend = backend
        self._backend_kwargs.update(kwargs)
        return self

    def with_task(self, task: str, **kwargs: Any) -> EnvConfigBuilder:
        self._task = task
        self._task_kwargs.update(kwargs)
        return self

    def with_policy(self, policy: str, **kwargs: Any) -> EnvConfigBuilder:
        self._policy = policy
        self._policy_kwargs.update(kwargs)
        return self

    def with_task_config(self, config: TaskConfig) -> EnvConfigBuilder:
        self._task_kwargs.update(config.to_task_kwargs())
        return self

    def add_sensor(self, name: str, **kwargs: Any) -> EnvConfigBuilder:
        self._sensors.append(name)
        if kwargs:
            self._sensor_kwargs[name] = kwargs
        return self

    def add_sensor_rig(self, rig: Any) -> EnvConfigBuilder:
        self._sensor_rigs.append(rig)
        return self

    def with_custom_modules(self, *modules: str) -> EnvConfigBuilder:
        self._custom_modules.extend(modules)
        return self

    def with_obs_pipeline(self, pipeline) -> EnvConfigBuilder:  # noqa: ANN001
        self._obs_pipeline = pipeline
        return self

    def with_max_episode_steps(self, steps: int) -> EnvConfigBuilder:
        self._max_episode_steps = int(steps)
        return self

    def validate(self) -> EnvConfigBuilder:
        missing = [name for name, val in [
            ("robot", self._robot),
            ("backend", self._backend),
            ("task", self._task),
            ("policy", self._policy),
        ] if val is None]
        if missing:
            raise ValueError(f"EnvConfigBuilder missing required fields: {', '.join(missing)}")
        return self

    def build(self) -> EnvConfig:
        self.validate()
        return EnvConfig(
            robot=self._robot,
            backend=self._backend,
            task=self._task,
            policy=self._policy,
            sensors=list(self._sensors),
            sensor_rigs=list(self._sensor_rigs) or None,
            obs_pipeline=self._obs_pipeline,
            backend_kwargs=dict(self._backend_kwargs) or None,
            task_kwargs=dict(self._task_kwargs) or None,
            policy_kwargs=dict(self._policy_kwargs) or None,
            sensor_kwargs=dict(self._sensor_kwargs) or None,
            custom_modules=list(self._custom_modules),
            obs_spec_policy=self._obs_spec_policy,
            max_episode_steps=self._max_episode_steps,
        )
