"""
TaskBase — shared scaffolding for all tasks.

Every concrete task (PickPlace, Pour, PegInsertion, ...) inherits from
TaskBase rather than ITask directly.

TaskBase provides:
  - Storage for the SceneSpec and ObsSpec declared at construction time.
  - A step counter incremented by RoboEnv (exposed for use in reward_fn).
  - Default failure_fn (returns False — no time-based failure by default).
  - Default max_steps (1000).
  - Standard __repr__.

Concrete tasks override:
  - scene_spec()              — declare what objects exist.
  - obs_spec()                — declare what sensors are needed.
  - language_instruction()    — the goal string.
  - reset_fn(backend)         — randomise and reposition scene objects.
  - reward_fn(obs, action)    — scalar reward computation.
  - success_fn(obs)           — termination condition.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from robodeploy.backends.capabilities import SupportsSceneEdit
from robodeploy.core.interfaces.task import ITask
from robodeploy.core.types import Action, ObsSpec, Observation, SceneSpec
from robodeploy.tasks.randomization import (
    DomainRandomizer,
    DomainRandomizerConfig,
    ObjectRandomConfig,
    RandomLevel,
)

if TYPE_CHECKING:
    from robodeploy.core.interfaces.backend import IBackend


class TaskBase(ITask):
    """Shared scaffolding for all tasks. Subclass this, not ITask directly."""

    def __init__(self, config: dict | None = None) -> None:
        """
        Args:
            config: Task-specific configuration (object names, pose ranges,
                    reward weights, success thresholds, etc.).
        """
        self.config:       dict = config or {}
        self._step_count:  int  = 0
        self._episode:     int  = 0
        self._backend = None

    # ------------------------------------------------------------------
    # ITask abstract methods — subclasses must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def obs_spec(self) -> ObsSpec:
        """Declare required sensors. See ITask.obs_spec() for full contract."""
        ...

    @abstractmethod
    def scene_spec(self) -> SceneSpec:
        """Declare scene objects and initial poses. See ITask.scene_spec()."""
        ...

    @abstractmethod
    def language_instruction(self) -> str:
        """Natural language task goal. See ITask.language_instruction()."""
        ...

    @abstractmethod
    def reset_fn(self, backend: IBackend) -> None:
        """Randomise scene for a new episode. See ITask.reset_fn()."""
        ...

    @abstractmethod
    def reward_fn(self, obs: Observation, action: Action) -> float:
        """Scalar reward for current step. See ITask.reward_fn()."""
        ...

    @abstractmethod
    def success_fn(self, obs: Observation) -> bool:
        """True when task is successfully completed. See ITask.success_fn()."""
        ...

    # ------------------------------------------------------------------
    # ITask optional overrides with sensible defaults
    # ------------------------------------------------------------------

    def failure_fn(self, obs: Observation) -> bool:
        """Default: never fail early. Override for tasks with failure conditions.

        Example overrides:
          - Object fell off table (ee_position.z < floor_z).
          - Joint limit exceeded (checked by SafetyFilter, but tasks can add more).
          - Episode timeout (checked via step_count >= max_steps() in RoboEnv).
        """
        return False

    def max_steps(self) -> int:
        """Default episode length. Override per task.

        Returns:
            int: Maximum steps. RoboEnv calls failure_fn after this many steps.
        """
        return self.config.get("max_steps", 1000)

    # ------------------------------------------------------------------
    # Step counter — managed by RoboEnv, readable in reward_fn / success_fn
    # ------------------------------------------------------------------

    def _on_reset(self) -> None:
        """Called by RoboEnv at the start of each episode. Do not call directly."""
        self._step_count = 0
        self._episode   += 1

    def _on_step(self) -> None:
        """Called by RoboEnv after each step. Do not call directly."""
        self._step_count += 1

    def on_activate(self) -> None:
        """Hook called when this task becomes active for a robot."""
        return

    def on_deactivate(self) -> None:
        """Hook called when this task stops being active for a robot."""
        return

    @property
    def step_count(self) -> int:
        """Steps elapsed in the current episode."""
        return self._step_count

    @property
    def episode(self) -> int:
        """Number of episodes completed."""
        return self._episode

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"episode={self._episode}, "
            f"step={self._step_count})"
        )

    # ------------------------------------------------------------------
    # Backend-aware helpers for object-centric tasks
    # ------------------------------------------------------------------

    def _bind_backend(self, backend: "IBackend") -> None:
        self._backend = backend

    @property
    def backend(self):
        return self._backend

    def scene_prop(self, name: str):
        for prop in self.scene_spec().to_world().props:
            if prop.name == name:
                return prop
        return None

    def prop_pose(self, name: str):
        backend = self._backend
        getter = getattr(backend, "get_prop_pose", None)
        if backend is None:
            return None
        if isinstance(backend, SupportsSceneEdit) or callable(getter):
            try:
                return getter(name)
            except Exception:
                return None
        return None

    def _domain_randomizer(self) -> DomainRandomizer | None:
        """Build a randomizer from task config, if enabled."""
        dr_cfg = self.config.get("domain_randomization")
        if dr_cfg is False:
            return None
        if isinstance(dr_cfg, DomainRandomizer):
            return dr_cfg
        if isinstance(dr_cfg, DomainRandomizerConfig):
            return DomainRandomizer(dr_cfg)
        if isinstance(dr_cfg, dict):
            level_name = str(dr_cfg.get("level", "light")).upper()
            level = RandomLevel[level_name] if level_name in RandomLevel.__members__ else RandomLevel.LIGHT
            objects = [
                ObjectRandomConfig(**item) if isinstance(item, dict) else item
                for item in dr_cfg.get("objects", self._default_object_random_configs())
            ]
            return DomainRandomizer(
                DomainRandomizerConfig(
                    level=level,
                    seed=dr_cfg.get("seed"),
                    objects=objects,
                )
            )
        if self.config.get("randomize_objects"):
            return DomainRandomizer(
                DomainRandomizerConfig(
                    level=RandomLevel.LIGHT,
                    seed=self.config.get("random_seed"),
                    objects=self._default_object_random_configs(),
                )
            )
        return None

    def _default_object_random_configs(self) -> list[ObjectRandomConfig]:
        jitter = float(self.config.get("pose_jitter_m", 0.03))
        configs: list[ObjectRandomConfig] = []
        for prop in self.scene_spec().to_world().props:
            if prop.is_fixed:
                continue
            configs.append(
                ObjectRandomConfig(
                    object_name=prop.name,
                    position_center=prop.position,
                    position_range=(jitter, jitter, 0.0),
                )
            )
        return configs

    def _apply_domain_randomization(self, backend: "IBackend") -> None:
        randomizer = self._domain_randomizer()
        if randomizer is not None:
            randomizer.randomize(backend)
