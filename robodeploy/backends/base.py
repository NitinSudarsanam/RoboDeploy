"""
BackendBase — shared lifecycle scaffolding for all backends.

Every concrete backend (MuJoCoBackend, IsaacLabBackend, ROS2Backend, ...)
inherits from BackendBase rather than directly from IBackend.

BackendBase provides:
  - Lifecycle state tracking (_initialized, _episode_count, _step_count).
  - Guards that raise clear errors if methods are called out of order.
  - Default no-op implementations of optional IBackend methods.
  - A standard __repr__ for logging.

Concrete backends only implement the five abstract methods from IBackend
plus their backend-specific private helpers. They do not duplicate
the lifecycle boilerplate.

Extension point for batching:
  BatchedBackendBase (not yet implemented) will subclass BackendBase and
  override step()/get_obs() to accept and return arrays with a leading
  batch dimension [N, ...]. The five abstract methods remain the same;
  only the array shapes change. NativeBatchedMuJoCoBackend will then
  subclass BatchedBackendBase.
"""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from robodeploy.core.interfaces.backend import IBackend
from robodeploy.core.spaces import AssetFormat
from robodeploy.core.types import Action, AssetSelection, Observation

if TYPE_CHECKING:
    from robodeploy.core.interfaces.sensor import ISensor
    from robodeploy.core.interfaces.task   import ITask
    from robodeploy.description.base       import RobotDescription
    from robodeploy.core.robot_config      import RobotConfig


class BackendBase(IBackend):
    """Shared scaffolding for all backends. Subclass this, not IBackend directly."""

    def __init__(self, config: dict | None = None) -> None:
        """
        Args:
            config: Backend-specific configuration dict. Each concrete backend
                    documents its accepted keys. Passed through to subclass
                    _setup() after lifecycle state is initialised.
        """
        self.config:          dict = config or {}
        self._initialized:    bool = False
        self._episode_count:  int  = 0
        self._step_count:     int  = 0
        self._asset_selections: dict[str, AssetSelection] = {}

        # Set by initialize() — subclasses access these freely
        self._description:    RobotDescription | None = None
        self._task:           ITask | None            = None
        self._sensors:        list[ISensor]           = []

    # ------------------------------------------------------------------
    # IBackend lifecycle — adds guards around the abstract methods
    # ------------------------------------------------------------------

    def initialize(
        self,
        description: RobotDescription,
        task:        ITask,
        sensors:     list[ISensor],
    ) -> None:
        """Store references and call _load(), which subclasses implement."""
        self._description = description
        self._task        = task
        self._sensors     = sensors
        self._asset_selections.clear()
        self._load(description, task, sensors)
        self._initialized = True

    # Multi-robot: require explicit backend support (no unsafe shims)
    def initialize_multi(
        self,
        robots: list["RobotConfig"],
        scene,  # SceneSpec
        shared_sensors: list["ISensor"],
    ) -> None:
        del robots, scene, shared_sensors
        raise NotImplementedError(
            f"{type(self).__name__} does not implement initialize_multi(). "
            "Use single-agent RoboEnv, or implement multi-robot support in this backend."
        )

    @property
    def asset_selections(self) -> dict[str, AssetSelection]:
        """Asset selections made during initialize()."""
        return dict(self._asset_selections)

    def _resolve_asset_path(
        self,
        robot_id: str,
        description: "RobotDescription",
        fmt: AssetFormat,
        *,
        variant: str = "default",
        allow_override_wildcard: bool = True,
    ) -> Path:
        overrides = self.config.get("asset_overrides", {}) or {}
        fmt_key = fmt.value

        override_path: Optional[str] = None
        if isinstance(overrides, dict):
            if robot_id in overrides and isinstance(overrides[robot_id], dict):
                override_path = overrides[robot_id].get(fmt_key)
            if override_path is None and allow_override_wildcard and "*" in overrides and isinstance(overrides["*"], dict):
                override_path = overrides["*"].get(fmt_key)

        if override_path:
            p = Path(override_path).expanduser()
            self._asset_selections[robot_id] = AssetSelection(
                robot_id=robot_id,
                requested_format=fmt,
                used_format=fmt,
                resolved_path=str(p),
                source="override",
                notes="asset_overrides",
            )
            return p

        p = description.asset_path(fmt, variant=variant)
        self._asset_selections[robot_id] = AssetSelection(
            robot_id=robot_id,
            requested_format=fmt,
            used_format=fmt,
            resolved_path=str(p),
            source="description",
            notes=f"variant={variant}",
        )
        return p

    def reset(self) -> Observation:
        """Guard + call _reset_impl()."""
        self._require_initialized("reset")
        obs = self._reset_impl()
        self._episode_count += 1
        self._step_count = 0
        return obs

    # Multi-robot shim: default maps to single-robot reset
    def reset_multi(self, robot_ids: list[str] | None = None) -> list[Observation]:
        return [self.reset()]

    def step(self, action: Action) -> Observation:
        """Guard + call _step_impl()."""
        self._require_initialized("step")
        obs = self._step_impl(action)
        self._step_count += 1
        return obs

    # Multi-robot shim: default maps first action to single-robot step
    def step_multi(self, actions: list[Action]) -> list[Observation]:
        if not actions:
            raise ValueError("step_multi() requires at least one Action.")
        return [self.step(actions[0])]

    def get_obs(self) -> Observation:
        """Guard + call _get_obs_impl()."""
        self._require_initialized("get_obs")
        return self._get_obs_impl()

    # Multi-robot shim: default maps to single-robot get_obs
    def get_obs_multi(self) -> list[Observation]:
        return [self.get_obs()]

    def close(self) -> None:
        """Guard + call _close_impl()."""
        if self._initialized:
            self._close_impl()
            self._initialized = False

    # ------------------------------------------------------------------
    # Abstract internal methods — subclasses implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def _load(
        self,
        description: RobotDescription,
        task:        ITask,
        sensors:     list[ISensor],
    ) -> None:
        """Load the robot model and scene assets. Called once by initialize()."""
        ...

    @abstractmethod
    def _reset_impl(self) -> Observation:
        """Reset physics/hardware and return initial observation."""
        ...

    @abstractmethod
    def _step_impl(self, action: Action) -> Observation:
        """Apply action, advance one control step, return observation."""
        ...

    @abstractmethod
    def _get_obs_impl(self) -> Observation:
        """Return current observation without advancing physics."""
        ...

    @abstractmethod
    def _close_impl(self) -> None:
        """Release all backend resources."""
        ...

    # ------------------------------------------------------------------
    # Properties — subclasses must still declare these (from IBackend)
    # ------------------------------------------------------------------

    # is_real, supported_action_spaces, control_hz remain abstract here.
    # Each concrete backend declares them as class-level or instance properties.

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _require_initialized(self, method: str) -> None:
        if not self._initialized:
            raise RuntimeError(
                f"{type(self).__name__}.{method}() called before initialize(). "
                "Call env.reset() or backend.initialize() first."
            )

    @property
    def episode_count(self) -> int:
        """Number of episodes completed since initialize()."""
        return self._episode_count

    @property
    def step_count(self) -> int:
        """Number of steps taken in the current episode."""
        return self._step_count

    def __repr__(self) -> str:
        status = "initialized" if self._initialized else "not initialized"
        robot  = self._description.display_name if self._description else "no robot"
        return f"{type(self).__name__}({robot}, {status})"
