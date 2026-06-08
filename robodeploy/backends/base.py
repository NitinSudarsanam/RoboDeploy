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
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Optional
import warnings

from robodeploy.core.interfaces.backend import IBackend
from robodeploy.core.spaces import AssetFormat
from robodeploy.core.types import Action, AssetSelection, Observation, SceneSpec, SensorData

if TYPE_CHECKING:
    from robodeploy.core.interfaces.sensor import ISensor
    from robodeploy.core.robot              import Robot
    from robodeploy.description.base       import RobotDescription


class BackendBase(IBackend):
    """Shared scaffolding for all backends. Subclass this, not IBackend directly."""

    def __init__(self, config: dict | None = None) -> None:
        """
        Args:
            config: Backend-specific configuration dict. Each concrete backend
                    documents its accepted keys. Passed through to subclass
                    _setup() after lifecycle state is initialised.

        Compatibility:
            Many examples pass nested settings as ``{"config": {...}}``.
            For convenience, if ``config`` contains a mapping under the key
            ``"config"``, those entries are merged into the top-level config
            (nested values win on key collisions).
        """
        raw = dict(config or {})
        nested = raw.pop("config", None)
        merged: dict = {**raw}
        if isinstance(nested, dict):
            merged = {**merged, **nested}
        self.config: dict = merged
        self._initialized:    bool = False
        self._episode_count:  int  = 0
        self._step_count:     int  = 0
        self._asset_selections: dict[str, AssetSelection] = {}
        self._sensor_errors: dict[str, str] = {}
        self._sensor_error_warned: set[str] = set()
        self._pending_sensor_reads: list[tuple[str, "SensorData"]] = []

        # Set by initialize() — subclasses access these freely
        self._description:    RobotDescription | None = None
        self._scene:          SceneSpec | None        = None
        self._sensors:        list[ISensor]           = []

    # ------------------------------------------------------------------
    # IBackend lifecycle — adds guards around the abstract methods
    # ------------------------------------------------------------------

    def initialize(
        self,
        description: RobotDescription,
        scene:       SceneSpec,
        sensors:     list[ISensor],
    ) -> None:
        """Store references and call _load(), which subclasses implement."""
        self._description = description
        self._scene       = scene
        self._sensors     = sensors
        self._asset_selections.clear()
        self._load(description, scene, sensors)
        self._initialized = True

    # Multi-robot: require explicit backend support (no unsafe shims)
    def initialize_multi(
        self,
        robots: list["Robot"],
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

    def reset_multi(self, robot_ids: list[str] | None = None) -> list[Observation]:
        del robot_ids
        raise NotImplementedError(
            f"{type(self).__name__} does not implement reset_multi()."
        )

    def step(self, action: Action) -> Observation:
        """Guard + call _step_impl()."""
        self._require_initialized("step")
        obs = self._step_impl(action)
        self._step_count += 1
        return obs

    def step_multi(self, actions: list[Action]) -> list[Observation]:
        del actions
        raise NotImplementedError(
            f"{type(self).__name__} does not implement step_multi()."
        )

    def step_multi_batch(self, action_batches: list[list[Action]]) -> list[list[Observation]]:
        """Step multiple independent multi-robot action lists.

        Default implementation is sequential (honest fallback). Backends with true
        parallel batching can override for performance.
        """
        return [self.step_multi(actions) for actions in action_batches]

    def get_obs(self) -> Observation:
        """Guard + call _get_obs_impl()."""
        self._require_initialized("get_obs")
        return self._get_obs_impl()

    def get_obs_multi(self) -> list[Observation]:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement get_obs_multi()."
        )

    def _merge_sensor_data(self, obs: Observation, sensors: list["ISensor"]) -> Observation:
        """Merge sensor reads into a new Observation instance."""

        if not sensors:
            return obs

        self._pending_sensor_reads.clear()
        images = dict(getattr(obs, "images", {}) or {})
        depths = dict(getattr(obs, "depths", {}) or {})
        rgb = obs.rgb
        depth = obs.depth
        ft_force = obs.ft_force
        ft_torque = obs.ft_torque
        ft_forces = dict(getattr(obs, "ft_forces", {}) or {})
        ft_torques = dict(getattr(obs, "ft_torques", {}) or {})
        imu_acceleration = obs.imu_acceleration
        imu_angular_velocity = obs.imu_angular_velocity
        timestamp_hw = obs.timestamp_hw
        timestamp_recv = obs.timestamp_recv
        objects = dict(getattr(obs, "objects", {}) or {})
        contact_state = dict(getattr(obs, "contact_state", {}) or {})
        sensor_status = dict(getattr(obs, "sensor_status", {}) or {})
        camera_frames = dict(getattr(obs, "camera_frames", {}) or {})
        camera_intrinsics = dict(getattr(obs, "camera_intrinsics", {}) or {})
        camera_extrinsics = dict(getattr(obs, "camera_extrinsics", {}) or {})

        for sensor in sensors:
            name = str(getattr(sensor, "name", type(sensor).__name__))
            try:
                sd = sensor.read()
            except Exception as exc:
                self._record_sensor_error(name, exc)
                sensor_status[name] = "error"
                if str(self.config.get("sensor_read_policy", "warn")).lower() == "raise":
                    raise RuntimeError(f"Sensor '{name}' read failed.") from exc
                continue
            self._sensor_errors.pop(name, None)
            self._pending_sensor_reads.append((name, sd))
            sensor_status[name] = str(getattr(sd, "status", "ok"))
            if sd.rgb is not None:
                images[name] = sd.rgb
                if rgb is None:
                    rgb = sd.rgb
            if sd.depth is not None:
                depths[name] = sd.depth
                if depth is None:
                    depth = sd.depth
            ft_force = sd.ft_force if sd.ft_force is not None else ft_force
            ft_torque = sd.ft_torque if sd.ft_torque is not None else ft_torque
            if getattr(sd, "ft_forces", None):
                ft_forces.update(sd.ft_forces)
            elif sd.ft_force is not None:
                ft_forces[name] = sd.ft_force
            if getattr(sd, "ft_torques", None):
                ft_torques.update(sd.ft_torques)
            elif sd.ft_torque is not None:
                ft_torques[name] = sd.ft_torque
            imu_acceleration = sd.imu_acceleration if sd.imu_acceleration is not None else imu_acceleration
            imu_angular_velocity = sd.imu_angular_velocity if sd.imu_angular_velocity is not None else imu_angular_velocity
            if getattr(sd, "objects", None):
                objects.update(sd.objects)
            if getattr(sd, "contact_state", None):
                contact_state.update(sd.contact_state)
            if getattr(sd, "frame_id", None):
                camera_frames[name] = str(sd.frame_id)
            if getattr(sd, "intrinsics", None):
                camera_intrinsics[name] = dict(sd.intrinsics)
            if getattr(sd, "extrinsics", None):
                camera_extrinsics[name] = dict(sd.extrinsics)
            timestamp_hw = max(float(timestamp_hw), float(sd.timestamp_hw or 0.0))
            timestamp_recv = max(float(timestamp_recv), float(sd.timestamp_recv or 0.0))

        return replace(
            obs,
            rgb=rgb,
            depth=depth,
            images=images,
            depths=depths,
            ft_force=ft_force,
            ft_torque=ft_torque,
            ft_forces=ft_forces,
            ft_torques=ft_torques,
            imu_acceleration=imu_acceleration,
            imu_angular_velocity=imu_angular_velocity,
            objects=objects,
            contact_state=contact_state,
            sensor_status=sensor_status,
            camera_frames=camera_frames,
            camera_intrinsics=camera_intrinsics,
            camera_extrinsics=camera_extrinsics,
            timestamp_hw=timestamp_hw,
            timestamp_recv=timestamp_recv,
        )

    def drain_sensor_reads(self) -> list[tuple[str, SensorData]]:
        """Return and clear sensor reads from the latest merge for ObsPipeline buffering."""
        items = list(self._pending_sensor_reads)
        self._pending_sensor_reads.clear()
        return items

    def _record_sensor_error(self, sensor_name: str, exc: Exception) -> None:
        msg = f"{type(exc).__name__}: {exc}"
        self._sensor_errors[sensor_name] = msg
        if sensor_name not in self._sensor_error_warned:
            warnings.warn(
                f"{type(self).__name__} skipped sensor '{sensor_name}' after read failure: {msg}",
                RuntimeWarning,
                stacklevel=2,
            )
            self._sensor_error_warned.add(sensor_name)

    def _sensor_diagnostics(self) -> dict:
        return {
            "sensor_errors": dict(self._sensor_errors),
            "sensor_count": len(self._sensors),
        }

    def close(self) -> None:
        """Guard + call _close_impl()."""
        if self._initialized:
            for sensor in list(self._sensors):
                try:
                    sensor.close()
                except Exception:
                    pass
            self._close_impl()
            self._initialized = False

    # ------------------------------------------------------------------
    # Abstract internal methods — subclasses implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def _load(
        self,
        description: RobotDescription,
        scene:       SceneSpec,
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

    def seed(self, seed: int) -> None:
        """Default: record seed in config; subclasses may re-seed physics RNGs."""
        self.config["rng_seed"] = int(seed)

    def __repr__(self) -> str:
        status = "initialized" if self._initialized else "not initialized"
        robot  = self._description.display_name if self._description else "no robot"
        return f"{type(self).__name__}({robot}, {status})"
