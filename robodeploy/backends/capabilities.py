"""Optional backend capability protocols.

Backends should implement only the capabilities they actually support.
RoboEnv depends on these via duck typing to keep the public API stable.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class SupportsVizSink(Protocol):
    """Backend accepts a backend-agnostic visualization payload."""

    def set_viz_payload(self, payload: Optional[dict]) -> None: ...


@runtime_checkable
class SupportsDiagnostics(Protocol):
    """Backend exposes lightweight runtime diagnostics."""

    def get_diagnostics(self) -> dict: ...


@runtime_checkable
class SupportsMultiRobot(Protocol):
    """Marker protocol for backends that explicitly support multi-robot APIs."""

    def initialize_multi(self, robots, scene, shared_sensors) -> None: ...
    def reset_multi(self, robot_ids: list[str] | None = None) -> list: ...
    def step_multi(self, actions: list) -> list: ...
    def get_obs_multi(self) -> list: ...


@runtime_checkable
class SupportsSceneEdit(Protocol):
    """Backend supports editing/querying scene props at runtime."""

    def set_prop_pose(self, name: str, position, orientation) -> None: ...  # noqa: ANN001
    def get_prop_pose(self, name: str): ...  # noqa: ANN001
    def get_prop_names(self) -> list[str]: ...


@runtime_checkable
class SupportsPayload(Protocol):
    """Backend supports setting dynamic payload parameters (mass/CoM)."""

    def set_payload(self, robot_id: str, *, mass: float, com) -> None: ...  # noqa: ANN001


@runtime_checkable
class SupportsPhysicsRandomization(Protocol):
    """Backend supports physics parameter randomization at runtime."""

    def set_physics_params(self, **kwargs) -> None: ...
