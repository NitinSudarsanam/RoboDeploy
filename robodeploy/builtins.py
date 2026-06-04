"""Built-in component registration helpers.

This module exists to preserve strict layering:
- `robodeploy.core.*` is the contract and must not import concrete backends.
- Built-in backends/robots/tasks/policies are optional and can be imported lazily
  from non-core code paths (e.g., `RoboEnv.make()`).
"""

from __future__ import annotations

from importlib import import_module
from typing import Iterable


_BUILTIN_MODULES: tuple[str, ...] = (
    # Backends
    "robodeploy.backends.sim.mujoco.backend",
    "robodeploy.backends.sim.isaacsim.backend",
    "robodeploy.backends.real.ros2.backend",
    "robodeploy.backends.sim.gazebo.backend",
    # Robots
    "robodeploy.description.franka.description",
    "robodeploy.description.kuka.description",
    "robodeploy.description.so101.description",
    # Sensors
    "robodeploy.sensors.camera.sim.mujoco_camera",
    "robodeploy.sensors.camera.real.realsense",
    "robodeploy.sensors.ft_sensor.sim.mujoco_ft",
    "robodeploy.sensors.ft_sensor.real.ati_ft",
    "robodeploy.backends.real.ros2.sensors.camera_rgbd",
    # Policies
    "robodeploy.policies.learned.robomimic",
    "robodeploy.policies.learned.diffusion",
    "robodeploy.policies.learned.vla",
    "robodeploy.policies.composition",
)


def import_builtins(modules: Iterable[str] = _BUILTIN_MODULES) -> None:
    """Best-effort import of builtin component modules.

    Only `ImportError` is suppressed to keep optional dependencies optional.
    Any other exception indicates a real bug and should be surfaced.
    """
    failed_builtin_imports(modules)


def failed_builtin_imports(modules: Iterable[str] = _BUILTIN_MODULES) -> list[str]:
    """Return ImportError messages for builtin modules that could not be loaded."""
    failures: list[str] = []
    for mod in modules:
        try:
            import_module(mod)
        except ImportError as exc:
            failures.append(f"{mod}: {exc}")
    return failures

