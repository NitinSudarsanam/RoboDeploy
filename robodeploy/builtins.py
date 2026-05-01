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
    # Tasks
    "robodeploy.tasks.manipulation.pick_place",
    "robodeploy.tasks.manipulation.pour",
    "robodeploy.tasks.manipulation.peg_insertion",
    # Policies
    "robodeploy.policies.learned.robomimic",
    "robodeploy.policies.learned.diffusion",
    "robodeploy.policies.learned.vla",
    "robodeploy.policies.scripted.waypoint",
    "robodeploy.policies.scripted.joint_pd",
)


def import_builtins(modules: Iterable[str] = _BUILTIN_MODULES) -> None:
    """Best-effort import of builtin component modules.

    Only `ImportError` is suppressed to keep optional dependencies optional.
    Any other exception indicates a real bug and should be surfaced.
    """

    for mod in modules:
        try:
            import_module(mod)
        except ImportError:
            continue

