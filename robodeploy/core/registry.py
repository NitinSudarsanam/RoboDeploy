"""
Registry — optional plugin system for string-based component lookup.

IMPORTANT: The registry is a convenience layer. Direct object injection
(passing your classes directly to RoboEnv()) is always preferred. Use the
registry only when you need config-driven component swapping via strings.

Three usage levels — choose the one that fits your project:

  Level 1 — Direct injection (recommended for most users):
      No registry needed. Pass objects directly to RoboEnv().
      See env.py for examples.

  Level 2 — use() + make() (for config-driven swapping):
      Define your components anywhere in your own project, call use() to
      register them, then call RoboEnv.make() with strings.

      # my_project/components.py
      from robodeploy.core.registry import register_robot

      @register_robot("myrobot")
      class MyRobotDescription(RobotDescription): ...

      # my_project/run.py
      from robodeploy import use, RoboEnv

      use("my_project.components")           # imports module → triggers decorators
      env = RoboEnv.make(robot="myrobot", backend="mujoco", task="my_task")

  Level 3 — Entry points (for pip-installable robot/task packages):
      Declare entry points in pyproject.toml. RoboDeploy auto-discovers them.
      See ARCHITECTURE.md for the pyproject.toml format.

Note: Components are registered when their module is first imported. If you
call make() before importing your component module, you get a KeyError. The
use() function exists to make that import explicit and readable.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Type, TypeVar

from robodeploy.core.types import SensorMount

T = TypeVar("T")

# Internal stores: name → class
_BACKENDS:  dict[str, type] = {}
_ROBOTS:    dict[str, type] = {}
_POLICIES:  dict[str, type] = {}
_TASKS:     dict[str, type] = {}
_SENSORS:   dict[str, type] = {}
_SENSOR_PAIRS: dict[str, "SensorPairSpec"] = {}
_ENTRY_POINT_DISCOVERY = False


def _put(store: dict[str, type], kind: str, name: str, cls: Type[T]) -> Type[T]:
    existing = store.get(name)
    if existing is cls:
        return cls
    if existing is not None:
        existing_ref = f"{existing.__module__}.{existing.__qualname__}"
        incoming_ref = f"{cls.__module__}.{cls.__qualname__}"
        if _ENTRY_POINT_DISCOVERY:
            warnings.warn(
                f"{kind} '{name}' already registered as {existing_ref}; "
                f"entry-point plugin overriding with {incoming_ref}.",
                stacklevel=4,
            )
            store[name] = cls
            return cls
        raise KeyError(
            f"{kind} '{name}' is already registered as {existing_ref}. "
            f"Attempted duplicate from {incoming_ref}."
        )
    store[name] = cls
    return cls


@dataclass
class SensorPairSpec:
    """Explicit sim/real pairing for a user-facing sensor name."""

    sim: type | None = None
    real: type | None = None
    by_backend: dict[str, type | None] = field(default_factory=dict)
    default_mount: SensorMount = field(default_factory=SensorMount)
    default_config: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Registration decorators
# ---------------------------------------------------------------------------

def register_backend(name: str):
    """Decorator: register a backend class under a string name.

    Args:
        name: Lookup key (e.g. "mujoco", "isaaclab", "ros2").

    Raises:
        KeyError: If a backend with this name is already registered.
    """
    def decorator(cls: Type[T]) -> Type[T]:
        return _put(_BACKENDS, "Backend", name, cls)
    return decorator


def register_robot(name: str):
    """Decorator: register a RobotDescription class under a string name.

    Args:
        name: Lookup key (e.g. "franka", "ur5", "spot").
    """
    def decorator(cls: Type[T]) -> Type[T]:
        return _put(_ROBOTS, "Robot", name, cls)
    return decorator


def register_policy(name: str):
    """Decorator: register a policy class under a string name.

    Args:
        name: Lookup key (e.g. "robomimic", "diffusion", "vla", "joint_pd").
    """
    def decorator(cls: Type[T]) -> Type[T]:
        return _put(_POLICIES, "Policy", name, cls)
    return decorator


def register_task(name: str):
    """Decorator: register a task class under a string name.

    Args:
        name: Lookup key (e.g. "my_task"; example repo tasks live under examples.tasks).
    """
    def decorator(cls: Type[T]) -> Type[T]:
        return _put(_TASKS, "Task", name, cls)
    return decorator


def register_sensor(name: str):
    """Decorator: register a sensor class under a string name.

    Args:
        name: Lookup key (e.g. "wrist_camera", "realsense", "ati_ft").
    """
    def decorator(cls: Type[T]) -> Type[T]:
        return _put(_SENSORS, "Sensor", name, cls)
    return decorator


def register_sensor_pair(
    name: str,
    *,
    sim: type | None = None,
    real: type | None = None,
    by_backend: dict[str, type | None] | None = None,
    default_mount: SensorMount | None = None,
    default_config: dict | None = None,
):
    """Decorator: register explicit sim/real sensor pairing metadata."""

    def decorator(cls: Type[T]) -> Type[T]:
        spec = SensorPairSpec(
            sim=sim or getattr(cls, "sim", None),
            real=real or getattr(cls, "real", None),
            by_backend=dict(by_backend or getattr(cls, "by_backend", {}) or {}),
            default_mount=default_mount or getattr(cls, "default_mount", SensorMount()),
            default_config=dict(default_config or getattr(cls, "default_config", {}) or {}),
        )
        if name in _SENSOR_PAIRS:
            existing = _SENSOR_PAIRS[name]
            merged_backends = dict(existing.by_backend)
            for backend_name, backend_cls in spec.by_backend.items():
                if backend_cls is not None:
                    merged_backends[backend_name] = backend_cls
            _SENSOR_PAIRS[name] = SensorPairSpec(
                sim=spec.sim or existing.sim,
                real=spec.real or existing.real,
                by_backend=merged_backends,
                default_mount=spec.default_mount or existing.default_mount,
                default_config={**existing.default_config, **spec.default_config},
            )
        else:
            _SENSOR_PAIRS[name] = spec
        return cls
    return decorator


# ---------------------------------------------------------------------------
# Lookup functions
# ---------------------------------------------------------------------------

def get_backend(name: str) -> type:
    """Look up a registered backend class by name.

    Args:
        name: Registered backend name.

    Returns:
        The backend class (not an instance — you instantiate it).

    Raises:
        KeyError: If no backend with this name is registered.
                  Message includes all registered names to aid debugging.
    """
    if name not in _BACKENDS:
        raise KeyError(
            f"Backend '{name}' not found. Registered: {list(_BACKENDS)}\n"
            "Did you forget to import the backend module?"
        )
    return _BACKENDS[name]


def get_robot(name: str) -> type:
    """Look up a registered RobotDescription class by name."""
    if name not in _ROBOTS:
        raise KeyError(
            f"Robot '{name}' not found. Registered: {list(_ROBOTS)}\n"
            "Did you forget to import the description module?"
        )
    return _ROBOTS[name]


def get_policy(name: str) -> type:
    """Look up a registered policy class by name."""
    if name not in _POLICIES:
        raise KeyError(
            f"Policy '{name}' not found. Registered: {list(_POLICIES)}\n"
            "Did you forget to import the policy module?"
        )
    return _POLICIES[name]


def get_task(name: str) -> type:
    """Look up a registered task class by name."""
    if name not in _TASKS:
        raise KeyError(
            f"Task '{name}' not found. Registered: {list(_TASKS)}\n"
            "Did you forget to import the task module?"
        )
    return _TASKS[name]


def get_sensor(name: str) -> type:
    """Look up a registered sensor class by name."""
    if name not in _SENSORS:
        raise KeyError(
            f"Sensor '{name}' not found. Registered: {list(_SENSORS)}\n"
            "Did you forget to import the sensor module?"
        )
    return _SENSORS[name]


def get_sensor_pair(name: str) -> SensorPairSpec:
    """Look up explicit sim/real pairing metadata for a sensor name."""

    if name not in _SENSOR_PAIRS:
        raise KeyError(
            f"Sensor pair '{name}' not found. Registered: {list(_SENSOR_PAIRS)}"
        )
    return _SENSOR_PAIRS[name]


def _ensure_sensor_class_type(cls: object, *, context: str) -> type:
    """Reject corrupted pair entries (e.g. a string placeholder) before instantiation."""
    if isinstance(cls, type):
        return cls
    raise TypeError(
        f"{context}: expected a sensor class type, got {type(cls).__name__!r} ({cls!r}). "
        "Check register_sensor_pair by_backend entries."
    )


def normalize_sensor_backend_name(name: str | None) -> str | None:
    if not name:
        return None
    raw = str(name).strip().lower()
    aliases = {
        "real_world": "ros2",
        "ros2_gazebo": "gazebo",
    }
    return aliases.get(raw, raw)


def resolve_sensor_class(name: str, *, is_real: bool, backend_name: str | None = None) -> type:
    """Resolve a user-facing sensor name to a concrete class.

    The explicit pair registry is preferred. The legacy `_sim` / `_real`
    suffix convention remains as a fallback during migration.
    """

    pair = _SENSOR_PAIRS.get(name)
    normalized_backend = normalize_sensor_backend_name(backend_name)
    if pair is not None:
        if normalized_backend is not None and normalized_backend in pair.by_backend:
            cls = pair.by_backend[normalized_backend]
            if cls is None:
                raise KeyError(
                    f"Sensor pair '{name}' has no implementation for backend '{normalized_backend}'."
                )
            return _ensure_sensor_class_type(
                cls,
                context=f"Sensor pair '{name}' backend '{normalized_backend}'",
            )
        cls = pair.real if is_real else pair.sim
        if cls is None:
            side = "real" if is_real else "sim"
            detail = f" backend '{normalized_backend}'" if normalized_backend is not None else ""
            raise KeyError(f"Sensor pair '{name}' has no {side} implementation for{detail}.")
        return _ensure_sensor_class_type(
            cls,
            context=f"Sensor pair '{name}' ({'real' if is_real else 'sim'})",
        )
    suffix = "_real" if is_real else "_sim"
    return get_sensor(name + suffix)


def unregister_backend(name: str) -> None:
    _BACKENDS.pop(name, None)


def unregister_robot(name: str) -> None:
    _ROBOTS.pop(name, None)


def unregister_policy(name: str) -> None:
    _POLICIES.pop(name, None)


def unregister_task(name: str) -> None:
    _TASKS.pop(name, None)


def unregister_sensor(name: str) -> None:
    _SENSORS.pop(name, None)


# ---------------------------------------------------------------------------
# User-facing helpers
# ---------------------------------------------------------------------------

def use(module_path: str) -> None:
    """Import a module by dotted path to trigger its @register_* decorators.

    This is the recommended way to register components that live in your own
    project without adding files to the RoboDeploy source tree.

    Args:
        module_path: Dotted Python module path, e.g. "my_project.components"
                     or "my_project.robots.myrobot".

    Raises:
        ImportError: If the module cannot be found. Check that the package is
                     installed or on PYTHONPATH.

    Example:
        from robodeploy import use, RoboEnv

        use("my_project.robots")     # registers @register_robot classes
        use("my_project.tasks")      # registers @register_task classes
        use("my_project.policies")   # registers @register_policy classes

        env = RoboEnv.make(robot="myrobot", backend="mujoco", task="my_task")
    """
    import importlib
    try:
        importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        raise ImportError(
            f"robodeploy.use('{module_path}') failed: module not found.\n"
            f"Original error: {exc}\n"
            "Check that the package is installed or that its parent directory "
            "is on your PYTHONPATH."
        ) from exc


def auto_discover_entry_points() -> None:
    """Discover and import components registered via Python packaging entry points.

    This enables third-party packages to extend RoboDeploy without any code
    changes. A package declares its components in pyproject.toml:

        [project.entry-points."robodeploy.robots"]
        myrobot = "my_package.robots.myrobot:MyRobotDescription"

        [project.entry-points."robodeploy.tasks"]
        my_task = "my_package.tasks.my_task:MyTask"

    Call this once at the top of your script to load all installed extensions:

        from robodeploy.core.registry import auto_discover_entry_points
        auto_discover_entry_points()

    Or use the top-level shorthand:

        from robodeploy import discover
        discover()

    Groups scanned: robodeploy.robots, robodeploy.backends, robodeploy.policies,
                    robodeploy.tasks, robodeploy.sensors
    """
    from importlib.metadata import entry_points

    groups = [
        "robodeploy.robots",
        "robodeploy.backends",
        "robodeploy.policies",
        "robodeploy.tasks",
        "robodeploy.sensors",
    ]
    global _ENTRY_POINT_DISCOVERY
    _ENTRY_POINT_DISCOVERY = True
    try:
        for group in groups:
            for ep in entry_points(group=group):
                try:
                    ep.load()   # importing the object triggers @register_* decorator
                except Exception as exc:
                    warnings.warn(
                        f"Failed to load entry point '{ep.name}' from group "
                        f"'{group}': {exc}",
                        stacklevel=2,
                    )
    finally:
        _ENTRY_POINT_DISCOVERY = False


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------

def list_registered() -> dict[str, list[str]]:
    """Return all registered names grouped by component type.

    Useful for debugging and for building CLI help text.

    Returns:
        dict with keys "backends", "robots", "policies", "tasks", "sensors",
        each mapping to a sorted list of registered names.
    """
    return {
        "backends": sorted(_BACKENDS),
        "robots":   sorted(_ROBOTS),
        "policies": sorted(_POLICIES),
        "tasks":    sorted(_TASKS),
        "sensors":  sorted(_SENSORS),
        "sensor_pairs": sorted(_SENSOR_PAIRS),
    }
