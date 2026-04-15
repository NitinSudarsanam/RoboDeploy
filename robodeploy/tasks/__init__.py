"""Task definitions used by policies and simulation demos.

Imports are lazy so that JAX-free backends (e.g. ros2_env without JAX) can
import PandaOscillationTask without triggering JAX imports from sim-only tasks.
"""

__all__ = ["BasicFrankaPickTask", "BasicKukaPickTask", "PandaOscillationTask"]


def __getattr__(name: str):
    if name == "BasicFrankaPickTask":
        from .franka_pick import BasicFrankaPickTask
        return BasicFrankaPickTask
    if name == "BasicKukaPickTask":
        from .kuka_pick import BasicKukaPickTask
        return BasicKukaPickTask
    if name == "PandaOscillationTask":
        from .panda_oscillation import PandaOscillationTask
        return PandaOscillationTask
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
