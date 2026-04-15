"""Simulation backends powered by JAX + MuJoCo MJX.

Import is lazy so that environments without JAX/mujoco-mjx (e.g. ros2_env)
can import other robodeploy modules without hitting an ImportError here.
MujocoEngine will raise a clear ImportError at instantiation time instead.
"""

__all__ = ["MujocoEngine"]


def __getattr__(name: str):
    if name == "MujocoEngine":
        from .mujoco_engine import MujocoEngine
        return MujocoEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
