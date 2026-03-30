"""Task definitions used by policies and simulation demos."""

from .franka_pick import BasicFrankaPickTask
from .kuka_pick import BasicKukaPickTask

__all__ = ["BasicFrankaPickTask", "BasicKukaPickTask"]
