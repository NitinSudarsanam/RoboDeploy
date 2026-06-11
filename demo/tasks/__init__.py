"""Demo tasks (register on import via ``demo.tasks``)."""

from . import pick_place as pick_place  # noqa: F401
from .pick_place import DemoPickPlaceTask

__all__ = ["DemoPickPlaceTask"]
