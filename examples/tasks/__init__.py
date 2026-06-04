"""Example tasks (register on import via ``use("examples.tasks")``)."""

from . import peg_insertion as peg_insertion  # noqa: F401
from . import pick_place as pick_place  # noqa: F401
from . import pour as pour  # noqa: F401
from .peg_insertion import PegTask
from .pick_place import PickPlaceTask
from .pour import PourTask

PegInsertionTask = PegTask

__all__ = ["PickPlaceTask", "PourTask", "PegTask", "PegInsertionTask"]
