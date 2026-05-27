"""Task implementations."""

from robodeploy.tasks.manipulation.peg_insertion import PegTask
from robodeploy.tasks.manipulation.pick_place import PickPlaceTask
from robodeploy.tasks.manipulation.pour import PourTask

PegInsertionTask = PegTask

__all__ = ["PickPlaceTask", "PourTask", "PegTask", "PegInsertionTask"]
