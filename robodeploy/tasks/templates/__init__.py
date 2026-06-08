"""Task templates that reduce boilerplate for common manipulation patterns."""

from robodeploy.tasks.templates.insertion import InsertionTemplate
from robodeploy.tasks.templates.pick_place import PickPlaceTemplate
from robodeploy.tasks.templates.pour import PourTemplate
from robodeploy.tasks.templates.stacking import StackingTemplate

__all__ = [
    "PickPlaceTemplate",
    "PourTemplate",
    "InsertionTemplate",
    "StackingTemplate",
]
