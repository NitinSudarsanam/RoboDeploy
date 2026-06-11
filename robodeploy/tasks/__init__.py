"""Task framework (reference tasks ship in ``examples.tasks``)."""

from robodeploy.tasks.base import TaskBase
from robodeploy.tasks.reward_builder import RewardBuilder
from robodeploy.tasks.success_predicates import CompoundSuccess, get_success_predicate, register_success
from robodeploy.tasks.choreography import TaskChoreography
from robodeploy.tasks.templates import InsertionTemplate, PickPlaceTemplate, PourTemplate, StackingTemplate

__all__ = [
    "TaskBase",
    "PickPlaceTemplate",
    "PourTemplate",
    "InsertionTemplate",
    "StackingTemplate",
    "TaskChoreography",
    "RewardBuilder",
    "CompoundSuccess",
    "register_success",
    "get_success_predicate",
]
