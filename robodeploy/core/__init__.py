"""
Core module for RoboDeploy.
Provides base classes, types, and utilities for robot control and simulation.
"""

from . import interfaces, interop, types
from .interfaces import backend, policy, sensor, task
from . import registry, robot, selectors, spaces, transforms
from .robot import Robot, RobotTask
from .selectors import IPolicySelector, ITaskSelector, WeightTaskSelector, WeightedPolicySelector
from .types import Action, Observation, SceneSpec, SensorData, WorldSpec

__all__ = [
    "types",
    "spaces",
    "interfaces",
    "registry",
    "transforms",
    "interop",
    "backend",
    "policy",
    "sensor",
    "task",
    "robot",
    "selectors",
    "Robot",
    "RobotTask",
    "Action",
    "Observation",
    "SceneSpec",
    "SensorData",
    "WorldSpec",
    "ITaskSelector",
    "IPolicySelector",
    "WeightTaskSelector",
    "WeightedPolicySelector",
]
