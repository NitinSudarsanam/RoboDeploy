from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List

from robodeploy.core.interfaces.sensor import ISensor
from robodeploy.description.base import RobotDescription

if TYPE_CHECKING:
    from robodeploy.action_adapter import ActionAdapter
    from robodeploy.obs_pipeline import ObsPipeline


def _default_obs_pipeline():
    from robodeploy.obs_pipeline import ObsPipeline
    return ObsPipeline()


def _default_action_adapter():
    from robodeploy.action_adapter import ActionAdapter
    return ActionAdapter()


@dataclass
class RobotConfig:
    """Per-robot configuration for multi-agent RoboEnv.

    Matches the shape described in ARCHITECTURE.md. Single-agent code should
    not construct this directly; RoboEnv will wrap description/task for you.
    """

    description: RobotDescription
    obs_pipeline: "ObsPipeline" = field(default_factory=_default_obs_pipeline)
    action_adapter: "ActionAdapter" = field(default_factory=_default_action_adapter)
    sensors: List[ISensor] = field(default_factory=list)
    robot_id: str = ""

    def swap_sensor(self, name: str, replacement: ISensor) -> None:
        """Placeholder for sensor hot-swap (not implemented yet)."""
        raise NotImplementedError("RobotConfig.swap_sensor is not implemented yet.")

