from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from robodeploy.action_adapter import ActionAdapter
from robodeploy.obs_pipeline import ObsPipeline
from robodeploy.core.interfaces.sensor import ISensor
from robodeploy.description.base import RobotDescription


@dataclass
class RobotConfig:
    """Per-robot configuration for multi-agent RoboEnv.

    Matches the shape described in ARCHITECTURE.md. Single-agent code should
    not construct this directly; RoboEnv will wrap description/task for you.
    """

    description: RobotDescription
    obs_pipeline: ObsPipeline = field(default_factory=ObsPipeline)
    action_adapter: ActionAdapter = field(default_factory=ActionAdapter)
    sensors: List[ISensor] = field(default_factory=list)
    robot_id: str = ""

    def swap_sensor(self, name: str, replacement: ISensor) -> None:
        """Placeholder for sensor hot-swap (not implemented yet)."""
        raise NotImplementedError("RobotConfig.swap_sensor is not implemented yet.")

