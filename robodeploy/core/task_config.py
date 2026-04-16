from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from robodeploy.core.interfaces.policy import IPolicy
from robodeploy.core.interfaces.task import ITask
from robodeploy.core.types import MultiTaskMode


@dataclass
class TaskConfig:
    """Per-task configuration for multi-agent RoboEnv."""

    task: ITask
    robot_ids: List[str]
    policy: Optional[IPolicy]
    task_id: str = ""
    obs_stats: Optional[Dict[str, Any]] = None
    mode: MultiTaskMode = "sequential"
    priority: int = 0
    control_hz_override: Optional[float] = None
    preserve_policy_state_on_deactivate: bool = False
    action_resolver: Optional[str] = None

