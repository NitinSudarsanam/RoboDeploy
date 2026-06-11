"""Example tasks — re-export packaged demos for repo-local development."""

from robodeploy.demos.tasks import (  # noqa: F401
    PegInsertionTask,
    PegTask,
    PickPlaceTask,
    PourTask,
    ShowcaseSceneTask,
)
from robodeploy.demos import tasks as tasks  # noqa: F401

__all__ = ["PickPlaceTask", "PourTask", "PegTask", "PegInsertionTask", "ShowcaseSceneTask"]
