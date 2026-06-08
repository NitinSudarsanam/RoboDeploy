"""Re-export reach_target task from manipulation_v1 (GOAL_11 — import, don't duplicate)."""

from benchmarks.manipulation_v1.reach_target.task import (  # noqa: F401
    BenchmarkReachScriptedPolicy,
    BenchmarkReachTargetTask,
)

__all__ = ["BenchmarkReachTargetTask", "BenchmarkReachScriptedPolicy"]
