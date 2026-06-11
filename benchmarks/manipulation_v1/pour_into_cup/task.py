"""Tier-4 pour_into_cup — dummy proxy via joint reach; full pour on sim backends."""

from __future__ import annotations

from robodeploy.demos.tasks.pour import PourTask as PourTask  # noqa: F401

from benchmarks.manipulation_v1.reach_target.task import (  # noqa: F401
    BenchmarkReachScriptedPolicy,
    BenchmarkReachTargetTask,
)
