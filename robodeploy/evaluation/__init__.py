"""Benchmark evaluation harness, transfer metrics, and reporting (Goals 5 + 11)."""

from robodeploy.evaluation.failure_analysis import FailureClassifier, classify_episode_failures
from robodeploy.evaluation.harness import EvalConfig, EvalHarness, run_benchmark_eval
from robodeploy.evaluation.leaderboard import list_submissions, show_leaderboard, submit_score
from robodeploy.evaluation.metrics import (
    AggregateMetrics,
    EpisodeMetrics,
    MetricsCollector,
    aggregate_episodes,
    ci95_binomial,
)
from robodeploy.evaluation.registry import BenchmarkRegistry, BenchmarkSpec, BenchmarkTask
from robodeploy.evaluation.manifest import BenchmarkRunManifest
from robodeploy.evaluation.report import EvalReport
from robodeploy.evaluation.render import render_comparison, render_report
from robodeploy.evaluation.schema_validate import validate_benchmark_spec, validate_leaderboard_submission
from robodeploy.evaluation.subproc_eval import SubprocEvalPool
from robodeploy.evaluation.transfer_metrics import (
    EpisodeRollout,
    TransferEvaluator,
    TransferMetrics,
    compute_trajectory_distance,
)
from robodeploy.evaluation.video import EpisodeVideoRecorder

__all__ = [
    "AggregateMetrics",
    "BenchmarkRegistry",
    "BenchmarkSpec",
    "BenchmarkTask",
    "EpisodeRollout",
    "EpisodeVideoRecorder",
    "EvalConfig",
    "EvalHarness",
    "EvalReport",
    "EpisodeMetrics",
    "FailureClassifier",
    "MetricsCollector",
    "BenchmarkRunManifest",
    "SubprocEvalPool",
    "TransferEvaluator",
    "TransferMetrics",
    "aggregate_episodes",
    "ci95_binomial",
    "classify_episode_failures",
    "compute_trajectory_distance",
    "list_submissions",
    "render_comparison",
    "render_report",
    "run_benchmark_eval",
    "show_leaderboard",
    "submit_score",
    "validate_benchmark_spec",
    "validate_leaderboard_submission",
]
