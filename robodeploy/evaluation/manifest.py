"""Benchmark eval manifest — re-exports Goal 10 RunManifest (no duplicate type)."""

from __future__ import annotations

from typing import Any

from robodeploy.observability.manifest import RunManifest

# Backward-compatible alias; EvalReport.manifest is RunManifest per GOAL_11 D3.
BenchmarkRunManifest = RunManifest


def build_benchmark_manifest(
    *,
    benchmark: str,
    benchmark_version: str,
    policy: str,
    backend: str,
    seed_base: int,
    n_episodes: int,
    preset_path: str | None = None,
    extra: dict[str, Any] | None = None,
) -> RunManifest:
    return RunManifest.for_benchmark_eval(
        benchmark=benchmark,
        benchmark_version=benchmark_version,
        policy=policy,
        backend=backend,
        seed_base=seed_base,
        n_episodes=n_episodes,
        preset_path=preset_path,
        extra=extra,
    )
