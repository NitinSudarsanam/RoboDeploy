"""Eval report serialization."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    from robodeploy.evaluation.harness import EvalConfig

from robodeploy.evaluation.metrics import AggregateMetrics, EpisodeMetrics, aggregate_episodes
from robodeploy.observability.manifest import RunManifest  # Goal 10 — single manifest type


@dataclass
class EvalReport:
    benchmark_name: str
    benchmark_version: str
    episodes: list[EpisodeMetrics]
    aggregate: AggregateMetrics
    config: EvalConfig
    manifest: RunManifest
    started_at: float
    finished_at: float
    task_results: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "benchmark_name": self.benchmark_name,
            "benchmark_version": self.benchmark_version,
            "aggregate": self.aggregate.to_dict(),
            "episodes": [ep.to_dict() for ep in self.episodes],
            "config": self.config.to_dict(),
            "manifest": self.manifest.to_dict(),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": max(0.0, self.finished_at - self.started_at),
            "task_results": list(self.task_results),
        }

    def save(self, path: Path | str) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_json(), indent=2, sort_keys=True), encoding="utf-8")
        return out

    def render_html(
        self,
        out: Path | str,
        *,
        baseline: EvalReport | dict[str, Any] | None = None,
        failure_counts: dict[str, int] | None = None,
        video_paths: list[str] | None = None,
    ) -> Path:
        from robodeploy.evaluation.render import render_report

        path = Path(out)
        render_report(
            self,
            baseline=baseline,
            out=path,
            failure_counts=failure_counts,
            video_paths=video_paths,
        )
        return path


def merge_task_reports(
    *,
    suite_name: str,
    suite_version: str,
    reports: Sequence[EvalReport],
    config: EvalConfig,
    manifest: RunManifest,
) -> EvalReport:
    episodes = [ep for report in reports for ep in report.episodes]
    weights = []
    for report in reports:
        weight = 1.0
        if report.task_results:
            weight = float(report.task_results[0].get("weight", 1.0))
        weights.extend([weight] * len(report.episodes))
    aggregate = aggregate_episodes(episodes, weights=weights if weights else None)
    started = min(r.started_at for r in reports) if reports else time.time()
    finished = max(r.finished_at for r in reports) if reports else time.time()
    task_results = [item for report in reports for item in report.task_results]
    return EvalReport(
        benchmark_name=suite_name,
        benchmark_version=suite_version,
        episodes=episodes,
        aggregate=aggregate,
        config=config,
        manifest=manifest,
        started_at=started,
        finished_at=finished,
        task_results=task_results,
    )
