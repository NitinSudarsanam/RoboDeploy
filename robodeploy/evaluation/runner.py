"""High-level benchmark runner — suite/task resolution, preset sweep, JSON output."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from robodeploy.evaluation.failure_analysis import classify_episode_failures
from robodeploy.evaluation.harness import EvalConfig, run_benchmark_eval
from robodeploy.evaluation.policy_loader import is_checkpoint_path
from robodeploy.evaluation.registry import BenchmarkRegistry, BenchmarkTask
from robodeploy.evaluation.report import EvalReport, merge_task_reports
from robodeploy.evaluation.manifest import BenchmarkRunManifest


def _override_policy(preset: dict[str, Any], policy: str | None) -> dict[str, Any]:
    if not policy or policy in {"scripted", "policy"}:
        return dict(preset)
    cfg = dict(preset)
    if is_checkpoint_path(policy):
        cfg["policy"] = policy
        cfg.setdefault("policy_kwargs", {})
        return cfg
    cfg["policy"] = policy
    return cfg


def _action_mode_for_policy(policy: str | None) -> str | None:
    if policy in {None, "", "scripted", "policy"}:
        return None
    if is_checkpoint_path(str(policy or "")):
        return None
    if policy in {"none", "zero", "hold", "sinusoid"}:
        return policy
    return None


def _policy_label(policy: str) -> str:
    if is_checkpoint_path(policy):
        return Path(policy).name
    return policy


def run_eval(
    *,
    benchmark: str,
    policy: str = "scripted",
    backend: str = "dummy",
    episodes: int = 100,
    base_seed: int = 0,
    max_steps: int | None = None,
    parallel: bool = False,
    n_workers: int = 4,
    sweep_backends: bool = False,
    benchmarks_root: str | None = None,
    record_videos: bool = False,
    video_dir: str | None = None,
    record_trajectories: bool = False,
    html_output: str | None = None,
    baseline_report: str | None = None,
) -> EvalReport:
    registry = BenchmarkRegistry(benchmarks_root)
    suite, _single_task = registry.resolve(benchmark)
    tasks = list(registry.iter_tasks(benchmark))
    if not tasks:
        raise ValueError(f"No tasks found for benchmark '{benchmark}'.")

    policy_label = _policy_label(policy)
    config = EvalConfig(
        n_episodes=int(episodes),
        base_seed=int(base_seed),
        parallel=bool(parallel),
        n_workers=int(n_workers),
        action_mode=_action_mode_for_policy(policy),
        record_videos=bool(record_videos),
        video_dir=video_dir,
        record_trajectories=bool(record_trajectories),
    )

    task_reports: list[EvalReport] = []
    for task in tasks:
        backends = task.available_backends() if sweep_backends else [backend]
        for backend_name in backends:
            preset = _override_policy(task.load_preset(backend_name), policy)
            task.import_task_module()
            steps = int(max_steps or task.max_steps)
            task_config = replace(config, max_steps_per_episode=steps)
            report = run_benchmark_eval(
                preset=preset,
                benchmark_name=task.qualified_name,
                benchmark_version=suite.version,
                backend=backend_name,
                policy_name=policy_label,
                config=task_config,
                manifest_extra={"tier": task.tier, "weight": task.weight},
            )
            report.task_results[0]["weight"] = task.weight
            report.task_results[0]["tier"] = task.tier
            report.task_results[0]["expected_success"] = task.expected_success
            task_reports.append(report)

    if len(task_reports) == 1:
        merged = task_reports[0]
    else:
        manifest = BenchmarkRunManifest.build(
            benchmark=benchmark,
            benchmark_version=suite.version,
            policy=policy_label,
            backend=backend if not sweep_backends else "sweep",
            seed_base=int(base_seed),
            n_episodes=int(episodes),
            extra={"sweep_backends": sweep_backends},
        )
        merged = merge_task_reports(
            suite_name=benchmark,
            suite_version=suite.version,
            reports=task_reports,
            config=config,
            manifest=manifest,
        )
        success_by_task = {
            str(item["benchmark"]).split("/", 1)[-1]: float(item["success_rate"])
            for item in merged.task_results
            if "benchmark" in item and "success_rate" in item
        }
        robo_score = suite.robo_score(success_by_task)
        merged.aggregate = replace(merged.aggregate, robo_score=robo_score)

    failure_counts = classify_episode_failures(merged.episodes)
    if html_output:
        baseline_payload = None
        if baseline_report:
            import json

            baseline_payload = json.loads(Path(baseline_report).read_text(encoding="utf-8"))
        video_paths = [
            str(ep.metadata["video_path"])
            for ep in merged.episodes
            if ep.metadata.get("video_path")
        ]
        merged.render_html(
            html_output,
            baseline=baseline_payload,
            failure_counts=failure_counts,
            video_paths=video_paths,
        )
    return merged


def list_benchmarks(*, benchmarks_root: str | None = None) -> dict[str, Any]:
    registry = BenchmarkRegistry(benchmarks_root)
    suites: dict[str, Any] = {}
    for suite_name in registry.list_suites():
        suite = registry.load_suite(suite_name)
        suites[suite_name] = {
            "version": suite.version,
            "tasks": [
                {
                    "name": task.name,
                    "tier": task.tier,
                    "max_steps": task.max_steps,
                    "backends": task.available_backends(),
                }
                for task in suite.tasks
            ],
        }
    return suites
