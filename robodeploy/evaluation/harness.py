"""Evaluation harness — runs N seeded episodes and aggregates metrics."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from robodeploy.cli_helpers import action_fn_for_mode, close_quietly
from robodeploy.core.interfaces.policy import IPolicy
from robodeploy.core.interfaces.task import ITask
from robodeploy.core.types import Observation
from robodeploy.env import RoboEnv
from robodeploy.evaluation.failure_analysis import classify_episode_failures
from robodeploy.evaluation.manifest import BenchmarkRunManifest, build_benchmark_manifest
from robodeploy.observability.manifest import RunManifest
from robodeploy.evaluation.metrics import EpisodeMetrics, MetricsCollector, aggregate_episodes
from robodeploy.evaluation.report import EvalReport
from robodeploy.evaluation.subproc_eval import SubprocEvalPool
from robodeploy.evaluation.video import EpisodeVideoRecorder


@dataclass
class EvalConfig:
    n_episodes: int = 100
    seeds: list[int] | None = None
    base_seed: int = 0
    max_steps_per_episode: int = 1000
    parallel: bool = False
    n_workers: int = 4
    parallel_backend: str = "subproc"
    deterministic_policy: bool = True
    record_videos: bool = False
    video_dir: str | None = None
    record_trajectories: bool = False
    trajectory_dir: str | None = None
    action_mode: str | None = None
    classify_failures: bool = True

    def resolve_seeds(self) -> list[int]:
        if self.seeds is not None:
            return list(self.seeds)
        return [int(self.base_seed) + i for i in range(int(self.n_episodes))]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _episode_job(
    *,
    preset: dict[str, Any],
    seed: int,
    max_steps: int,
    action_mode: str | None,
    record_videos: bool,
    video_dir: str | None,
    record_trajectories: bool,
    trajectory_dir: str | None = None,
    trajectory_manifest: dict[str, Any] | None = None,
    episode_index: int = 0,
) -> EpisodeMetrics:
    from robodeploy.evaluation.env_builder import build_env_from_preset

    from robodeploy.demo_recording import DemoRecorder, DemoSession

    env = build_env_from_preset(preset, seed=int(seed))
    trajectories: list[Observation] = []
    video_recorder: EpisodeVideoRecorder | None = None
    recorder = DemoRecorder()
    recorder.metadata["seed"] = int(seed)
    session = DemoSession(env, recorder=recorder) if record_trajectories and trajectory_dir else None
    try:
        primary_task = next(iter(env.primary_robot.tasks.values())).task
        collector = MetricsCollector(
            task=primary_task,
            robot_description=env.primary_robot.description,
        )
        if record_videos and video_dir:
            video_recorder = EpisodeVideoRecorder(env=env, out_dir=Path(video_dir))
            video_recorder.start(f"ep_{seed}")

        obs, info = (session.reset(seed=int(seed)) if session else env.reset(seed=int(seed)))
        if record_trajectories:
            trajectories.append(obs)
        if video_recorder is not None:
            video_recorder.observe(obs)

        action_fn = action_fn_for_mode(action_mode, env) if action_mode else None

        for _step in range(int(max_steps)):
            action = action_fn(obs) if action_fn is not None else None
            if session is not None:
                obs, reward, done, info = session.step(action)
            else:
                obs, reward, done, info = env.step(action)
            collector.observe(obs, action, reward, info)
            if record_trajectories:
                trajectories.append(obs)
            if video_recorder is not None:
                video_recorder.observe(obs)
            if done:
                break

        metrics = collector.finalize()
        if session is not None and trajectory_dir and trajectory_manifest:
            from robodeploy.observability.trajectory_checkpoint import write_trajectory_checkpoint

            manifest = RunManifest(**trajectory_manifest)
            write_trajectory_checkpoint(
                out_dir=trajectory_dir,
                recorder=recorder,
                manifest=manifest,
                metrics=metrics,
                episode_index=episode_index,
                seed=int(seed),
                episode_id=str(getattr(info, "episode_id", episode_index)),
            )
        metrics.metadata["seed"] = int(seed)
        metrics.metadata["max_steps"] = int(max_steps)
        if video_recorder is not None:
            video_path = video_recorder.finish()
            if video_path is not None:
                metrics.metadata["video_path"] = str(video_path)
        if trajectories:
            metrics.metadata["trajectory_len"] = len(trajectories)
        if not metrics.success and trajectories:
            from robodeploy.evaluation.failure_analysis import FailureClassifier

            metrics.metadata["failure_category"] = FailureClassifier().classify(metrics, trajectories)
        return metrics
    finally:
        close_quietly(env)


def run_episode_job(job: dict[str, Any]) -> EpisodeMetrics:
    """Top-level picklable entry for SubprocEvalPool workers."""
    return _episode_job(**job)


class EvalHarness:
    def __init__(
        self,
        *,
        env_factory: Callable[[int], RoboEnv],
        task: ITask,
        config: EvalConfig,
        benchmark_name: str,
        benchmark_version: str,
        manifest: RunManifest,
        policy_factory: Callable[[], IPolicy] | None = None,
        action_fn: Callable | None = None,
        preset: dict[str, Any] | None = None,
    ) -> None:
        self._env_factory = env_factory
        self._task = task
        self._config = config
        self._benchmark_name = benchmark_name
        self._benchmark_version = benchmark_version
        self._manifest = manifest
        self._policy_factory = policy_factory
        self._action_fn = action_fn
        self._preset = dict(preset or {})

    def run(self) -> EvalReport:
        started = time.time()
        seeds = self._config.resolve_seeds()
        if self._config.parallel and len(seeds) > 1 and self._preset:
            episodes = self._run_parallel_subproc(seeds)
        elif self._config.parallel and len(seeds) > 1:
            episodes = [self._run_episode(seed, episode_index=i) for i, seed in enumerate(seeds)]
        else:
            episodes = [self._run_episode(seed, episode_index=i) for i, seed in enumerate(seeds)]

        if self._config.classify_failures:
            classify_episode_failures(episodes)

        aggregate = aggregate_episodes(episodes)
        finished = time.time()
        return EvalReport(
            benchmark_name=self._benchmark_name,
            benchmark_version=self._benchmark_version,
            episodes=episodes,
            aggregate=aggregate,
            config=self._config,
            manifest=self._manifest,
            started_at=started,
            finished_at=finished,
        )

    def _run_parallel_subproc(self, seeds: list[int]) -> list[EpisodeMetrics]:
        workers = max(1, min(int(self._config.n_workers), len(seeds)))
        jobs = [
            {
                "preset": self._preset,
                "seed": int(seed),
                "max_steps": int(self._config.max_steps_per_episode),
                "action_mode": self._config.action_mode,
                "record_videos": bool(self._config.record_videos),
                "video_dir": self._config.video_dir,
                "record_trajectories": bool(self._config.record_trajectories),
                "trajectory_dir": self._config.trajectory_dir,
                "trajectory_manifest": self._manifest.to_dict(),
                "episode_index": i,
            }
            for i, seed in enumerate(seeds)
        ]
        with SubprocEvalPool(run_episode_job, n_workers=workers) as pool:
            return pool.map_episode_jobs(jobs)

    def _run_episode(self, seed: int, *, episode_index: int = 0) -> EpisodeMetrics:
        if self._preset:
            return _episode_job(
                preset=self._preset,
                seed=int(seed),
                max_steps=int(self._config.max_steps_per_episode),
                action_mode=self._config.action_mode,
                record_videos=bool(self._config.record_videos),
                video_dir=self._config.video_dir,
                record_trajectories=bool(self._config.record_trajectories),
                trajectory_dir=self._config.trajectory_dir,
                trajectory_manifest=self._manifest.to_dict(),
                episode_index=episode_index,
            )
        env = self._env_factory(seed)
        try:
            collector = MetricsCollector(
                task=self._task,
                robot_description=env.primary_robot.description,
            )
            obs, _info = env.reset(seed=int(seed))
            action_fn = self._action_fn
            if action_fn is None and self._config.action_mode:
                action_fn = action_fn_for_mode(self._config.action_mode, env)

            for _step in range(int(self._config.max_steps_per_episode)):
                action = action_fn(obs) if action_fn is not None else None
                obs, reward, done, info = env.step(action)
                collector.observe(obs, action, reward, info)
                if done:
                    break
            metrics = collector.finalize()
            metrics.metadata["seed"] = int(seed)
            return metrics
        finally:
            close_quietly(env)


def run_benchmark_eval(
    *,
    preset: dict[str, Any],
    benchmark_name: str,
    benchmark_version: str,
    backend: str,
    policy_name: str,
    config: EvalConfig,
    manifest_extra: dict[str, Any] | None = None,
) -> EvalReport:
    from robodeploy.evaluation.env_builder import build_env_from_preset, make_env_factory

    sample_env = build_env_from_preset(preset, seed=int(config.base_seed))
    try:
        primary_task = next(iter(sample_env.primary_robot.tasks.values())).task
    finally:
        close_quietly(sample_env)

    manifest = BenchmarkRunManifest.build(
        benchmark=benchmark_name,
        benchmark_version=benchmark_version,
        policy=policy_name,
        backend=backend,
        seed_base=int(config.base_seed),
        n_episodes=int(config.n_episodes),
        extra=dict(manifest_extra or {}),
    )

    harness = EvalHarness(
        env_factory=make_env_factory(preset),
        task=primary_task,
        config=config,
        benchmark_name=benchmark_name,
        benchmark_version=benchmark_version,
        manifest=manifest,
        preset=preset,
    )
    report = harness.run()
    report.task_results.append(
        {
            "benchmark": benchmark_name,
            "backend": backend,
            "policy": policy_name,
            "success_rate": report.aggregate.success_rate,
            "mean_reward": report.aggregate.mean_reward,
        }
    )
    return report
