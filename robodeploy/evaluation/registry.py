"""Benchmark suite registry — discovers tasks under ``benchmarks/``."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import yaml


def _installed_benchmarks_root() -> Path | None:
    """Return benchmarks/ bundled with the installed ``benchmarks`` package."""
    try:
        import benchmarks as benchmarks_pkg
    except ImportError:
        return None
    root = Path(benchmarks_pkg.__file__).resolve().parent
    if (root / "manipulation_v1" / "spec.json").is_file():
        return root
    return None


def benchmarks_root(explicit: str | Path | None = None) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Benchmarks root not found: {path}")
        return path
    env = os.environ.get("ROBODEPLOY_BENCHMARKS_ROOT", "").strip()
    if env:
        path = Path(env).expanduser().resolve()
        if path.is_dir():
            return path
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "benchmarks"
        if candidate.is_dir() and (candidate / "manipulation_v1" / "spec.json").is_file():
            return candidate.resolve()
    installed = _installed_benchmarks_root()
    if installed is not None:
        return installed
    cwd = Path.cwd() / "benchmarks"
    if cwd.is_dir() and (cwd / "manipulation_v1" / "spec.json").is_file():
        return cwd.resolve()
    raise FileNotFoundError(
        "Could not locate benchmarks/ directory. Set ROBODEPLOY_BENCHMARKS_ROOT, "
        "run from the repo root, or pip install robodeploy (bundles benchmarks/)."
    )


@dataclass(frozen=True)
class BenchmarkTask:
    suite: str
    name: str
    tier: int
    max_steps: int
    expected_success: float
    weight: float
    task_dir: Path
    reference_scores_path: Path | None

    @property
    def qualified_name(self) -> str:
        return f"{self.suite}/{self.name}"

    def available_backends(self) -> list[str]:
        backends: list[str] = []
        for path in sorted(self.task_dir.glob("preset_*.yaml")):
            stem = path.stem
            if stem.startswith("preset_"):
                backends.append(stem[len("preset_") :])
        return backends

    def preset_path(self, backend: str) -> Path:
        path = self.task_dir / f"preset_{backend}.yaml"
        if not path.is_file():
            known = ", ".join(self.available_backends()) or "(none)"
            raise FileNotFoundError(
                f"No preset_{backend}.yaml for {self.qualified_name}. Available backends: {known}"
            )
        return path

    def load_preset(self, backend: str) -> dict[str, Any]:
        data = yaml.safe_load(self.preset_path(backend).read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Preset must be a mapping: {self.preset_path(backend)}")
        return dict(data)

    def load_reference_scores(self) -> dict[str, Any] | None:
        if self.reference_scores_path is None or not self.reference_scores_path.is_file():
            return None
        return json.loads(self.reference_scores_path.read_text(encoding="utf-8"))

    def import_task_module(self) -> None:
        task_py = self.task_dir / "task.py"
        if not task_py.is_file():
            return
        module_name = f"benchmarks.{self.suite}.{self.name}.task"
        from robodeploy.core.registry import use

        use(module_name)


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    version: str
    tasks: tuple[BenchmarkTask, ...]
    spec_path: Path
    scoring_method: str
    tie_breaker: str

    def get_task(self, name: str) -> BenchmarkTask:
        for task in self.tasks:
            if task.name == name:
                return task
        known = ", ".join(t.name for t in self.tasks) or "(none)"
        raise KeyError(f"Unknown benchmark task '{name}' in suite '{self.name}'. Known: {known}")

    def robo_score(self, success_by_task: dict[str, float]) -> float:
        total_w = 0.0
        score = 0.0
        for task in self.tasks:
            if task.name not in success_by_task:
                continue
            total_w += task.weight
            score += task.weight * success_by_task[task.name]
        if total_w <= 0:
            return 0.0
        return score / total_w


class BenchmarkRegistry:
    def __init__(self, root: str | Path | None = None) -> None:
        self._root = benchmarks_root(root)

    @property
    def root(self) -> Path:
        return self._root

    def list_suites(self) -> list[str]:
        names: list[str] = []
        for path in sorted(self._root.iterdir()):
            if path.is_dir() and (path / "spec.json").is_file():
                names.append(path.name)
        return names

    def load_suite(self, suite_name: str) -> BenchmarkSpec:
        suite_dir = self._root / suite_name
        spec_path = suite_dir / "spec.json"
        if not spec_path.is_file():
            raise FileNotFoundError(f"Suite spec not found: {spec_path}")
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        tasks_cfg = spec.get("tasks") or []
        tasks: list[BenchmarkTask] = []
        for entry in tasks_cfg:
            name = str(entry["name"])
            task_dir = suite_dir / name
            if not task_dir.is_dir():
                continue
            ref = task_dir / "reference_scores.json"
            tasks.append(
                BenchmarkTask(
                    suite=suite_name,
                    name=name,
                    tier=int(entry.get("tier", 0)),
                    max_steps=int(entry.get("max_steps", 1000)),
                    expected_success=float(entry.get("expected_success", 0.0)),
                    weight=float(entry.get("weight", 1.0)),
                    task_dir=task_dir.resolve(),
                    reference_scores_path=ref if ref.is_file() else None,
                )
            )
        scoring = spec.get("scoring") or {}
        return BenchmarkSpec(
            name=str(spec.get("name", suite_name)),
            version=str(spec.get("version", "1.0")),
            tasks=tuple(tasks),
            spec_path=spec_path.resolve(),
            scoring_method=str(scoring.get("method", "weighted_success_rate")),
            tie_breaker=str(scoring.get("tie_breaker", "mean_reward")),
        )

    def resolve(self, benchmark: str) -> tuple[BenchmarkSpec, BenchmarkTask | None]:
        """Resolve ``suite`` or ``suite/task``."""
        parts = [p for p in str(benchmark).strip("/\\").split("/") if p]
        if not parts:
            raise ValueError("Benchmark name is required.")
        suite = self.load_suite(parts[0])
        if len(parts) == 1:
            return suite, None
        return suite, suite.get_task(parts[1])

    def iter_tasks(self, benchmark: str) -> Iterator[BenchmarkTask]:
        suite, task = self.resolve(benchmark)
        if task is not None:
            yield task
            return
        yield from suite.tasks
