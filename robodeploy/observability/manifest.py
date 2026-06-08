"""Run manifest for reproducible experiment metadata."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from robodeploy.env import RoboEnv


def _git_state() -> tuple[str | None, bool]:
    try:
        import git

        repo = git.Repo(search_parent_directories=True)
        return repo.head.commit.hexsha[:12], repo.is_dirty()
    except Exception:
        pass
    try:
        root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()[:12]
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=root,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        )
        return commit, dirty
    except Exception:
        return None, False


def _package_version() -> str:
    try:
        from importlib.metadata import version

        return version("robodeploy")
    except Exception:
        from robodeploy import __version__

        return __version__


@dataclass
class RunManifest:
    run_name: str
    started_at: float
    finished_at: float | None = None
    seed: int | None = None
    env_config: dict[str, Any] = field(default_factory=dict)
    backend: str = ""
    backend_version: str | None = None
    robot: str = ""
    task: str = ""
    policy: str = ""
    policy_checkpoint: str | None = None
    git_hash: str | None = None
    git_dirty: bool = False
    python_version: str = ""
    package_version: str = ""
    asset_manifest_hash: str | None = None
    sensor_rig: list[str] = field(default_factory=list)
    seeds: dict[str, int] = field(default_factory=dict)
    # Benchmark eval fields (Goal 11 consumes the same type — no duplicate manifest).
    benchmark: str | None = None
    benchmark_version: str | None = None
    n_episodes: int | None = None
    preset_path: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "RunManifest":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in payload.items() if k in known}
        return cls(**filtered)

    def finish(self) -> None:
        self.finished_at = time.time()

    @classmethod
    def build(
        cls,
        *,
        benchmark: str,
        benchmark_version: str,
        policy: str,
        backend: str,
        seed_base: int,
        n_episodes: int,
        preset_path: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> "RunManifest":
        """Backward-compatible alias for benchmark eval manifests."""
        return cls.for_benchmark_eval(
            benchmark=benchmark,
            benchmark_version=benchmark_version,
            policy=policy,
            backend=backend,
            seed_base=seed_base,
            n_episodes=n_episodes,
            preset_path=preset_path,
            extra=extra,
        )

    @classmethod
    def for_benchmark_eval(
        cls,
        *,
        benchmark: str,
        benchmark_version: str,
        policy: str,
        backend: str,
        seed_base: int,
        n_episodes: int,
        preset_path: str | None = None,
        extra: dict[str, Any] | None = None,
        run_name: str | None = None,
    ) -> "RunManifest":
        """Build a manifest for benchmark evaluation (shared with EvalReport)."""
        git_hash, git_dirty = _git_state()
        extra_dict = dict(extra or {})
        return cls(
            run_name=run_name or f"eval-{benchmark.replace('/', '-')}",
            started_at=time.time(),
            seed=int(seed_base),
            env_config={"benchmark": benchmark, "benchmark_version": benchmark_version, **extra_dict},
            backend=backend,
            policy=policy,
            benchmark=benchmark,
            benchmark_version=benchmark_version,
            n_episodes=int(n_episodes),
            preset_path=preset_path,
            git_hash=git_hash,
            git_dirty=git_dirty,
            python_version=platform.python_version(),
            package_version=_package_version(),
            extra=extra_dict,
        )


class ManifestRecorder:
    """Capture run metadata from a RoboEnv instance."""

    def __init__(self, env: "RoboEnv", *, run_name: str | None = None) -> None:
        self._env = env
        git_hash, git_dirty = _git_state()
        primary = env.primary_robot
        primary_task = primary.tasks.get(primary.active_task_id or next(iter(primary.tasks)))
        policy_name = ""
        policy_checkpoint = None
        if primary_task is not None and primary_task.policies:
            policy_name = next(iter(primary_task.policies))
            policy_obj = primary_task.policies[policy_name]
            policy_checkpoint = str(getattr(policy_obj, "config", {}).get("checkpoint") or "") or None

        sensor_names = [str(getattr(s, "name", type(s).__name__)) for s in env._all_sensors()]

        self.manifest = RunManifest(
            run_name=run_name or f"run-{int(time.time())}",
            started_at=time.time(),
            seed=getattr(env, "master_seed", None),
            env_config=getattr(env, "env_config_snapshot", {}),
            backend=type(env.backend).__name__,
            backend_version=getattr(env.backend, "version", None),
            robot=getattr(primary.description, "display_name", type(primary.description).__name__),
            task=primary.active_task_id or "",
            policy=policy_name,
            policy_checkpoint=policy_checkpoint,
            git_hash=git_hash,
            git_dirty=git_dirty,
            python_version=platform.python_version(),
            package_version=_package_version(),
            sensor_rig=sensor_names,
            seeds=getattr(env, "seed_snapshot", {}),
        )

    def write(self, out_dir: str | Path) -> Path:
        self.manifest.finish()
        path = Path(out_dir) / "manifest.json"
        self.manifest.save(path)
        return path
