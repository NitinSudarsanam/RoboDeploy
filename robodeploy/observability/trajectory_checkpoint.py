"""Standard trajectory checkpoint format for eval replay and failure analysis (Goal 10 → 11)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robodeploy.demo_recording import DemoRecorder

if TYPE_CHECKING:
    from robodeploy.evaluation.metrics import EpisodeMetrics
    from robodeploy.observability.manifest import RunManifest

CHECKPOINT_SCHEMA_VERSION = 1


@dataclass
class TrajectoryCheckpoint:
    """Single-episode trajectory bundle saved alongside benchmark eval runs."""

    schema_version: int = CHECKPOINT_SCHEMA_VERSION
    episode_index: int = 0
    seed: int = 0
    episode_id: str = ""
    manifest: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] | None = None
    frames: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return out

    @classmethod
    def load(cls, path: str | Path) -> "TrajectoryCheckpoint":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in payload.items() if k in known}
        return cls(**filtered)

    @classmethod
    def from_episode(
        cls,
        *,
        recorder: DemoRecorder,
        manifest: "RunManifest",
        metrics: "EpisodeMetrics | None" = None,
        episode_index: int,
        seed: int,
        episode_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> "TrajectoryCheckpoint":
        return cls(
            episode_index=int(episode_index),
            seed=int(seed),
            episode_id=str(episode_id or recorder.metadata.get("episode_id", "")),
            manifest=manifest.to_dict(),
            metrics=metrics.to_dict() if metrics is not None else None,
            frames=[asdict(frame) for frame in recorder.frames],
            metadata={**dict(recorder.metadata), **dict(metadata or {})},
        )

    def default_filename(self) -> str:
        return f"episode_{self.episode_index:04d}_seed{self.seed}.checkpoint.json"

    def to_recorder(self) -> DemoRecorder:
        """Convert checkpoint frames into a DemoRecorder for TrajectoryReplayer."""
        from robodeploy.demo_recording import DemoFrame

        recorder = DemoRecorder()
        recorder.metadata = dict(self.metadata)
        recorder.metadata.setdefault("seed", int(self.seed))
        for item in self.frames:
            recorder.frames.append(DemoFrame(**item))
        return recorder

    def run_manifest(self) -> "RunManifest":
        """Reconstruct the embedded Goal 10 RunManifest."""
        from robodeploy.observability.manifest import RunManifest

        known = {f.name for f in RunManifest.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in self.manifest.items() if k in known}
        return RunManifest(**filtered)


def write_trajectory_checkpoint(
    *,
    out_dir: str | Path,
    recorder: DemoRecorder,
    manifest: "RunManifest",
    metrics: "EpisodeMetrics | None",
    episode_index: int,
    seed: int,
    episode_id: str = "",
) -> Path:
    """Persist a checkpoint under ``out_dir`` using the standard filename."""
    checkpoint = TrajectoryCheckpoint.from_episode(
        recorder=recorder,
        manifest=manifest,
        metrics=metrics,
        episode_index=episode_index,
        seed=seed,
        episode_id=episode_id,
    )
    return checkpoint.save(Path(out_dir) / checkpoint.default_filename())
