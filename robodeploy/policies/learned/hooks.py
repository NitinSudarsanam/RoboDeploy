"""Training-loop hooks for learned policy export (GOAL 02 integration point)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from robodeploy.policies.learned.loader import ModelSpec


class TrainingCheckpointHook(Protocol):
    """Called by a future training loop when a checkpoint is saved."""

    def on_checkpoint_saved(self, path: Path, spec: ModelSpec) -> None: ...


@dataclass
class DeployableCheckpointHook:
    """Write a sidecar JSON spec next to each training checkpoint."""

    def on_checkpoint_saved(self, path: Path, spec: ModelSpec) -> None:
        sidecar = path.with_suffix(path.suffix + ".robodeploy.json")
        payload = {
            "framework": spec.get("framework", "custom"),
            "checkpoint": str(path),
            "expected_action_space": getattr(spec.get("expected_action_space"), "name", str(spec.get("expected_action_space"))),
            "expected_action_dim": spec.get("expected_action_dim"),
            "expected_obs_keys": spec.get("expected_obs_keys", []),
        }
        sidecar.write_text(__import__("json").dumps(payload, indent=2), encoding="utf-8")
