"""Cartesian reach policy for PickPlaceTask (YAML-driven reach DSL)."""

from __future__ import annotations

from pathlib import Path

import yaml

from robodeploy.core.registry import register_policy
from robodeploy.core.types import SceneSpec
from robodeploy.policies.reach_dsl import ReachTrajectoryPolicy

_DEFAULT_YAML = Path(__file__).resolve().parent / "reach_pick_place.yaml"


def _load_spec(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return next(iter(raw.values())) if isinstance(raw, dict) and len(raw) == 1 else raw


@register_policy("example_reach_pick")
class ReachPickPlacePolicy(ReachTrajectoryPolicy):
    """Thin shim loading ``reach_pick_place.yaml`` (all tuning lives in YAML)."""

    def __init__(
        self,
        *,
        scene: SceneSpec | None = None,
        description=None,
        config: dict | None = None,
        yaml_path: str | Path | None = None,
        **kwargs: object,
    ) -> None:
        del kwargs
        spec = _load_spec(Path(yaml_path) if yaml_path else _DEFAULT_YAML)
        super().__init__(spec, scene=scene, description=description, config=config)
