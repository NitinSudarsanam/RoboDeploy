"""Load SceneSpec from YAML for CLI scene validate/inspect."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from robodeploy.core.types import GeomSpec, MaterialSpec, PropConfig, SceneSpec, TerrainSpec, WorldSpec


def _tuple3(values: Any, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if not values:
        return default
    seq = list(values)
    if len(seq) < 3:
        raise ValueError(f"Expected 3 floats, got {values!r}")
    return (float(seq[0]), float(seq[1]), float(seq[2]))


def _tuple4(values: Any, default: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    if not values:
        return default
    seq = list(values)
    if len(seq) < 4:
        raise ValueError(f"Expected 4 floats, got {values!r}")
    return (float(seq[0]), float(seq[1]), float(seq[2]), float(seq[3]))


def load_scene_yaml(path: Path | str) -> SceneSpec:
    """Parse a YAML scene file into ``SceneSpec``."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Scene YAML must be a mapping: {path}")

    props: list[PropConfig] = []
    for entry in data.get("props", []) or []:
        if not isinstance(entry, dict):
            raise ValueError(f"Each prop must be a mapping: {entry!r}")
        geom_data = entry.get("geom")
        geom = None
        if geom_data:
            size = tuple(float(v) for v in geom_data.get("size", ()))
            geom = GeomSpec(
                kind=str(geom_data["kind"]),
                size=size,
                mesh_path=geom_data.get("mesh_path"),
            )
        mat_data = entry.get("material") or {}
        material = MaterialSpec(
            rgba=_tuple4(mat_data.get("rgba"), (0.8, 0.2, 0.2, 1.0)),
        )
        friction_dist = entry.get("friction_dist")
        props.append(
            PropConfig(
                name=str(entry["name"]),
                asset_path=str(entry.get("asset_path", "")),
                position=_tuple3(entry.get("position"), (0.0, 0.0, 0.0)),
                orientation=_tuple4(entry.get("orientation"), (1.0, 0.0, 0.0, 0.0)),
                mass=float(entry.get("mass", 0.1)),
                is_fixed=bool(entry.get("is_fixed", False)),
                geom=geom,
                material=material,
                collision_layer=int(entry.get("collision_layer", 0)),
                collision_mask=int(entry.get("collision_mask", 0xFFFF)),
                friction_dist=tuple(float(v) for v in friction_dist) if friction_dist else None,
            )
        )

    terrain_data = data.get("terrain") or {}
    terrain = TerrainSpec(
        kind=str(terrain_data.get("kind", "flat")),
        size=(
            float(terrain_data.get("size", [4.0, 4.0])[0]),
            float(terrain_data.get("size", [4.0, 4.0])[1]),
        ),
        heightfield_path=terrain_data.get("heightfield_path"),
        procedural_params=terrain_data.get("procedural_params"),
    )

    return SceneSpec(
        props=props,
        table_height=float(data.get("table_height", 0.0)),
        lighting=str(data.get("lighting", "default")),
        world=WorldSpec(terrain=terrain),
    )
