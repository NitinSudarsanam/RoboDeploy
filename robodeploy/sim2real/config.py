"""Sim/real preset pairing and shared deployment configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from robodeploy.tasks.randomization import (
    DomainRandomizerConfig,
    dr_config_to_dict,
    resolve_domain_randomizer_config,
)

_DEFAULT_PAIRS_FILE = Path(__file__).resolve().parents[2] / "examples" / "config" / "sim2real_pairs.yaml"


@dataclass
class Sim2RealPair:
    """Named sim + real preset pair with fields shared across deployment."""

    name: str
    sim_preset: str
    real_preset: str
    shared: dict[str, Any] = field(default_factory=dict)
    domain_randomization: dict[str, Any] | None = None
    transfer_notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "sim_preset": self.sim_preset,
            "real_preset": self.real_preset,
        }
        if self.shared:
            payload["shared"] = dict(self.shared)
        if self.domain_randomization is not None:
            payload["domain_randomization"] = dict(self.domain_randomization)
        if self.transfer_notes:
            payload["transfer_notes"] = self.transfer_notes
        return payload


@lru_cache(maxsize=8)
def _load_pairs_file(pairs_file: str) -> dict[str, dict[str, Any]]:
    path = Path(pairs_file)
    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {str(name): dict(values) for name, values in data.items()}


def resolve_sim2real_pair(
    spec: str | dict[str, Any] | Sim2RealPair,
    *,
    pairs_file: Path | str | None = None,
) -> Sim2RealPair:
    """Resolve a pair name, inline dict, or ``Sim2RealPair`` instance."""
    if isinstance(spec, Sim2RealPair):
        return spec
    if isinstance(spec, str):
        return load_sim2real_pair(spec, pairs_file=pairs_file)
    if not isinstance(spec, dict):
        raise TypeError(f"Unsupported sim2real pair spec: {type(spec).__name__}")

    name = str(spec.get("name", "inline"))
    sim_preset = spec.get("sim_preset")
    real_preset = spec.get("real_preset")
    if not sim_preset or not real_preset:
        raise ValueError("sim2real pair dict requires sim_preset and real_preset")
    return Sim2RealPair(
        name=name,
        sim_preset=str(sim_preset),
        real_preset=str(real_preset),
        shared=dict(spec.get("shared") or {}),
        domain_randomization=spec.get("domain_randomization"),
        transfer_notes=str(spec.get("transfer_notes", "")),
    )


def load_sim2real_pair(
    name: str,
    *,
    pairs_file: Path | str | None = None,
) -> Sim2RealPair:
    """Load a named pair from ``examples/config/sim2real_pairs.yaml`` by default."""
    path = Path(pairs_file) if pairs_file else _DEFAULT_PAIRS_FILE
    pairs = _load_pairs_file(str(path))
    if name not in pairs:
        known = ", ".join(sorted(pairs)) or "(none)"
        raise KeyError(f"Unknown sim2real pair '{name}' in {path}. Known: {known}")
    raw = pairs[name]
    return Sim2RealPair(
        name=name,
        sim_preset=str(raw["sim_preset"]),
        real_preset=str(raw["real_preset"]),
        shared=dict(raw.get("shared") or {}),
        domain_randomization=raw.get("domain_randomization"),
        transfer_notes=str(raw.get("transfer_notes", "")),
    )


def apply_shared_fields(base_cfg: dict[str, Any], pair: Sim2RealPair) -> dict[str, Any]:
    """Merge ``pair.shared`` into a preset config without overwriting preset keys."""
    merged = dict(base_cfg)
    for key, value in pair.shared.items():
        if key not in merged:
            merged[key] = value
            continue
        if isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = {**value, **merged[key]}
        else:
            merged[key] = value
    return merged


def pair_name_from_preset(preset_cfg: dict[str, Any]) -> str | None:
    """Read ``sim2real_pair`` metadata from a benchmark or deploy preset."""
    name = preset_cfg.get("sim2real_pair")
    return str(name).strip() if name else None


def load_pair_for_preset(
    preset_cfg: dict[str, Any],
    *,
    pairs_file: Path | str | None = None,
) -> Sim2RealPair | None:
    """Resolve the named sim2real pair referenced by a preset, if any."""
    name = pair_name_from_preset(preset_cfg)
    if not name:
        return None
    return load_sim2real_pair(name, pairs_file=pairs_file)


def merge_preset_with_dr(
    preset_cfg: dict[str, Any],
    dr_spec: DomainRandomizerConfig | dict[str, Any] | None,
    *,
    pair: Sim2RealPair | None = None,
) -> dict[str, Any]:
    """Attach domain randomization to a preset config for sim training sweeps."""
    cfg = dict(preset_cfg)
    if pair is not None:
        cfg = apply_shared_fields(cfg, pair)
        if dr_spec is None and pair.domain_randomization is not None:
            dr_spec = pair.domain_randomization

    if dr_spec is None:
        return cfg

    resolved = resolve_domain_randomizer_config(dr_spec)
    if resolved is None:
        return cfg

    task_kwargs = dict(cfg.get("task_kwargs") or {})
    task_kwargs["domain_randomization"] = dr_config_to_dict(resolved)
    cfg["task_kwargs"] = task_kwargs
    return cfg
