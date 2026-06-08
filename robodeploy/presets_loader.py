"""Optional YAML preset resolution (requires ``examples/`` on PYTHONPATH)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from robodeploy.core.env_config import EnvConfig


def _default_presets_path() -> Path:
    return Path(__file__).resolve().parents[1] / "examples" / "config" / "presets.yaml"


def _examples_config():
    try:
        from examples.config import (  # noqa: PLC0415
            _is_listable_preset,
            _load_merged_yaml,
            load_preset,
        )
    except ImportError as exc:
        raise ImportError(
            "Preset loading requires the RoboDeploy examples package on PYTHONPATH."
        ) from exc
    return load_preset, _load_merged_yaml, _is_listable_preset


def resolve_preset(name: str, *, presets_file=None) -> EnvConfig:
    load_preset, _, _ = _examples_config()
    path = presets_file or _default_presets_path()
    return EnvConfig.from_dict(load_preset(name, presets_file=path))


def validate_presets_file(path) -> list[str]:
    from pathlib import Path

    try:
        import yaml
    except ImportError:
        return ["PyYAML is required to validate presets"]

    preset_path = Path(path)
    if not preset_path.is_file():
        return [f"Presets file not found: {preset_path}"]

    try:
        _, _load_merged_yaml, _is_listable_preset = _examples_config()
        data = _load_merged_yaml(preset_path)
    except ImportError as exc:
        return [str(exc)]
    except Exception as exc:
        return [str(exc)]

    if not isinstance(data, dict):
        return ["Presets YAML root must be a mapping"]

    errors: list[str] = []
    single_required = ("robot", "backend", "task", "policy")
    robot_entry_required = ("robot", "task", "policy")
    for preset_name, preset in data.items():
        if not _is_listable_preset(str(preset_name), preset if isinstance(preset, dict) else {}):
            if str(preset_name).startswith("_") or str(preset_name).startswith("base_"):
                continue
            if not isinstance(preset, dict):
                continue
        if not isinstance(preset, dict):
            errors.append(f"{preset_name}: preset must be a mapping")
            continue
        if "robots" in preset:
            if "backend" not in preset:
                errors.append(f"{preset_name}: missing keys ['backend']")
            robots = preset.get("robots")
            if not isinstance(robots, list) or not robots:
                errors.append(f"{preset_name}: 'robots' must be a non-empty list")
                continue
            for idx, entry in enumerate(robots):
                if not isinstance(entry, dict):
                    errors.append(f"{preset_name}: robots[{idx}] must be a mapping")
                    continue
                missing = [key for key in robot_entry_required if key not in entry]
                if missing:
                    errors.append(f"{preset_name}: robots[{idx}] missing keys {missing}")
            continue
        missing = [key for key in single_required if key not in preset]
        if missing:
            errors.append(f"{preset_name}: missing keys {missing}")
    return errors


def diff_presets(preset_a: str, preset_b: str, *, presets_file=None) -> dict[str, tuple[Any, Any]]:
    path = presets_file or _default_presets_path()
    a = resolve_preset(preset_a, presets_file=path).to_dict()
    b = resolve_preset(preset_b, presets_file=path).to_dict()
    keys = sorted(set(a) | set(b))
    out: dict[str, tuple[Any, Any]] = {}
    for key in keys:
        va, vb = a.get(key), b.get(key)
        if va != vb:
            out[key] = (va, vb)
    return out
