"""Example preset YAML — lives outside the robodeploy package."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PRESETS_FILE = Path(__file__).with_name("presets.yaml")
PRESETS_DIR = Path(__file__).resolve().parents[1] / "presets"

_REQUIRED_PRESET_KEYS = ("robot", "backend", "task", "policy")
_MULTI_ROBOT_PRESET_KEYS = ("backend", "robots")
_ANCHOR_PREFIXES = ("_", "base_")


def _preset_validation_error(values: dict[str, Any]) -> list[str]:
    if "robots" in values:
        missing = [key for key in _MULTI_ROBOT_PRESET_KEYS if key not in values]
        robots = values.get("robots")
        if not isinstance(robots, list) or not robots:
            missing.append("robots (non-empty list)")
        return missing
    return [key for key in _REQUIRED_PRESET_KEYS if key not in values]


def _is_valid_preset(values: dict[str, Any]) -> bool:
    return not _preset_validation_error(values)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key == "include":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_includes(data: dict[str, Any], *, base_dir: Path) -> dict[str, Any]:
    includes = data.pop("include", None)
    if not includes:
        return dict(data)
    paths = [includes] if isinstance(includes, str) else list(includes)
    merged: dict[str, Any] = {}
    for rel in paths:
        inc_path = (base_dir / str(rel)).resolve()
        if not inc_path.is_file():
            raise FileNotFoundError(f"Preset include not found: {inc_path}")
        fragment = yaml.safe_load(inc_path.read_text(encoding="utf-8")) or {}
        if isinstance(fragment, dict):
            merged = _deep_merge(merged, fragment)
    return _deep_merge(merged, data)


def _is_listable_preset(name: str, values: dict[str, Any]) -> bool:
    if any(name.startswith(prefix) for prefix in _ANCHOR_PREFIXES):
        return False
    return _is_valid_preset(values)


def _peek_include_list(path: Path) -> list[str]:
    includes: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("- ") and includes is not None:
            includes.append(stripped[2:].strip())
            continue
        if stripped == "include:":
            includes = []
            continue
        if includes is not None and stripped and not stripped.startswith("#"):
            break
    return includes


def _load_merged_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    include_paths = _peek_include_list(path)
    if not include_paths:
        data = yaml.safe_load(text) or {}
        return data if isinstance(data, dict) else {}

    prefix = ""
    for rel in include_paths:
        inc_path = (path.parent / str(rel)).resolve()
        if not inc_path.is_file():
            raise FileNotFoundError(f"Preset include not found: {inc_path}")
        prefix += inc_path.read_text(encoding="utf-8") + "\n"

    main_lines = []
    skipping = False
    for line in text.splitlines():
        if line.strip() == "include:":
            skipping = True
            continue
        if skipping:
            if line.strip().startswith("- "):
                continue
            skipping = False
        if not skipping:
            main_lines.append(line)
    merged = yaml.safe_load(prefix + "\n".join(main_lines)) or {}
    return merged if isinstance(merged, dict) else {}


@lru_cache(maxsize=16)
def _load_all_presets(presets_file: str) -> dict[str, dict[str, Any]]:
    path = Path(presets_file)
    if not path.is_file():
        return {}
    data = _load_merged_yaml(path)
    presets: dict[str, dict[str, Any]] = {}
    for name, values in data.items():
        if not isinstance(values, dict):
            continue
        resolved = _resolve_includes(dict(values), base_dir=path.parent)
        presets[str(name)] = resolved
    return presets


def list_presets(*, presets_file: Path | str) -> list[str]:
    all_presets = _load_all_presets(str(presets_file))
    return sorted(name for name, values in all_presets.items() if _is_listable_preset(name, values))


def load_preset(name: str, *, presets_file: Path | str) -> dict[str, Any]:
    presets = _load_all_presets(str(presets_file))
    if name not in presets:
        known = ", ".join(list_presets(presets_file=presets_file)) or "(none)"
        raise KeyError(f"Unknown preset '{name}' in {presets_file}. Known presets: {known}")
    preset = dict(presets[name])
    missing = _preset_validation_error(preset)
    if missing:
        raise ValueError(f"Preset '{name}' missing required keys: {missing}")
    return preset


def list_example_presets() -> list[str]:
    return list_presets(presets_file=PRESETS_FILE)


def load_example_preset(name: str) -> dict[str, Any]:
    return load_preset(name, presets_file=PRESETS_FILE)
