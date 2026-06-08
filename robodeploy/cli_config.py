"""Config show / resolve / validate / diff CLI."""

from __future__ import annotations

from pathlib import Path

from robodeploy.cli_helpers import print_json
from robodeploy.presets_loader import diff_presets, resolve_preset, validate_presets_file


def _default_presets_file() -> Path:
    repo = Path(__file__).resolve().parents[1]
    return repo / "examples" / "config" / "presets.yaml"


def cmd_config_show(*, preset: str, presets_file: Path | None, as_json: bool, pretty: bool) -> int:
    path = presets_file or _default_presets_file()
    cfg = resolve_preset(preset, presets_file=path)
    payload = cfg.to_dict()
    if as_json:
        print_json(payload, pretty=pretty)
    else:
        for key, value in sorted(payload.items()):
            print(f"{key}: {value}")
    return 0


def cmd_config_resolve(*, preset: str, presets_file: Path | None, as_json: bool, pretty: bool) -> int:
    path = presets_file or _default_presets_file()
    cfg = resolve_preset(preset, presets_file=path)
    payload = cfg.resolve()
    if as_json:
        print_json(payload, pretty=pretty)
    else:
        for key, value in sorted(payload.items()):
            print(f"{key}: {value}")
    return 0


def cmd_config_validate(*, presets_file: str, as_json: bool, pretty: bool) -> int:
    path = Path(presets_file)
    errors = validate_presets_file(path)
    payload = {"file": str(path), "ok": not errors, "errors": errors}
    if as_json:
        print_json(payload, pretty=pretty)
    else:
        if errors:
            for err in errors:
                print(f"ERROR: {err}")
        else:
            print(f"OK: {path}")
    return 0 if not errors else 1


def cmd_config_diff(
    *,
    preset_a: str,
    preset_b: str,
    presets_file: Path | None,
    as_json: bool,
    pretty: bool,
) -> int:
    path = presets_file or _default_presets_file()
    diff = diff_presets(preset_a, preset_b, presets_file=path)
    payload = {"a": preset_a, "b": preset_b, "diff": diff}
    if as_json:
        print_json(payload, pretty=pretty)
    else:
        if not diff:
            print(f"No differences between {preset_a} and {preset_b}")
        else:
            for key, (va, vb) in diff.items():
                print(f"{key}:")
                print(f"  {preset_a}: {va}")
                print(f"  {preset_b}: {vb}")
    return 0
