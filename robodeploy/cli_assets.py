"""Asset catalog list / resolve / info CLI."""

from __future__ import annotations

from pathlib import Path

from robodeploy.cli_helpers import print_json
from robodeploy.core.asset_loader import AssetLoader


def cmd_assets_list(
    *,
    robot: bool,
    mesh: bool,
    mjcf: bool,
    as_json: bool,
    pretty: bool,
) -> int:
    loader = AssetLoader()
    kind = None
    if robot:
        kind = "robot"
    elif mesh:
        kind = "mesh"
    elif mjcf:
        kind = "mjcf"
    items = loader.list_all(kind=kind)
    if as_json:
        print_json(
            [{"name": i.name, "kind": i.kind, "path": i.path, "dof": i.dof, "notes": i.notes} for i in items],
            pretty=pretty,
        )
    else:
        for item in items:
            extra = f" ({item.path})" if item.path else ""
            dof = f" dof={item.dof}" if item.dof is not None else ""
            print(f"{item.kind}: {item.name}{dof}{extra}")
    return 0


def cmd_assets_resolve(*, name: str, backend: str, as_json: bool, pretty: bool) -> int:
    loader = AssetLoader()
    resolved = loader.resolve(name, backend=backend)
    payload = {"name": name, "backend": backend, "path": resolved}
    if as_json:
        print_json(payload, pretty=pretty)
    else:
        print(resolved or "(not found)")
    return 0 if resolved else 1


def cmd_assets_info(*, name: str, as_json: bool, pretty: bool) -> int:
    loader = AssetLoader()
    info = loader.info(name)
    if info is None:
        if as_json:
            print_json({"name": name, "found": False}, pretty=pretty)
        else:
            print(f"Unknown asset: {name}")
        return 1
    payload = {
        "name": info.name,
        "kind": info.kind,
        "path": info.path,
        "dof": info.dof,
        "notes": info.notes,
    }
    if as_json:
        print_json(payload, pretty=pretty)
    else:
        for key, value in payload.items():
            if value:
                print(f"{key}: {value}")
    return 0
