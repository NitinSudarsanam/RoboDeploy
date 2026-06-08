"""Asset catalog — robots, meshes, and format resolution."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from robodeploy.core.spaces import AssetFormat

_PKG_ROOT = Path(__file__).resolve().parents[1]
_DESCRIPTION_DIR = _PKG_ROOT / "description"
_DEFAULT_SEARCH = (
    _DESCRIPTION_DIR,
    _PKG_ROOT.parent / "examples" / "assets",
)
_DEFAULT_PRIORITY = (AssetFormat.MJCF, AssetFormat.URDF, AssetFormat.USD)
_BACKEND_FORMAT = {
    "mujoco": AssetFormat.MJCF,
    "gazebo": AssetFormat.URDF,
    "isaacsim": AssetFormat.USD,
    "isaac": AssetFormat.USD,
    "ros2": AssetFormat.URDF,
}


@dataclass
class AssetInfo:
    name: str
    kind: Literal["robot", "mesh", "mjcf", "urdf", "catalog"]
    path: str | None = None
    dof: int | None = None
    notes: str = ""


class AssetLoader:
    """Index built-in robot descriptions and example catalog assets."""

    def __init__(
        self,
        *,
        catalog_file: Path | str | None = None,
        search_paths: list[Path | str] | None = None,
        format_priority: list[AssetFormat] | None = None,
    ) -> None:
        self._catalog_file = catalog_file
        self._paths = [Path(p) for p in (search_paths or _DEFAULT_SEARCH)]
        self._priority = list(format_priority or _DEFAULT_PRIORITY)

    def list_robots(self) -> list[str]:
        names: list[str] = []
        if _DESCRIPTION_DIR.is_dir():
            for child in sorted(_DESCRIPTION_DIR.iterdir()):
                if child.is_dir() and (child / "description.py").is_file():
                    names.append(child.name)
        try:
            from examples.catalog.load import list_robots

            for name in list_robots(catalog_file=self._catalog_path()):
                if name not in names:
                    names.append(name)
        except Exception:
            pass
        return sorted(names)

    def list_meshes(self) -> list[str]:
        meshes: list[str] = []
        roots = [p for p in self._paths if p.is_dir()]
        if _DESCRIPTION_DIR.is_dir() and _DESCRIPTION_DIR not in roots:
            roots.append(_DESCRIPTION_DIR)
        for root in roots:
            for path in sorted(root.rglob("*.stl")):
                try:
                    meshes.append(str(path.relative_to(_PKG_ROOT)))
                except ValueError:
                    meshes.append(str(path))
            for path in sorted(root.rglob("*.obj")):
                try:
                    meshes.append(str(path.relative_to(_PKG_ROOT)))
                except ValueError:
                    meshes.append(str(path))
        return sorted(set(meshes))

    def list_all(self, *, kind: str | None = None) -> list[AssetInfo]:
        items: list[AssetInfo] = []
        if kind in (None, "robot"):
            for name in self.list_robots():
                items.append(AssetInfo(name=name, kind="robot", notes=self._robot_notes(name)))
        if kind in (None, "mesh"):
            for rel in self.list_meshes():
                items.append(AssetInfo(name=rel, kind="mesh", path=rel))
        if kind in (None, "mjcf"):
            for path in sorted(_DESCRIPTION_DIR.rglob("*.xml")):
                rel = str(path.relative_to(_PKG_ROOT))
                items.append(AssetInfo(name=path.stem, kind="mjcf", path=rel))
        if kind in (None, "urdf"):
            for path in sorted(_DESCRIPTION_DIR.rglob("*.urdf")):
                rel = str(path.relative_to(_PKG_ROOT))
                items.append(AssetInfo(name=path.stem, kind="urdf", path=rel))
        return items

    def catalog(self) -> list[dict[str, Any]]:
        """List all known assets with formats present."""
        rows: list[dict[str, Any]] = []
        for item in self.list_all():
            formats: list[str] = []
            if item.path:
                suffix = Path(item.path).suffix.lower()
                if suffix == ".xml":
                    formats.append("mjcf")
                elif suffix == ".urdf":
                    formats.append("urdf")
                elif suffix in (".usd", ".usda", ".usdc"):
                    formats.append("usd")
                elif suffix in (".stl", ".obj", ".dae"):
                    formats.append("mesh")
            for fmt in self._formats_for_name(item.name):
                if fmt not in formats:
                    formats.append(fmt)
            rows.append(
                {
                    "name": item.name,
                    "kind": item.kind,
                    "path": item.path,
                    "formats": formats,
                    "dof": item.dof,
                    "notes": item.notes,
                }
            )
        return rows

    def resolve(self, name: str, *, backend: str = "mujoco") -> str | None:
        """Find best matching asset variant for backend. Auto-convert if needed."""
        preferred = _backend_format(backend)
        direct = self._resolve_format(name, preferred)
        if direct is not None:
            return direct

        for fmt in self._priority:
            if fmt == preferred:
                continue
            path = self._resolve_format(name, fmt)
            if path is None:
                continue
            converted = self._auto_convert(path, fmt, preferred)
            if converted is not None:
                return converted
        return None

    def info(self, name: str) -> AssetInfo | None:
        desc_path = _DESCRIPTION_DIR / name / "description.py"
        if desc_path.is_file():
            dof = self._robot_dof(name)
            return AssetInfo(
                name=name,
                kind="robot",
                path=str(desc_path.relative_to(_PKG_ROOT)),
                dof=dof,
                notes=self._robot_notes(name),
            )
        resolved = self.resolve(name)
        if resolved:
            kind: Literal["robot", "mesh", "mjcf", "urdf", "catalog"] = "mesh"
            if resolved.endswith(".xml"):
                kind = "mjcf"
            elif resolved.endswith(".urdf"):
                kind = "urdf"
            return AssetInfo(name=name, kind=kind, path=resolved)
        return None

    def _formats_for_name(self, name: str) -> list[str]:
        formats: list[str] = []
        robot_dir = _DESCRIPTION_DIR / name / "assets"
        if robot_dir.is_dir():
            for sub in robot_dir.iterdir():
                if sub.is_dir():
                    formats.append(sub.name)
        for root in self._paths:
            if not root.is_dir():
                continue
            for ext, fmt in ((".xml", "mjcf"), (".urdf", "urdf"), (".usd", "usd"), (".usda", "usd")):
                if list(root.rglob(f"*{name}*{ext}")):
                    formats.append(fmt)
        return sorted(set(formats))

    def _resolve_format(self, name: str, fmt: AssetFormat) -> str | None:
        robot_dir = _DESCRIPTION_DIR / name / "assets" / fmt.value
        if robot_dir.is_dir():
            for ext in (".xml", ".urdf", ".usd", ".usda", ".usdc"):
                matches = list(robot_dir.glob(f"*{ext}"))
                if matches:
                    return str(matches[0].relative_to(_PKG_ROOT))
        for root in self._paths:
            if not root.is_dir():
                continue
            for ext in (".xml", ".urdf", ".usd", ".usda", ".usdc", ".stl", ".obj", ".dae"):
                matches = list(root.rglob(f"*{name}*{ext}"))
                if matches:
                    try:
                        return str(matches[0].relative_to(_PKG_ROOT))
                    except ValueError:
                        return str(matches[0])
        mesh_candidates = list(_DESCRIPTION_DIR.rglob(f"*{name}*"))
        for path in mesh_candidates:
            if path.suffix.lower() in (".stl", ".obj", ".dae"):
                return str(path.relative_to(_PKG_ROOT))
        return None

    def _auto_convert(self, rel_path: str, source_fmt: AssetFormat, target_fmt: AssetFormat) -> str | None:
        if source_fmt == AssetFormat.URDF and target_fmt == AssetFormat.MJCF:
            return self._urdf_to_mjcf(rel_path)
        return None

    def _urdf_to_mjcf(self, rel_path: str) -> str | None:
        """Compile URDF to cached MJCF via MuJoCo (simple robots only)."""
        src = (_PKG_ROOT / rel_path).resolve()
        if not src.is_file():
            src = Path(rel_path).resolve()
        if not src.is_file():
            return None
        try:
            import mujoco

            model = mujoco.MjModel.from_xml_path(str(src))
        except Exception:
            return None
        cache_dir = Path(tempfile.gettempdir()) / "robodeploy" / "asset_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        out = cache_dir / f"{src.stem}_from_urdf.xml"
        try:
            mujoco.mj_saveLastXML(str(out), model)
        except Exception:
            return None
        return str(out) if out.is_file() else None

    def _catalog_path(self) -> Path | None:
        if self._catalog_file is not None:
            return Path(self._catalog_file)
        repo_catalog = _PKG_ROOT.parent / "examples" / "catalog" / "mujoco_catalog.yaml"
        return repo_catalog if repo_catalog.is_file() else None

    def _robot_notes(self, name: str) -> str:
        try:
            from examples.catalog.load import get_robot

            path = self._catalog_path()
            if path is None:
                return ""
            return str(get_robot(name, catalog_file=path).get("notes", ""))
        except Exception:
            return ""

    def _robot_dof(self, name: str) -> int | None:
        try:
            from robodeploy.builtins import import_builtins
            from robodeploy.core.registry import get_robot

            import_builtins()
            desc = get_robot(name)()
            return int(getattr(desc, "dof", 0) or 0)
        except Exception:
            return None


def _backend_format(backend: str) -> AssetFormat:
    return _BACKEND_FORMAT.get(backend.lower(), AssetFormat.MJCF)
