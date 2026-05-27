"""Resolve SO-101 URDF for sim / ROS: optional mesh fallback when STL files are absent."""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path

_LOG = logging.getLogger(__name__)

_PLACEHOLDER_BOX = ("0.02", "0.02", "0.02")


def _urdf_dir(canonical_urdf: Path) -> Path:
    return canonical_urdf.resolve().parent


def _resolve_mesh_path(canonical_urdf: Path, mesh_filename: str) -> Path:
    fn = (mesh_filename or "").strip()
    if fn.startswith("package://"):
        # Strip `package://pkg/` — not used by bundled SO-101 URDF, but be defensive.
        rest = fn[len("package://") :]
        slash = rest.find("/")
        if slash >= 0:
            fn = rest[slash + 1 :]
    return (_urdf_dir(canonical_urdf) / fn).resolve()


def _collect_mesh_filenames(root: ET.Element) -> list[str]:
    out: list[str] = []
    for mesh in root.iter("mesh"):
        fn = mesh.attrib.get("filename", "").strip()
        if fn:
            out.append(fn)
    return out


def _any_mesh_missing(canonical_urdf: Path, root: ET.Element) -> tuple[bool, list[str]]:
    missing: list[str] = []
    seen: set[str] = set()
    for fn in _collect_mesh_filenames(root):
        if fn in seen:
            continue
        seen.add(fn)
        p = _resolve_mesh_path(canonical_urdf, fn)
        if not p.is_file():
            missing.append(fn)
    return (len(missing) > 0, missing)


def _replace_missing_mesh_geometries(canonical_urdf: Path, root: ET.Element) -> None:
    """In-place: replace `<geometry><mesh/></geometry>` with a small box when the STL is missing."""
    for link in root.findall("link"):
        for tag in ("visual", "collision"):
            for vc in link.findall(tag):
                geom = vc.find("geometry")
                if geom is None:
                    continue
                mesh_el = geom.find("mesh")
                if mesh_el is None:
                    continue
                fn = mesh_el.attrib.get("filename", "").strip()
                if not fn:
                    continue
                resolved = _resolve_mesh_path(canonical_urdf, fn)
                if resolved.is_file():
                    continue
                for child in list(geom):
                    geom.remove(child)
                ET.SubElement(geom, "box", {"size": " ".join(_PLACEHOLDER_BOX)})


def _rewrite_mesh_filenames_to_abs(canonical_urdf: Path, root: ET.Element) -> None:
    """In-place: rewrite all `<mesh filename=.../>` to absolute paths.

    Some URDF consumers (notably MuJoCo's URDF importer) may ignore path prefixes
    and only look up the basename. Using absolute paths makes resolution robust.
    """
    for mesh in root.iter("mesh"):
        fn = mesh.attrib.get("filename", "").strip()
        if not fn:
            continue
        p = _resolve_mesh_path(canonical_urdf, fn)
        # RViz uses `resource_retriever` which is most reliable with file:// URIs.
        # MuJoCo is also fine with forward slashes on Windows.
        mesh.attrib["filename"] = f"file://{p.as_posix()}"


def _copy_meshes_next_to_urdf(canonical_urdf: Path, root: ET.Element, *, out_dir: Path) -> None:
    """Best-effort: copy mesh files next to the generated URDF by basename.

    MuJoCo's URDF importer may ignore directory prefixes and attempt to open only the basename
    relative to the URDF directory. Copying meshes alongside the cached URDF makes this robust
    without mutating the canonical URDF.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for mesh in root.iter("mesh"):
        fn = mesh.attrib.get("filename", "").strip()
        if not fn:
            continue
        src = _resolve_mesh_path(canonical_urdf, fn)
        if not src.is_file():
            continue
        dst = (out_dir / src.name)
        if dst.is_file():
            continue
        try:
            dst.write_bytes(src.read_bytes())
        except Exception as exc:
            raise OSError(f"Failed to copy SO-101 mesh '{src}' to '{dst}'.") from exc


def resolve_urdf_with_mesh_fallback(canonical_urdf: Path) -> Path:
    """Return a URDF path safe for MuJoCo / RViz.

    If every ``<mesh filename=...>`` resolves on disk next to the URDF, returns
    ``canonical_urdf``. Otherwise writes a cached copy with missing mesh
    geometries replaced by small boxes (same tree otherwise) under
    ``~/.robodeploy/so101/`` and returns that path.
    """
    canonical_urdf = Path(canonical_urdf).resolve()
    if not canonical_urdf.is_file():
        raise FileNotFoundError(f"URDF not found: {canonical_urdf}")

    root = ET.parse(str(canonical_urdf)).getroot()
    any_missing, missing_list = _any_mesh_missing(canonical_urdf, root)

    if any_missing:
        _LOG.warning(
            "SO-101 URDF mesh fallback: %d STL file(s) missing under %s: %s. "
            "Using placeholder box geometry (cached). Drop STLs into assets/urdf/assets/ to restore meshes.",
            len(missing_list),
            _urdf_dir(canonical_urdf),
            ", ".join(missing_list),
        )
        _replace_missing_mesh_geometries(canonical_urdf, root)

    # Always rewrite existing mesh filenames to absolute paths for robust runtime loading.
    _rewrite_mesh_filenames_to_abs(canonical_urdf, root)

    mtime_key = int(canonical_urdf.stat().st_mtime)
    cache_dir = Path.home() / ".robodeploy" / "so101"
    cache_dir.mkdir(parents=True, exist_ok=True)
    suffix = "stripped" if any_missing else "resolved"
    out_path = cache_dir / f"so101_{suffix}_{mtime_key}.urdf"

    xml_body = ET.tostring(root, encoding="unicode")
    out_path.write_text('<?xml version="1.0" encoding="UTF-8"?>\n' + xml_body, encoding="utf-8")

    # Ensure meshes are also reachable via basename next to the cached URDF.
    _copy_meshes_next_to_urdf(canonical_urdf, root, out_dir=out_path.parent)
    return out_path
