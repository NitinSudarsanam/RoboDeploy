"""Pre-flight scene validation against backend capabilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from robodeploy.core.scene_ir import SceneIR, UnifiedPropSpec

IssueLevel = Literal["error", "warning", "info"]

_BACKEND_FORMAT = {
    "mujoco": "mjcf",
    "gazebo": "urdf",
    "isaacsim": "usd",
    "isaac": "usd",
}

_SUPPORTED_GEOMS: dict[str, set[str]] = {
    "mujoco": {"box", "sphere", "cylinder", "capsule", "mesh", "plane", "heightfield"},
    "gazebo": {"box", "sphere", "cylinder", "mesh", "plane"},
    "isaacsim": {"box", "sphere", "cylinder", "capsule", "mesh", "plane"},
    "isaac": {"box", "sphere", "cylinder", "capsule", "mesh", "plane"},
}


@dataclass
class ValidationIssue:
    level: IssueLevel
    message: str
    prop_name: str | None = None
    suggested_fix: str | None = None
    line: int | None = None


@dataclass
class ValidationReport:
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(i.level == "error" for i in self.issues)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.level == "warning"]


class SceneValidationError(ValueError):
    def __init__(self, report: ValidationReport) -> None:
        self.report = report
        msgs = "; ".join(i.message for i in report.errors)
        super().__init__(f"Scene validation failed: {msgs}")


class SceneValidator:
    def validate_spec(self, spec, *, backend_name: str | None = None) -> ValidationReport:
        """Validate a legacy ``SceneSpec``."""
        report = ValidationReport()
        names: set[str] = set()
        for prop in [*getattr(spec, "props", []), *getattr(getattr(spec, "world", None), "props", [])]:
            if prop.name in names:
                report.issues.append(
                    ValidationIssue(
                        level="error",
                        message=f"Duplicate prop name '{prop.name}'",
                        prop_name=prop.name,
                        suggested_fix="Use unique prop names.",
                    )
                )
            names.add(prop.name)
        ir_report = self.validate(spec.to_ir(), backend_name or "mujoco")
        report.issues.extend(ir_report.issues)
        return report

    def validate(self, ir: SceneIR, backend_name: str) -> ValidationReport:
        report = ValidationReport()
        backend = str(backend_name).lower()
        names: set[str] = set()
        for prop in ir.props:
            if prop.name in names:
                report.issues.append(
                    ValidationIssue(
                        level="error",
                        message=f"Duplicate prop name '{prop.name}'",
                        prop_name=prop.name,
                        suggested_fix="Use unique prop names.",
                    )
                )
            names.add(prop.name)
            self._validate_prop(prop, backend, names, report)
        return report

    def _validate_prop(
        self,
        prop: UnifiedPropSpec,
        backend: str,
        names: set[str],
        report: ValidationReport,
    ) -> None:
        kind = prop.geometry.kind
        supported = _SUPPORTED_GEOMS.get(backend, _SUPPORTED_GEOMS["mujoco"])
        if kind not in supported:
            report.issues.append(
                ValidationIssue(
                    level="warning" if kind == "capsule" and backend == "gazebo" else "error",
                    message=f"Geom kind '{kind}' not natively supported on backend '{backend}'",
                    prop_name=prop.name,
                    suggested_fix=f"Use a supported geom or provide a {backend} asset variant.",
                )
            )

        if prop.parent_frame is not None and prop.parent_frame not in names:
            report.issues.append(
                ValidationIssue(
                    level="error",
                    message=f"parent_frame '{prop.parent_frame}' not found among props",
                    prop_name=prop.name,
                    suggested_fix="Declare the parent prop before the child.",
                )
            )

        if not prop.physics.is_fixed and prop.physics.mass <= 0.0:
            report.issues.append(
                ValidationIssue(
                    level="error",
                    message="Non-fixed prop must have mass > 0",
                    prop_name=prop.name,
                    suggested_fix="Set mass > 0 or mark is_fixed=True.",
                )
            )

        slide = prop.physics.friction[0] if prop.physics.friction else 1.0
        if slide < 0.1 or slide > 2.0:
            report.issues.append(
                ValidationIssue(
                    level="warning",
                    message=f"Friction slide={slide} outside typical [0.1, 2.0]",
                    prop_name=prop.name,
                )
            )

        if prop.physics.collision_mask == 0:
            report.issues.append(
                ValidationIssue(
                    level="warning",
                    message="collision_mask=0 disables all collisions",
                    prop_name=prop.name,
                )
            )

        if kind == "heightfield" and prop.geometry.heightfield_path:
            if not Path(prop.geometry.heightfield_path).exists():
                report.issues.append(
                    ValidationIssue(
                        level="error",
                        message=f"Heightfield path does not exist: {prop.geometry.heightfield_path}",
                        prop_name=prop.name,
                    )
                )

        if kind == "mesh" and prop.geometry.mesh_path:
            if not Path(prop.geometry.mesh_path).exists():
                report.issues.append(
                    ValidationIssue(
                        level="warning",
                        message=f"Mesh path does not exist: {prop.geometry.mesh_path}",
                        prop_name=prop.name,
                    )
                )

        preferred = _BACKEND_FORMAT.get(backend)
        if preferred and prop.geometry.kind == "mesh":
            has_variant = preferred in prop.variants
            has_mesh = bool(prop.geometry.mesh_path)
            if not has_variant and not has_mesh:
                report.issues.append(
                    ValidationIssue(
                        level="info",
                        message=f"No {preferred} variant or mesh_path for mesh prop",
                        prop_name=prop.name,
                        suggested_fix=f"Add variants['{preferred}'] or mesh_path.",
                    )
                )
