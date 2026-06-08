"""Scene validate / inspect CLI."""

from __future__ import annotations

import json
from pathlib import Path

from robodeploy.cli_helpers import print_json
from robodeploy.core.scene_validator import SceneValidator
from robodeploy.core.scene_yaml import load_scene_yaml
from robodeploy.linter import format_report
from robodeploy.core.scene_validator import ValidationReport


def cmd_scene_validate(*, scene: str, backend: str | None, as_json: bool, pretty: bool) -> int:
    path = Path(scene)
    spec = load_scene_yaml(path)
    report = SceneValidator().validate_spec(spec, backend_name=backend)
    if as_json:
        print_json(_report_to_dict(report, scene=str(path), backend=backend), pretty=pretty)
    else:
        print(format_report(_validation_to_lint(report, str(path))))
    return 0 if report.ok else 1


def cmd_scene_inspect(*, scene: str, backend: str | None, as_json: bool, pretty: bool) -> int:
    path = Path(scene)
    spec = load_scene_yaml(path)
    report = SceneValidator().validate_spec(spec, backend_name=backend)
    world = spec.to_world()
    payload = {
        "scene": str(path),
        "backend": backend,
        "table_height": spec.table_height,
        "lighting": spec.lighting,
        "prop_count": len(world.props),
        "props": [
            {
                "name": p.name,
                "position": p.position,
                "geom": p.geom.kind if p.geom else None,
                "fixed": p.is_fixed,
                "mass": p.mass,
            }
            for p in world.props
        ],
        "validation": _report_to_dict(report, scene=str(path), backend=backend),
    }
    if as_json:
        print_json(payload, pretty=pretty)
    else:
        print(json.dumps(payload, indent=2 if pretty else None))
    return 0


def _report_to_dict(report: ValidationReport, *, scene: str, backend: str | None) -> dict:
    return {
        "scene": scene,
        "backend": backend,
        "ok": report.ok,
        "issues": [
            {
                "level": i.level,
                "message": i.message,
                "prop_name": i.prop_name,
                "suggested_fix": i.suggested_fix,
            }
            for i in report.issues
        ],
    }


def _validation_to_lint(report: ValidationReport, file: str):
    from robodeploy.linter import LintReport, LintIssue

    lint = LintReport()
    for issue in report.issues:
        lint.issues.append(
            LintIssue(
                level=issue.level,
                message=issue.message,
                file=file,
                line=getattr(issue, "line", None),
                suggested_fix=issue.suggested_fix,
            )
        )
    return lint
