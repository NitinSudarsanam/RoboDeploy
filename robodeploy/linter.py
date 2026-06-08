"""Static linter for tasks, policies, and presets."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

from robodeploy.presets_loader import validate_presets_file

IssueLevel = Literal["error", "warning", "info"]


@dataclass
class LintIssue:
    level: IssueLevel
    message: str
    file: str
    line: int | None = None
    suggested_fix: str | None = None


@dataclass
class LintReport:
    issues: list[LintIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(i.level == "error" for i in self.issues)

    def add(
        self,
        level: IssueLevel,
        message: str,
        *,
        file: str,
        line: int | None = None,
        suggested_fix: str | None = None,
    ) -> None:
        self.issues.append(
            LintIssue(
                level=level,
                message=message,
                file=file,
                line=line,
                suggested_fix=suggested_fix,
            )
        )


_TASK_METHODS = (
    "obs_spec",
    "scene_spec",
    "language_instruction",
    "reset_fn",
    "reward_fn",
    "success_fn",
)

_TASK_BASE_NAMES = frozenset(
    {
        "TaskBase",
        "PickPlaceTemplate",
        "PourTemplate",
        "InsertionTemplate",
        "PegInsertionTemplate",
    }
)

_TASK_TEMPLATE_BASES = frozenset(
    {
        "PickPlaceTemplate",
        "InsertionTemplate",
        "PourTemplate",
        "StackingTemplate",
    }
)

_POLICY_METHODS = ("get_action",)


def lint_task(path: Path | str) -> LintReport:
    return _lint_python_component(
        path,
        kind="task",
        register_decorator="register_task",
        base_class="TaskBase",
        required_methods=_TASK_METHODS,
    )


def lint_policy(path: Path | str) -> LintReport:
    file_path = Path(path)
    if file_path.suffix.lower() in (".yaml", ".yml"):
        return _lint_reach_dsl_yaml(file_path)
    report = _lint_python_component(
        path,
        kind="policy",
        register_decorator="register_policy",
        base_class="PolicyBase",
        required_methods=_POLICY_METHODS,
    )
    if report.ok:
        _check_action_space(file_path, report)
    return report


def lint_preset(
    path: Path | str,
    *,
    check: str | None = None,
) -> LintReport:
    report = LintReport()
    file_path = Path(path)
    for msg in validate_presets_file(file_path):
        report.add("error", msg, file=str(file_path))
    if check:
        try:
            from examples.config import _load_merged_yaml, list_presets

            data = _load_merged_yaml(file_path)
            known = list_presets(presets_file=file_path)
            if check not in known:
                report.add(
                    "error",
                    f"Preset '{check}' not found in {file_path}",
                    file=str(file_path),
                    suggested_fix=f"Known presets: {', '.join(known)}",
                )
        except Exception as exc:
            report.add("error", str(exc), file=str(file_path))
    return report


def lint_all(*, root: Path | str | None = None) -> LintReport:
    repo = Path(root) if root else Path(__file__).resolve().parents[1].parent
    report = LintReport()
    tasks_dir = repo / "examples" / "tasks"
    policies_dir = repo / "examples" / "policies"
    presets = repo / "examples" / "config" / "presets.yaml"
    for py in sorted(tasks_dir.glob("*.py")):
        if py.name.startswith("_"):
            continue
        sub = lint_task(py)
        report.issues.extend(sub.issues)
    for py in sorted(policies_dir.glob("*.py")):
        if py.name.startswith("_"):
            continue
        sub = lint_policy(py)
        report.issues.extend(sub.issues)
    if presets.is_file():
        sub = lint_preset(presets)
        report.issues.extend(sub.issues)
    return report


def format_report(report: LintReport) -> str:
    if not report.issues:
        return "No issues found."
    lines: list[str] = []
    for issue in report.issues:
        loc = f"{issue.file}"
        if issue.line is not None:
            loc += f":{issue.line}"
        prefix = issue.level.upper()
        line = f"[{prefix}] {loc} — {issue.message}"
        if issue.suggested_fix:
            line += f" (fix: {issue.suggested_fix})"
        lines.append(line)
    return "\n".join(lines)


def _lint_python_component(
    path: Path | str,
    *,
    kind: str,
    register_decorator: str,
    base_class: str,
    required_methods: tuple[str, ...],
) -> LintReport:
    report = LintReport()
    file_path = Path(path)
    if not file_path.is_file():
        report.add("error", f"File not found: {file_path}", file=str(file_path))
        return report

    source = file_path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError as exc:
        report.add("error", f"Syntax error: {exc}", file=str(file_path), line=exc.lineno)
        return report

    has_register = False
    task_classes: list[ast.ClassDef] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if _class_inherits(node, base_class) or (
                kind == "task" and _class_inherits_from_task_template(node)
            ):
                task_classes.append(node)
            for dec in node.decorator_list:
                if _decorator_is_register(dec, register_decorator):
                    has_register = True

    if not has_register:
        report.add(
            "error",
            f"Missing @{register_decorator} decorator",
            file=str(file_path),
            suggested_fix=f"Add @register_{kind}('my_{kind}') above your class.",
        )

    if not task_classes:
        report.add(
            "error",
            f"No class inheriting {base_class} found",
            file=str(file_path),
            suggested_fix=f"Subclass {base_class} and implement required methods.",
        )
        return report

    for cls in task_classes:
        if _class_inherits_from_task_template(cls):
            continue
        defined = {n.name for n in cls.body if isinstance(n, ast.FunctionDef)}
        for method in required_methods:
            if method not in defined:
                report.add(
                    "error",
                    f"Class '{cls.name}' missing method '{method}'",
                    file=str(file_path),
                    line=cls.lineno,
                    suggested_fix=f"Implement def {method}(self, ...).",
                )

    if "template_version" not in source:
        report.add(
            "warning",
            "Missing template_version comment (scaffold may be outdated)",
            file=str(file_path),
        )

    _check_deprecated_api(source, file_path, report)
    return report


def _lint_reach_dsl_yaml(path: Path) -> LintReport:
    report = LintReport()
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        report.add("error", str(exc), file=str(path))
        return report
    if not isinstance(data, dict):
        report.add("error", "Reach DSL YAML must be a mapping", file=str(path))
        return report
    for policy_name, spec in data.items():
        if not isinstance(spec, dict):
            report.add("error", f"Policy '{policy_name}' must be a mapping", file=str(path))
            continue
        phases = spec.get("phases")
        if not phases:
            report.add(
                "error",
                f"Policy '{policy_name}' missing 'phases'",
                file=str(path),
                suggested_fix="Add phases: [{name: pregrasp, ...}, ...]",
            )
            continue
        for phase in phases:
            if not isinstance(phase, dict) or "name" not in phase:
                report.add(
                    "error",
                    f"Each phase needs a 'name' field in '{policy_name}'",
                    file=str(path),
                )
    return report


def _check_action_space(path: Path, report: LintReport) -> None:
    source = path.read_text(encoding="utf-8")
    if "action_space" not in source and "ActionSpace" not in source:
        report.add(
            "error",
            "Policy must declare action_space (typically in __init__)",
            file=str(path),
            suggested_fix="Call super().__init__(action_space=ActionSpace.JOINT_POS, ...)",
        )


def _check_deprecated_api(source: str, path: Path, report: LintReport) -> None:
    if "has_prop_contact" in source:
        report.add(
            "warning",
            "Uses deprecated backend.has_prop_contact — prefer sensor-driven contact",
            file=str(path),
        )


def _template_base_name(node: ast.ClassDef) -> str | None:
    for base in node.bases:
        name = base.id if isinstance(base, ast.Name) else (base.attr if isinstance(base, ast.Attribute) else None)
        if name and name.endswith("Template"):
            return name
    return None


def _class_inherits(node: ast.ClassDef, base_name: str) -> bool:
    allowed = _TASK_BASE_NAMES if base_name == "TaskBase" else frozenset({base_name})
    for base in node.bases:
        if isinstance(base, ast.Name) and base.id in allowed:
            return True
        if isinstance(base, ast.Attribute) and base.attr in allowed:
            return True
    return False


def _class_inherits_from_task_template(node: ast.ClassDef) -> bool:
    for base in node.bases:
        if isinstance(base, ast.Name) and base.id in _TASK_TEMPLATE_BASES:
            return True
        if isinstance(base, ast.Attribute) and base.attr in _TASK_TEMPLATE_BASES:
            return True
    return False


def _decorator_is_register(dec: ast.expr, register_name: str) -> bool:
    if isinstance(dec, ast.Call):
        func = dec.func
        if isinstance(func, ast.Name) and func.id == register_name:
            return True
        if isinstance(func, ast.Attribute) and func.attr == register_name:
            return True
    if isinstance(dec, ast.Name) and dec.id == register_name:
        return True
    return False
