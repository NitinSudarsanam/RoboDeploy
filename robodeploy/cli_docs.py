"""Generate CLI reference markdown from argparse introspection."""

from __future__ import annotations

import argparse
from pathlib import Path


def _format_parser(parser: argparse.ArgumentParser, *, heading_level: int = 2) -> list[str]:
    lines: list[str] = []
    prefix = "#" * heading_level
    lines.append(f"{prefix} `{parser.prog}`")
    if parser.description:
        lines.append("")
        lines.append(parser.description.strip())
    lines.append("")

    for action in parser._actions:
        if action.dest in {"help", argparse.SUPPRESS}:
            continue
        if isinstance(action, argparse._SubParsersAction):
            for name, sub in sorted(action.choices.items()):
                lines.append(f"{prefix}# `{parser.prog} {name}`")
                if sub.description:
                    lines.append("")
                    lines.append(sub.description.strip())
                lines.append("")
                for sub_action in sub._actions:
                    if sub_action.dest in {"help", argparse.SUPPRESS}:
                        continue
                    if isinstance(sub_action, argparse._SubParsersAction):
                        for sub_name, nested in sorted(sub_action.choices.items()):
                            lines.extend(
                                _format_subcommand(parser.prog, name, sub_name, nested)
                            )
                    else:
                        lines.extend(_format_action(sub_action))
                lines.append("")
            continue
        lines.extend(_format_action(action))

    return lines


def _format_subcommand(prog: str, parent: str, name: str, parser: argparse.ArgumentParser) -> list[str]:
    lines = [f"### `{prog} {parent} {name}`"]
    if parser.description:
        lines.append("")
        lines.append(parser.description.strip())
    lines.append("")
    for action in parser._actions:
        if action.dest in {"help", argparse.SUPPRESS}:
            continue
        if not isinstance(action, argparse._SubParsersAction):
            lines.extend(_format_action(action))
    lines.append("")
    return lines


def _format_action(action: argparse.Action) -> list[str]:
    if isinstance(action, argparse._SubParsersAction):
        return []
    opts = ", ".join(action.option_strings) if action.option_strings else action.dest
    req = "required" if action.required else "optional"
    default = ""
    if action.default not in (None, argparse.SUPPRESS) and action.default != "":
        default = f" (default: `{action.default}`)"
    help_text = (action.help or "").strip()
    return [f"- **{opts}** — {req}{default}. {help_text}"]


def generate_cli_reference() -> str:
    from robodeploy.cli import _build_parser

    parser = _build_parser()
    lines = [
        "# RoboDeploy CLI Reference",
        "",
        "Auto-generated from `robodeploy.cli._build_parser()`. Regenerate with:",
        "",
        "```bash",
        "python -c \"from robodeploy.cli_docs import write_cli_reference; write_cli_reference()\"",
        "```",
        "",
        "Install the package first: `pip install -e .`",
        "",
    ]
    lines.extend(_format_parser(parser, heading_level=2))
    return "\n".join(lines).rstrip() + "\n"


def write_cli_reference(path: Path | str | None = None) -> Path:
    out = Path(path) if path else Path(__file__).resolve().parents[1] / "docs" / "CLI_REFERENCE.md"
    out.write_text(generate_cli_reference(), encoding="utf-8")
    return out
