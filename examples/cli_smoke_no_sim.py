"""Minimal CLI smoke example (no simulator required).

This is intentionally tiny: it just shows the CLI subcommands that work
everywhere by using the built-in dummy environment.
"""

from __future__ import annotations

import subprocess
import sys


def main() -> int:
    cmds = [
        [sys.executable, "-m", "robodeploy.cli", "list-presets"],
        [sys.executable, "-m", "robodeploy.cli", "list-registry", "--builtins"],
        [
            sys.executable,
            "-m",
            "robodeploy.cli",
            "run-episode",
            "--preset",
            "kuka_pick_mujoco",
            "--dummy",
            "--steps",
            "5",
            "--action",
            "sinusoid",
        ],
    ]
    for cmd in cmds:
        subprocess.check_call(cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

