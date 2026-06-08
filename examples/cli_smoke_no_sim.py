"""Minimal CLI smoke example (no simulator required).

Uses example presets under ``examples/config/presets.yaml`` (not the robodeploy package).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    presets_file = Path(__file__).resolve().parent / "config" / "presets.yaml"
    cmds = [
        [sys.executable, "-m", "examples.cli", "list-presets", "--presets-file", str(presets_file)],
        [sys.executable, "-m", "robodeploy.cli", "list-registry", "--builtins"],
        [
            sys.executable,
            "-m",
            "examples.cli",
            "run-episode",
            "--preset",
            "kuka_pick_mujoco",
            "--presets-file",
            str(presets_file),
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
