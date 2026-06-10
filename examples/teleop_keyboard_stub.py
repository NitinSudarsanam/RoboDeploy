"""Read-only ITeleopDevice contract demo (no MuJoCo, no pynput listener).

Shows how any teleop backend implements ``start`` / ``poll`` / ``stop`` and returns
``TeleopCommand`` snapshots. Use this to validate device wiring before attaching
a real env::

    python -m examples.teleop_keyboard_stub
    python -m examples.teleop_keyboard_stub --keys w,d,space,r
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robodeploy.teleop.base import ITeleopDevice, TeleopCommand
from robodeploy.teleop.keyboard import KeyboardTeleop


class KeyboardStubTeleop(ITeleopDevice):
    """Keyboard teleop with injectable keys only (contract reference implementation)."""

    def __init__(self) -> None:
        self._keyboard = KeyboardTeleop(use_listener=False)

    def start(self) -> None:
        self._keyboard.start()

    def poll(self) -> TeleopCommand | None:
        return self._keyboard.poll()

    def stop(self) -> None:
        self._keyboard.stop()

    def inject_key(self, key: str, *, pressed: bool) -> None:
        self._keyboard.inject_key(key, pressed=pressed)


def _demo_sequence() -> list[tuple[str, bool]]:
    return [
        ("w", True),
        ("w", False),
        ("d", True),
        ("d", False),
        ("space", True),
        ("space", False),
        ("r", True),
        ("r", False),
    ]


def run_stub_demo(keys: list[str] | None = None) -> list[TeleopCommand]:
    device = KeyboardStubTeleop()
    device.start()
    commands: list[TeleopCommand] = []
    try:
        sequence = _demo_sequence()
        if keys:
            sequence = [(k, True) for k in keys] + [(k, False) for k in keys]
        for key, pressed in sequence:
            device.inject_key(key, pressed=pressed)
            cmd = device.poll()
            if cmd is not None:
                commands.append(cmd)
    finally:
        device.stop()
    return commands


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ITeleopDevice keyboard stub (read-only demo).")
    parser.add_argument(
        "--keys",
        default="",
        help="Comma-separated keys to inject (default: built-in WASD/space/R sequence).",
    )
    args = parser.parse_args(argv)
    keys = [k.strip() for k in str(args.keys).split(",") if k.strip()] or None
    commands = run_stub_demo(keys)
    for i, cmd in enumerate(commands):
        parts: list[str] = []
        if cmd.delta_position is not None:
            parts.append(f"delta_pos={np.round(cmd.delta_position, 4).tolist()}")
        if cmd.gripper_command is not None:
            parts.append(f"gripper={cmd.gripper_command}")
        if cmd.reset_episode:
            parts.append("reset_episode")
        if cmd.record_toggle:
            parts.append("record_toggle")
        if cmd.e_stop:
            parts.append("e_stop")
        label = ", ".join(parts) if parts else "idle"
        print(f"[{i}] {label}")
    print(f"polled {len(commands)} command(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
