"""Keyboard teleop via pynput (preferred) or an injectable key-state fallback."""

from __future__ import annotations

import threading
from typing import Callable

import numpy as np

from robodeploy.teleop.base import ITeleopDevice, TeleopCommand

_DEFAULT_BINDINGS: dict[str, str] = {
    "w": "x+",
    "s": "x-",
    "a": "y-",
    "d": "y+",
    "q": "z-",
    "e": "z+",
    "i": "pitch+",
    "k": "pitch-",
    "j": "yaw-",
    "l": "yaw+",
    "u": "roll-",
    "o": "roll+",
    "space": "gripper_toggle",
    "r": "reset_episode",
    "tab": "record_toggle",
    "esc": "e_stop",
    "[": "step_down",
    "]": "step_up",
}

_AXIS_MAP = {
    "x+": ("delta_position", 0, 1.0),
    "x-": ("delta_position", 0, -1.0),
    "y+": ("delta_position", 1, 1.0),
    "y-": ("delta_position", 1, -1.0),
    "z+": ("delta_position", 2, 1.0),
    "z-": ("delta_position", 2, -1.0),
    "pitch+": ("delta_orientation_rpy", 1, 1.0),
    "pitch-": ("delta_orientation_rpy", 1, -1.0),
    "yaw+": ("delta_orientation_rpy", 2, 1.0),
    "yaw-": ("delta_orientation_rpy", 2, -1.0),
    "roll+": ("delta_orientation_rpy", 0, 1.0),
    "roll-": ("delta_orientation_rpy", 0, -1.0),
}


def _normalize_key(key) -> str | None:  # noqa: ANN001
    if key is None:
        return None
    if hasattr(key, "char") and key.char:
        return str(key.char).lower()
    if hasattr(key, "name") and key.name:
        return str(key.name).lower()
    return None


class KeyboardTeleop(ITeleopDevice):
    """WASD-style end-effector control with recording and episode hot-keys."""

    def __init__(
        self,
        *,
        step_position: float = 0.01,
        step_orientation: float = 0.05,
        bindings: dict[str, str] | None = None,
        use_listener: bool | None = None,
    ) -> None:
        self._step_position = float(step_position)
        self._step_orientation = float(step_orientation)
        self._bindings = dict(_DEFAULT_BINDINGS)
        if bindings:
            self._bindings.update(bindings)
        self._action_by_key = {k.lower(): v for k, v in self._bindings.items()}

        self._pressed: set[str] = set()
        self._edge: dict[str, bool] = {
            "record_toggle": False,
            "reset_episode": False,
            "e_stop": False,
            "gripper_toggle": False,
            "step_down": False,
            "step_up": False,
        }
        self._gripper = 0.0
        self._listener = None
        self._alive = False
        self._lock = threading.Lock()

        if use_listener is None:
            use_listener = _pynput_available()
        self._use_listener = bool(use_listener)

    def start(self) -> None:
        if self._alive:
            return
        self._alive = True
        if not self._use_listener:
            return
        from pynput import keyboard

        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.start()

    def stop(self) -> None:
        self._alive = False
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
        with self._lock:
            self._pressed.clear()

    @property
    def is_alive(self) -> bool:
        return self._alive

    def inject_key(self, key: str, *, pressed: bool) -> None:
        """Test helper: set key state without a live keyboard listener."""
        normalized = str(key).lower()
        with self._lock:
            if pressed:
                self._on_press_key(normalized)
                self._pressed.add(normalized)
            else:
                self._pressed.discard(normalized)

    def _on_press(self, key) -> None:  # noqa: ANN001
        normalized = _normalize_key(key)
        if normalized is None:
            return
        with self._lock:
            self._on_press_key(normalized)
            self._pressed.add(normalized)

    def _on_release(self, key) -> None:  # noqa: ANN001
        normalized = _normalize_key(key)
        if normalized is None:
            return
        with self._lock:
            self._pressed.discard(normalized)

    def _on_press_key(self, key: str) -> None:
        action = self._action_by_key.get(key)
        if action in ("record_toggle", "reset_episode", "e_stop", "gripper_toggle", "step_down", "step_up"):
            self._edge[action] = True

    def poll(self) -> TeleopCommand | None:
        with self._lock:
            pressed = set(self._pressed)
            edge = dict(self._edge)
            for flag in self._edge:
                self._edge[flag] = False

        if edge.get("step_down"):
            self._step_position = max(0.001, self._step_position * 0.8)
            self._step_orientation = max(0.005, self._step_orientation * 0.8)
        if edge.get("step_up"):
            self._step_position = min(0.2, self._step_position * 1.25)
            self._step_orientation = min(0.5, self._step_orientation * 1.25)

        delta_pos = np.zeros(3, dtype=np.float32)
        delta_rpy = np.zeros(3, dtype=np.float32)
        has_motion = False

        for key in pressed:
            action = self._action_by_key.get(key)
            if action is None or action in self._edge:
                continue
            spec = _AXIS_MAP.get(action)
            if spec is None:
                continue
            field_name, index, sign = spec
            scale = self._step_position if field_name == "delta_position" else self._step_orientation
            if field_name == "delta_position":
                delta_pos[index] += float(sign) * scale
                has_motion = True
            else:
                delta_rpy[index] += float(sign) * scale
                has_motion = True

        gripper_changed = bool(edge.get("gripper_toggle"))
        if gripper_changed:
            self._gripper = 0.0 if self._gripper > 0.5 else 1.0

        has_pos = bool(np.any(delta_pos))
        has_rpy = bool(np.any(delta_rpy))
        has_hotkey = any(
            (
                edge.get("record_toggle"),
                edge.get("reset_episode"),
                edge.get("e_stop"),
                gripper_changed,
            )
        )
        if not has_pos and not has_rpy and not has_hotkey:
            return None

        return TeleopCommand(
            delta_position=delta_pos.copy() if has_pos else None,
            delta_orientation_rpy=delta_rpy.copy() if has_rpy else None,
            gripper_command=self._gripper if gripper_changed else None,
            record_toggle=bool(edge.get("record_toggle")),
            reset_episode=bool(edge.get("reset_episode")),
            e_stop=bool(edge.get("e_stop")),
        )


def _pynput_available() -> bool:
    try:
        import pynput.keyboard  # noqa: F401

        return True
    except Exception:
        return False
