"""Xbox / PlayStation gamepad teleop via pygame.joystick."""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from robodeploy.teleop.base import ITeleopDevice, TeleopCommand


def _apply_deadzone(value: float, deadzone: float) -> float:
    v = float(value)
    limit = float(deadzone)
    if abs(v) < limit:
        return 0.0
    sign = 1.0 if v >= 0.0 else -1.0
    return sign * (abs(v) - limit) / max(1e-6, 1.0 - limit)


class GamepadTeleop(ITeleopDevice):
    """Two-stick + triggers + buttons gamepad teleop."""

    def __init__(
        self,
        *,
        joystick_index: int = 0,
        deadzone: float = 0.1,
        scale_position: float = 0.005,
        scale_orientation: float = 0.05,
        backend: Any | None = None,
    ) -> None:
        self._joystick_index = int(joystick_index)
        self._deadzone = float(deadzone)
        self._scale_position = float(scale_position)
        self._scale_orientation = float(scale_orientation)
        self._backend = backend
        self._joystick = None
        self._alive = False
        self._lock = threading.Lock()
        self._injected_axes: dict[int, float] = {}
        self._injected_buttons: dict[int, bool] = {}
        self._edge: dict[str, bool] = {
            "record_toggle": False,
            "reset_episode": False,
            "e_stop": False,
        }
        self._prev_buttons: dict[int, bool] = {}

    def start(self) -> None:
        if self._alive:
            return
        self._alive = True
        if self._backend is not None:
            return
        try:
            import pygame
        except ImportError as exc:
            raise ImportError(
                "GamepadTeleop requires pygame. Install with: pip install 'robodeploy[teleop]'"
            ) from exc
        if not pygame.get_init():
            pygame.init()
        if not pygame.joystick.get_count():
            raise RuntimeError("No gamepad detected")
        self._joystick = pygame.joystick.Joystick(self._joystick_index)
        self._joystick.init()

    def stop(self) -> None:
        self._alive = False
        if self._joystick is not None:
            self._joystick.quit()
            self._joystick = None
        with self._lock:
            self._injected_axes.clear()
            self._injected_buttons.clear()

    @property
    def is_alive(self) -> bool:
        return self._alive

    def inject_axis(self, index: int, value: float) -> None:
        """Test helper: set analog axis value [-1, 1]."""
        with self._lock:
            self._injected_axes[int(index)] = float(value)

    def inject_button(self, index: int, *, pressed: bool) -> None:
        """Test helper: set digital button state."""
        with self._lock:
            self._injected_buttons[int(index)] = bool(pressed)

    def _read_axis(self, index: int) -> float:
        with self._lock:
            if index in self._injected_axes:
                return float(self._injected_axes[index])
        if self._joystick is None:
            return 0.0
        try:
            return float(self._joystick.get_axis(index))
        except Exception:
            return 0.0

    def _read_button(self, index: int) -> bool:
        with self._lock:
            if index in self._injected_buttons:
                return bool(self._injected_buttons[index])
        if self._joystick is None:
            return False
        try:
            return bool(self._joystick.get_button(index))
        except Exception:
            return False

    def _edge_button(self, index: int, action: str) -> None:
        pressed = self._read_button(index)
        prev = self._prev_buttons.get(index, False)
        if pressed and not prev:
            self._edge[action] = True
        self._prev_buttons[index] = pressed

    def poll(self) -> TeleopCommand | None:
        if self._backend is None and self._joystick is not None:
            try:
                import pygame

                pygame.event.pump()
            except Exception:
                pass

        left_x = _apply_deadzone(self._read_axis(0), self._deadzone)
        left_y = _apply_deadzone(self._read_axis(1), self._deadzone)
        right_x = _apply_deadzone(self._read_axis(2), self._deadzone)
        right_y = _apply_deadzone(self._read_axis(3), self._deadzone)
        lb = self._read_button(4)
        rb = self._read_button(5)
        lt = max(0.0, _apply_deadzone(self._read_axis(4) if self._joystick else 0.0, 0.05))
        rt = max(0.0, _apply_deadzone(self._read_axis(5) if self._joystick else 0.0, 0.05))

        delta_pos = np.array(
            [
                left_y * self._scale_position,
                left_x * self._scale_position,
                (float(rb) - float(lb)) * self._scale_position,
            ],
            dtype=np.float32,
        )
        delta_rpy = np.array(
            [
                0.0,
                -right_y * self._scale_orientation,
                right_x * self._scale_orientation,
            ],
            dtype=np.float32,
        )

        self._edge_button(0, "reset_episode")
        self._edge_button(1, "e_stop")
        self._edge_button(2, "record_toggle")
        edge = dict(self._edge)
        for flag in self._edge:
            self._edge[flag] = False

        has_pos = bool(np.any(np.abs(delta_pos) > 1e-8))
        has_rpy = bool(np.any(np.abs(delta_rpy) > 1e-8))
        gripper_command = None
        if lt > 0.05 or rt > 0.05:
            gripper_command = float(rt - lt)

        has_hotkey = any(edge.values()) or gripper_command is not None
        if not has_pos and not has_rpy and not has_hotkey:
            return None

        return TeleopCommand(
            delta_position=delta_pos if has_pos else None,
            delta_orientation_rpy=delta_rpy if has_rpy else None,
            gripper_command=gripper_command,
            record_toggle=bool(edge.get("record_toggle")),
            reset_episode=bool(edge.get("reset_episode")),
            e_stop=bool(edge.get("e_stop")),
        )
