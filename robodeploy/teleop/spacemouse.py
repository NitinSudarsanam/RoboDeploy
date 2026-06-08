"""SpaceMouse / SpaceNavigator 6-DOF teleop via pyspacemouse."""

from __future__ import annotations

import threading

import numpy as np

from robodeploy.teleop.base import ITeleopDevice, TeleopCommand

_DEFAULT_BUTTON_MAP: dict[int, str] = {
    0: "gripper_toggle",
    1: "record_toggle",
}


def _apply_deadzone(values: np.ndarray, deadzone: float) -> np.ndarray:
    out = np.asarray(values, dtype=np.float32).reshape(-1).copy()
    limit = float(deadzone)
    for index, value in enumerate(out):
        if abs(float(value)) < limit:
            out[index] = 0.0
        else:
            sign = 1.0 if value >= 0.0 else -1.0
            out[index] = sign * (abs(float(value)) - limit) / max(1e-6, 1.0 - limit)
    return out


class SpaceMouseTeleop(ITeleopDevice):
    """3Dconnexion SpaceMouse / SpaceNavigator (6-DOF analog input)."""

    def __init__(
        self,
        *,
        deadzone: float = 0.05,
        scale_position: float = 0.002,
        scale_orientation: float = 0.01,
        button_map: dict[int, str] | None = None,
        driver: object | None = None,
    ) -> None:
        self._deadzone = float(deadzone)
        self._scale_position = float(scale_position)
        self._scale_orientation = float(scale_orientation)
        self._button_map = dict(_DEFAULT_BUTTON_MAP)
        if button_map:
            self._button_map.update(button_map)

        self._driver = driver
        self._alive = False
        self._lock = threading.Lock()
        self._translation = np.zeros(3, dtype=np.float32)
        self._rotation = np.zeros(3, dtype=np.float32)
        self._edge: dict[str, bool] = {
            "record_toggle": False,
            "reset_episode": False,
            "e_stop": False,
            "gripper_toggle": False,
        }
        self._gripper = 0.0

    def start(self) -> None:
        if self._alive:
            return
        self._alive = True
        if self._driver is not None:
            return
        try:
            import pyspacemouse
        except ImportError as exc:
            raise ImportError(
                "SpaceMouseTeleop requires pyspacemouse. "
                "Install with: pip install 'robodeploy[teleop]'"
            ) from exc
        if not pyspacemouse.open(
            callback=self._on_state,
            button_callback=self._on_button,
        ):
            raise RuntimeError("SpaceMouse not detected")

    def stop(self) -> None:
        self._alive = False
        if self._driver is None:
            try:
                import pyspacemouse

                pyspacemouse.close()
            except Exception:
                pass
        with self._lock:
            self._translation.fill(0.0)
            self._rotation.fill(0.0)

    @property
    def is_alive(self) -> bool:
        return self._alive

    def inject_state(
        self,
        *,
        translation: np.ndarray | list[float] | None = None,
        rotation: np.ndarray | list[float] | None = None,
        button: int | None = None,
    ) -> None:
        """Test helper: push a synthetic HID state without hardware."""
        if translation is not None:
            self._on_state(type("State", (), {"x": translation[0], "y": translation[1], "z": translation[2]})())
        if rotation is not None:
            self._on_state(
                type(
                    "State",
                    (),
                    {"roll": rotation[0], "pitch": rotation[1], "yaw": rotation[2]},
                )()
            )
        if button is not None:
            self._on_button(int(button), 1)

    def _on_state(self, state) -> None:  # noqa: ANN001
        with self._lock:
            if hasattr(state, "x"):
                self._translation = np.array(
                    [getattr(state, "x", 0.0), getattr(state, "y", 0.0), getattr(state, "z", 0.0)],
                    dtype=np.float32,
                )
            if hasattr(state, "roll"):
                self._rotation = np.array(
                    [
                        getattr(state, "roll", 0.0),
                        getattr(state, "pitch", 0.0),
                        getattr(state, "yaw", 0.0),
                    ],
                    dtype=np.float32,
                )

    def _on_button(self, button: int, state: int) -> None:
        if int(state) != 1:
            return
        action = self._button_map.get(int(button))
        if action in self._edge:
            with self._lock:
                self._edge[action] = True

    def poll(self) -> TeleopCommand | None:
        with self._lock:
            translation = self._translation.copy()
            rotation = self._rotation.copy()
            edge = dict(self._edge)
            for flag in self._edge:
                self._edge[flag] = False
            self._translation.fill(0.0)
            self._rotation.fill(0.0)

        translation = _apply_deadzone(translation, self._deadzone)
        rotation = _apply_deadzone(rotation, self._deadzone)
        delta_pos = translation * self._scale_position
        delta_rpy = rotation * self._scale_orientation

        has_pos = bool(np.any(np.abs(delta_pos) > 1e-8))
        has_rpy = bool(np.any(np.abs(delta_rpy) > 1e-8))
        gripper_changed = bool(edge.get("gripper_toggle"))
        if gripper_changed:
            self._gripper = 0.0 if self._gripper > 0.5 else 1.0

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
            delta_position=delta_pos.astype(np.float32) if has_pos else None,
            delta_orientation_rpy=delta_rpy.astype(np.float32) if has_rpy else None,
            gripper_command=self._gripper if gripper_changed else None,
            record_toggle=bool(edge.get("record_toggle")),
            reset_episode=bool(edge.get("reset_episode")),
            e_stop=bool(edge.get("e_stop")),
        )
