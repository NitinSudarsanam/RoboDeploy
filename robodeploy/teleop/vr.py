"""OpenXR VR controller teleop (stub — requires pyopenxr + headset hardware)."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from robodeploy.teleop.base import ITeleopDevice, TeleopCommand


class VRTeleop(ITeleopDevice):
    """OpenXR-based VR controller teleop. Requires Quest/Index + pyopenxr.

    This is a deferred MVP stub: ``start()`` raises ``ImportError`` when
    ``pyopenxr`` is not installed. Pass ``session`` for unit tests.
    """

    def __init__(
        self,
        *,
        controller: Literal["left", "right"] = "right",
        scale_position: float = 1.0,
        scale_orientation: float = 1.0,
        session: Any | None = None,
    ) -> None:
        self._controller = str(controller)
        self._scale_position = float(scale_position)
        self._scale_orientation = float(scale_orientation)
        self._session = session
        self._alive = False
        self._last_pose: np.ndarray | None = None
        self._injected: TeleopCommand | None = None

    def start(self) -> None:
        if self._alive:
            return
        if self._session is not None:
            self._alive = True
            return
        try:
            import pyopenxr  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "VRTeleop requires pyopenxr and OpenXR runtime hardware. "
                "VR teleop is deferred; use keyboard, gamepad, or SpaceMouse instead."
            ) from exc
        raise NotImplementedError(
            "VRTeleop OpenXR session wiring is not implemented yet. "
            "See plans/GOAL_04_TELEOP_DATA_COLLECTION.md (phase 4.5+)."
        )

    def stop(self) -> None:
        self._alive = False
        self._last_pose = None
        self._injected = None

    @property
    def is_alive(self) -> bool:
        return self._alive

    def inject_command(self, cmd: TeleopCommand | None) -> None:
        """Test hook: supply a command without OpenXR hardware."""
        self._injected = cmd

    def poll(self) -> TeleopCommand | None:
        if not self._alive:
            return None
        if self._injected is not None:
            cmd = self._injected
            self._injected = None
            return cmd
        return None
