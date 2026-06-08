"""MuJoCo passive-viewer mouse drag teleop (EE target → delta command)."""

from __future__ import annotations

from typing import Any

import numpy as np

from robodeploy.teleop.base import ITeleopDevice, TeleopCommand


class MuJoCoMouseIKTeleop(ITeleopDevice):
    """Drag an EE marker in the MuJoCo viewer to teleoperate."""

    def __init__(
        self,
        *,
        backend: Any,
        ee_body_name: str = "wrist_link",
        viewer: Any | None = None,
    ) -> None:
        self._backend = backend
        self._ee_body_name = str(ee_body_name)
        self._viewer = viewer
        self._alive = False
        self._target_pos: np.ndarray | None = None
        self._target_quat: np.ndarray | None = None
        self._gripper = 0.0
        self._button_left = False

    def start(self) -> None:
        if self._alive:
            return
        self._alive = True
        if self._viewer is not None:
            return
        try:
            import mujoco
            import mujoco.viewer
        except ImportError as exc:
            raise ImportError("MuJoCoMouseIKTeleop requires mujoco. pip install robodeploy[sim]") from exc
        model = getattr(self._backend, "_model", None)
        data = getattr(self._backend, "_data", None)
        if model is None or data is None:
            raise RuntimeError("MuJoCoMouseIKTeleop requires a MuJoCoBackend with _model/_data")
        self._viewer = mujoco.viewer.launch_passive(model, data)
        self._install_mouse_callback()

    def stop(self) -> None:
        self._alive = False
        if self._viewer is not None and hasattr(self._viewer, "close"):
            try:
                self._viewer.close()
            except Exception:
                pass
        self._viewer = None

    @property
    def is_alive(self) -> bool:
        return self._alive

    def inject_target(
        self,
        *,
        position: np.ndarray | list[float],
        quaternion: np.ndarray | list[float] | None = None,
    ) -> None:
        """Test helper: set EE target without a live viewer."""
        self._target_pos = np.asarray(position, dtype=np.float32).reshape(3)
        if quaternion is not None:
            self._target_quat = np.asarray(quaternion, dtype=np.float32).reshape(4)

    def _install_mouse_callback(self) -> None:
        if self._viewer is None:
            return

        def _on_mouse_move(xpos: float, ypos: float) -> None:
            if not self._button_left:
                return
            self._target_pos = np.array([float(xpos) * 0.001, float(ypos) * 0.001, 0.4], dtype=np.float32)

        def _on_mouse_button(button: int, act: int) -> None:
            if int(button) == 0:
                self._button_left = int(act) == 1

        if hasattr(self._viewer, "set_mouse_callback"):
            self._viewer.set_mouse_callback(_on_mouse_move, _on_mouse_button)

    def _current_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        if hasattr(self._backend, "get_ee_pose"):
            pos, quat = self._backend.get_ee_pose(self._ee_body_name)
            return (
                np.asarray(pos, dtype=np.float32).reshape(3),
                np.asarray(quat, dtype=np.float32).reshape(4),
            )
        obs = self._backend.get_observation() if hasattr(self._backend, "get_observation") else None
        if obs is not None:
            return (
                np.asarray(obs.ee_position, dtype=np.float32).reshape(3),
                np.asarray(obs.ee_orientation, dtype=np.float32).reshape(4),
            )
        return np.zeros(3, dtype=np.float32), np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def poll(self) -> TeleopCommand | None:
        if self._target_pos is None:
            return None
        current_pos, _current_quat = self._current_ee_pose()
        delta = (self._target_pos - current_pos).astype(np.float32)
        self._target_pos = None
        if float(np.linalg.norm(delta)) < 1e-8:
            return None
        return TeleopCommand(
            delta_position=delta,
            gripper_command=self._gripper,
        )
