"""MuJoCo camera sensor (sim)."""

from __future__ import annotations

import time

from robodeploy.core.registry import register_sensor, register_sensor_pair
from robodeploy.core.types import SensorData, SensorMount
from robodeploy.sensors.base import SensorBase
from robodeploy.sensors.camera.sim.isaacsim_camera import (
    IsaacSimCameraRenderer,
    IsaacSimOverheadCameraRenderer,
)


@register_sensor("wrist_camera_sim")
class MuJoCoCameraRenderer(SensorBase):
    def __init__(
        self,
        name: str | dict | None = None,
        *,
        config: dict | None = None,
        mount: SensorMount | None = None,
    ) -> None:
        if isinstance(name, dict) and config is None:
            cfg = dict(name)
            sensor_name = str(cfg.get("name", "wrist_camera"))
        else:
            cfg = dict(config or {})
            sensor_name = str(name or cfg.get("name", "wrist_camera"))
        if mount is None and isinstance(cfg.get("mount"), dict):
            mount = SensorMount(**cfg["mount"])
        super().__init__(name=sensor_name, is_real=False, config=cfg, mount=mount)

    def _init_impl(self, backend) -> None:
        if not hasattr(backend, "_mujoco") or not hasattr(backend, "_model") or not hasattr(backend, "_data"):
            raise RuntimeError("MuJoCoCameraRenderer requires an initialized MuJoCoBackend.")
        self._mujoco = backend._mujoco
        self._model = backend._model
        self._data = backend._data
        self._width = int(self.config.get("width", self.config.get("image_width", 640)))
        self._height = int(self.config.get("height", self.config.get("image_height", 480)))
        self._render_depth = bool(self.config.get("depth", False))
        requested = str(self.config.get("camera_name", self.config.get("name", self.name)))
        camera_id = self._mujoco.mj_name2id(self._model, self._mujoco.mjtObj.mjOBJ_CAMERA, requested)
        if camera_id < 0 and bool(self.config.get("allow_camera_fallback", False)) and requested != "main":
            requested = "main"
            camera_id = self._mujoco.mj_name2id(self._model, self._mujoco.mjtObj.mjOBJ_CAMERA, requested)
        if camera_id < 0:
            raise KeyError(f"MuJoCo camera '{requested}' not found.")
        self._camera_name = requested
        self._renderer = self._mujoco.Renderer(self._model, height=self._height, width=self._width)

    def _read_impl(self) -> SensorData:
        self._renderer.update_scene(self._data, camera=self._camera_name)
        rgb = self._renderer.render()
        depth = None
        if self._render_depth:
            self._renderer.enable_depth_rendering()
            self._renderer.update_scene(self._data, camera=self._camera_name)
            depth = self._renderer.render()
            self._renderer.disable_depth_rendering()
        sim_time = float(self._data.time)
        return SensorData(
            rgb=rgb,
            depth=depth,
            timestamp=sim_time,
            timestamp_hw=sim_time,
            timestamp_recv=time.monotonic(),
            timestamp_source="sim",
        )

    def _close_impl(self) -> None:
        renderer = getattr(self, "_renderer", None)
        if renderer is not None:
            try:
                renderer.close()
            except Exception:
                pass


@register_sensor("overhead_camera_sim")
class MuJoCoOverheadCameraRenderer(MuJoCoCameraRenderer):
    def __init__(self, config: dict | None = None, *, mount: SensorMount | None = None) -> None:
        cfg = {"name": "overhead_camera", "camera_name": "overhead_camera", **dict(config or {})}
        super().__init__(config=cfg, mount=mount)


@register_sensor_pair(
    "wrist_camera",
    sim=MuJoCoCameraRenderer,
    by_backend={
        "mujoco": MuJoCoCameraRenderer,
        "isaacsim": IsaacSimCameraRenderer,
        "ros2": None,
        "ros2_rviz": None,
        "gazebo": None,
    },
)
class WristCameraPair:
    pass


@register_sensor_pair(
    "overhead_camera",
    sim=MuJoCoOverheadCameraRenderer,
    by_backend={
        "mujoco": MuJoCoOverheadCameraRenderer,
        "isaacsim": IsaacSimOverheadCameraRenderer,
        "ros2": None,
        "ros2_rviz": None,
        "gazebo": None,
    },
)
class OverheadCameraPair:
    pass

