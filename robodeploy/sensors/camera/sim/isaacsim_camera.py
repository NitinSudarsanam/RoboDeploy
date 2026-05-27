"""Isaac Sim camera sensor (sim)."""

from __future__ import annotations

import time
from pathlib import PurePosixPath

import numpy as np

from robodeploy.core.registry import register_sensor
from robodeploy.core.types import SensorData, SensorMount
from robodeploy.sensors.base import SensorBase


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _quat_to_xyz_degrees(quat: tuple[float, float, float, float]) -> tuple[float, float, float]:
    import math

    w, x, y, z = (float(v) for v in quat)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = math.asin(_clamp(sinp, -1.0, 1.0))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return tuple(math.degrees(v) for v in (roll, pitch, yaw))


@register_sensor("wrist_camera_isaacsim")
class IsaacSimCameraRenderer(SensorBase):
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
        if not hasattr(backend, "_world") or not hasattr(backend, "_simulation_app"):
            raise RuntimeError("IsaacSimCameraRenderer requires an initialized IsaacSimBackend.")
        self._backend = backend
        self._width = int(self.config.get("width", self.config.get("image_width", 640)))
        self._height = int(self.config.get("height", self.config.get("image_height", 480)))
        self._render_depth = bool(self.config.get("depth", False))
        self._camera_name = str(self.config.get("camera_name", self.config.get("name", self.name)))
        self._prim_path = str(self.config.get("prim_path") or self._default_prim_path(backend))
        self._camera = self._create_camera_handle()

    def _read_impl(self) -> SensorData:
        rgb = None
        depth = None
        if hasattr(self._camera, "get_current_frame"):
            frame = self._camera.get_current_frame()
            if isinstance(frame, dict):
                rgb = frame.get("rgba") or frame.get("rgb")
                depth = frame.get("depth") or frame.get("distance_to_image_plane")
        if rgb is None:
            for method_name in ("get_rgba", "get_rgb"):
                method = getattr(self._camera, method_name, None)
                if callable(method):
                    rgb = method()
                    break
        if depth is None and self._render_depth:
            for method_name in ("get_depth", "get_depth_map", "get_distance_to_image_plane"):
                method = getattr(self._camera, method_name, None)
                if callable(method):
                    depth = method()
                    break
        sim_time = float(getattr(self._backend, "_sim_time", 0.0))
        return SensorData(
            rgb=self._coerce_rgb(rgb),
            depth=self._coerce_depth(depth),
            timestamp=sim_time,
            timestamp_hw=sim_time,
            timestamp_recv=time.monotonic(),
            timestamp_source="sim",
        )

    def _close_impl(self) -> None:
        camera = getattr(self, "_camera", None)
        if camera is None:
            return
        for method_name in ("destroy", "close", "stop"):
            method = getattr(camera, method_name, None)
            if callable(method):
                try:
                    method()
                except Exception:
                    pass
                break

    def _create_camera_handle(self):
        camera_cls = self._resolve_camera_class()
        self._ensure_camera_prim()
        ctor_attempts = (
            lambda: camera_cls(
                prim_path=self._prim_path,
                name=self._camera_name,
                resolution=(self._width, self._height),
                frequency=max(1, int(getattr(self._backend, "control_hz", 30.0))),
            ),
            lambda: camera_cls(prim_path=self._prim_path, resolution=(self._width, self._height)),
            lambda: camera_cls(prim_path=self._prim_path),
            lambda: camera_cls(self._prim_path, resolution=(self._width, self._height)),
            lambda: camera_cls(self._prim_path),
        )
        last_error = None
        camera = None
        for attempt in ctor_attempts:
            try:
                camera = attempt()
                break
            except TypeError as exc:
                last_error = exc
                continue
        if camera is None:
            raise TypeError(f"Could not construct Isaac camera for '{self._prim_path}': {last_error}")

        for method_name, value in (
            ("set_resolution", (self._width, self._height)),
            ("set_frequency", max(1, int(getattr(self._backend, "control_hz", 30.0)))),
        ):
            method = getattr(camera, method_name, None)
            if callable(method):
                try:
                    method(value)
                except Exception:
                    pass
        init = getattr(camera, "initialize", None)
        if callable(init):
            init()
        return camera

    @staticmethod
    def _resolve_camera_class():
        try:
            from isaacsim.sensors.camera import Camera  # type: ignore[import-not-found]

            return Camera
        except Exception:
            try:
                from omni.isaac.sensor import Camera  # type: ignore[import-not-found]

                return Camera
            except Exception as exc:
                raise ImportError(
                    "IsaacSimCameraRenderer requires Isaac Sim camera sensor support. "
                    "Install/run inside Isaac Sim."
                ) from exc

    def _ensure_camera_prim(self) -> None:
        try:
            import omni.usd  # type: ignore[import-not-found]
            from pxr import Gf, UsdGeom  # type: ignore[import-not-found]
        except Exception:
            return

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return
        cam = UsdGeom.Camera.Define(stage, self._prim_path)
        xform = UsdGeom.XformCommonAPI(cam)
        xform.SetTranslate(Gf.Vec3d(*[float(v) for v in self.mount.position]))
        if hasattr(xform, "SetRotate"):
            xform.SetRotate(_quat_to_xyz_degrees(self.mount.orientation))

    def _default_prim_path(self, backend) -> str:  # noqa: ANN001
        override = str(self.config.get("parent_prim_path", "") or "").strip()
        if override:
            return str(PurePosixPath(override) / self._camera_name)
        if self.mount.parent_link and hasattr(backend, "_robot_prim_path"):
            base = PurePosixPath(str(getattr(backend, "_robot_prim_path", "/World/robot0")))
            return str(base / str(self.mount.parent_link) / self._camera_name)
        return str(PurePosixPath("/World") / self._camera_name)

    @staticmethod
    def _coerce_rgb(value):  # noqa: ANN001
        if value is None:
            return None
        arr = np.asarray(value)
        if arr.ndim == 3 and arr.shape[-1] == 4:
            arr = arr[..., :3]
        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating) and float(arr.max(initial=0.0)) <= 1.0:
                arr = np.clip(arr * 255.0, 0.0, 255.0)
            arr = arr.astype(np.uint8)
        return arr

    @staticmethod
    def _coerce_depth(value):  # noqa: ANN001
        if value is None:
            return None
        arr = np.asarray(value, dtype=np.float32)
        return arr


@register_sensor("overhead_camera_isaacsim")
class IsaacSimOverheadCameraRenderer(IsaacSimCameraRenderer):
    def __init__(self, config: dict | None = None, *, mount: SensorMount | None = None) -> None:
        cfg = {"name": "overhead_camera", "camera_name": "overhead_camera", **dict(config or {})}
        super().__init__(config=cfg, mount=mount)
