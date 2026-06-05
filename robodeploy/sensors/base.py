"""
SensorBase — shared scaffolding for all sensors.

Both sim sensors (MuJoCoCameraRenderer) and real sensors (RealSenseCamera)
inherit from SensorBase.

SensorBase provides:
  - State tracking (_initialized).
  - A guard in read() that raises if called before initialize().
  - Last-valid-data caching: if _read_impl() raises, the last successful
    reading is returned with a warning. This prevents a single dropped USB
    frame from crashing the control loop. On the very first read, with no
    cache, SensorTimeoutError is propagated to the caller.
  - warmup(): reads and discards N frames so cameras auto-expose and IMUs
    converge before episode data is collected.
  - Standard __repr__.

Pairing convention:
  Every sensor type has two implementations under the same name:
    sensors/camera/sim/mujoco_camera.py   → @register_sensor("wrist_camera_sim")
    sensors/camera/real/realsense.py      → @register_sensor("wrist_camera_real")

  RoboEnv selects the correct variant based on backend.is_real at construction.
  User code only specifies "wrist_camera"; the sim/real suffix is resolved
  automatically. This means user scripts never change between sim and real.

Implementation note for _read_impl():
  Real sensor drivers must apply their own driver-level timeouts (e.g.
  realsense2 pipeline.wait_for_frames(timeout_ms=100)). SensorBase will
  catch any exception and fall back to cached data, but a blocking driver
  call with no timeout will still hang the InferenceLoop. Always configure
  your driver with a timeout matching your control period.
"""

from __future__ import annotations

import warnings
from abc import abstractmethod
from typing import Optional, TYPE_CHECKING

from robodeploy.core.interfaces.sensor import ISensor
from robodeploy.core.types             import SensorData, SensorMount, SensorTimeoutError

if TYPE_CHECKING:
    from robodeploy.core.interfaces.backend import IBackend


class SensorBase(ISensor):
    """Shared scaffolding for all sensors. Subclass this, not ISensor directly."""

    def __init__(self, name: str, is_real: bool, config: dict | None = None, mount: SensorMount | None = None) -> None:
        """
        Args:
            name:    Unique sensor identifier (e.g. "wrist_camera", "ft_sensor").
            is_real: True if this is a real-hardware sensor implementation.
            config:  Sensor-specific configuration (resolution, frame rate, etc.).
            mount:   Optional link/world pose metadata for sim backends.
        """
        self._name:        str                   = name
        self._is_real:     bool                  = is_real
        self._initialized: bool                  = False
        self._last_data:   Optional[SensorData]  = None   # last-valid cache
        self.config:       dict                  = config or {}
        self.mount:        SensorMount           = mount or SensorMount()

    # ------------------------------------------------------------------
    # ISensor lifecycle
    # ------------------------------------------------------------------

    def initialize(self, backend: IBackend) -> None:
        """Call _init_impl() and mark as initialized."""
        self._init_impl(backend)
        self._initialized = True

    def mount_extrinsics(self) -> dict[str, object] | None:
        """Return mount pose metadata for camera-frame transforms."""
        mount = getattr(self, "mount", None)
        if mount is None or not mount.parent_link:
            return None
        return {
            "parent_link": str(mount.parent_link),
            "position": tuple(float(v) for v in mount.position),
            "orientation": tuple(float(v) for v in mount.orientation),
        }

    def read(self) -> SensorData:
        """Return the latest reading, falling back to last-valid data on failure.

        If _read_impl() raises any exception (dropped frame, driver timeout,
        USB reconnect) and a prior reading exists, the stale reading is
        returned with a warning. This keeps the InferenceLoop alive rather
        than crashing on a single bad frame.

        If no prior reading exists (first call ever), SensorTimeoutError is
        raised — the caller must handle sensor startup failure explicitly.
        """
        if not self._initialized:
            raise RuntimeError(
                f"Sensor '{self._name}' read() called before initialize(). "
                "Call env.reset() first."
            )
        try:
            data = self._read_impl()
            self._last_data = data
            return data
        except Exception as exc:
            if self._last_data is not None:
                warnings.warn(
                    f"[{self._name}] read() failed ({exc!r}); returning last valid data.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return self._last_data
            raise SensorTimeoutError(self._name, timeout_s=0.0) from exc

    def warmup(self, n_frames: int = 30) -> None:
        """Prime the last-valid cache and let the sensor reach steady state.

        Reads n_frames and discards them. On completion, _last_data is
        populated so the first real read() will always have a fallback.

        Args:
            n_frames: Frames to discard. 30 ≈ 1 second at 30 Hz.
        """
        for _ in range(n_frames):
            try:
                self._last_data = self._read_impl()
            except Exception:
                pass   # ignore failures during warmup

    def close(self) -> None:
        """Call _close_impl() if initialized."""
        if self._initialized:
            self._close_impl()
            self._initialized = False
            self._last_data   = None

    # ------------------------------------------------------------------
    # Abstract internal methods — subclasses implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def _init_impl(self, backend: IBackend) -> None:
        """Attach to backend or open hardware connection.

        For sim cameras: attach to renderer, set camera intrinsics and pose.
        For real cameras: open USB/GigE stream, set resolution and framerate.

        Args:
            backend: Active backend instance.
        """
        ...

    @abstractmethod
    def _read_impl(self) -> SensorData:
        """Return one sensor reading. Must be fast (called every control step)."""
        ...

    @abstractmethod
    def _close_impl(self) -> None:
        """Release hardware resources or detach from sim renderer."""
        ...

    # ------------------------------------------------------------------
    # ISensor properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_real(self) -> bool:
        return self._is_real

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        kind   = "real" if self._is_real else "sim"
        status = "initialized" if self._initialized else "not initialized"
        return f"{type(self).__name__}(name='{self._name}', {kind}, {status})"
