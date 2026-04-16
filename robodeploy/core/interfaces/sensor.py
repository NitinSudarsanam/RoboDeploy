"""
ISensor — the modular perception interface.

Each sensor is an independent unit of perception. Sensors come in paired
sim and real implementations that share the same interface:

  sensors/camera/sim/mujoco_camera.py   ←── ISensor
  sensors/camera/real/realsense.py      ←── ISensor  (same interface)

This pairing is what makes sim-to-real work for perception: the backend
wires up sim sensors in simulation and real sensors on hardware, but the
policy and ObsPipeline see the same ISensor.read() call either way.

A sensor is initialised with a reference to the active backend so that
sim sensors can attach to the physics renderer and real sensors can open
hardware connections.

SensorData from ISensor.read() is merged into the Observation by the
backend after each step. The ObsPipeline then normalises the merged result.

Adding a new sensor:
  1. Create sensors/<type>/sim/<name>.py and sensors/<type>/real/<name>.py.
  2. Both subclass SensorBase (which subclasses ISensor).
  3. Implement initialize(), read(), and close().
  4. Register with @register_sensor("<name>").
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from robodeploy.core.types import SensorData

if TYPE_CHECKING:
    from robodeploy.core.interfaces.backend import IBackend


class ISensor(ABC):
    """Abstract sensor: produces SensorData on demand."""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def initialize(self, backend: IBackend) -> None:
        """Attach to the backend and open hardware or sim connections.

        For sim sensors: attach to the renderer, set camera pose/intrinsics.
        For real sensors: open USB/ethernet connection, start streaming.

        Args:
            backend: The active backend. Sim sensors use it to access the
                     renderer. Real sensors may use it to sync timestamps.

        Raises:
            RuntimeError: If the hardware connection cannot be established.
        """
        ...

    @abstractmethod
    def read(self) -> SensorData:
        """Return the latest sensor reading.

        Must complete within the backend's control period.
        For cameras at 30 Hz this may return the same frame multiple times
        within a 100 Hz control loop — that is acceptable.

        Returns:
            SensorData with only the fields this sensor provides populated.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Release hardware resources and stop any background threads."""
        ...

    # ------------------------------------------------------------------
    # Properties (declared by subclasses)
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique sensor identifier within an environment.

        Used by the backend to merge multiple SensorData into one Observation.
        Example: "wrist_camera", "ft_sensor", "base_imu"
        """
        ...

    @property
    @abstractmethod
    def is_real(self) -> bool:
        """True if this sensor reads from physical hardware."""
        ...

    # ------------------------------------------------------------------
    # Optional override
    # ------------------------------------------------------------------

    def warmup(self, n_frames: int = 30) -> None:
        """Discard initial unstable frames and prime the last-valid cache.

        Default is a no-op. Override for sensors that need settling time:
          - Cameras: auto-exposure and auto-white-balance stabilisation.
          - IMUs: bias estimate convergence.
          - FT sensors: tare/zero-offset settling.

        Args:
            n_frames: Number of frames to read and discard. The default (30)
                      covers ~1 second at 30 Hz camera rates.
        """
        pass
