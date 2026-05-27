"""ATI NetFT force/torque sensor (real)."""

from __future__ import annotations

import socket
import struct
import time

import numpy as np

from robodeploy.core.registry import register_sensor, register_sensor_pair
from robodeploy.core.types import SensorData
from robodeploy.sensors.base import SensorBase


@register_sensor("ft_sensor_real")
class ATIFTSensor(SensorBase):
    def __init__(self, config: dict | None = None) -> None:
        super().__init__(name=str((config or {}).get("name", "ft_sensor")), is_real=True, config=config)

    def _init_impl(self, backend) -> None:
        del backend
        host = self.config.get("host")
        if not host:
            raise RuntimeError("ATIFTSensor requires config={'host': '<sensor-ip>'}.")
        self._addr = (str(host), int(self.config.get("port", 49152)))
        self._timeout_s = float(self.config.get("timeout_s", 0.05))
        self._scale = float(self.config.get("scale", 1.0 / 1_000_000.0))
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.settimeout(self._timeout_s)
        # ATI NetFT RDT start streaming command. If unsupported, first read will time out clearly.
        command = struct.pack("!HHI", 0x1234, 2, int(self.config.get("samples", 0)))
        self._sock.sendto(command, self._addr)

    def _read_impl(self) -> SensorData:
        packet, _ = self._sock.recvfrom(1024)
        if len(packet) < 36:
            raise RuntimeError(f"ATI NetFT packet too short: {len(packet)} bytes.")
        counts = np.asarray(struct.unpack("!6i", packet[-24:]), dtype=np.float32)
        wrench = counts * self._scale
        now = time.monotonic()
        return SensorData(
            ft_force=wrench[:3],
            ft_torque=wrench[3:],
            timestamp=now,
            timestamp_hw=now,
            timestamp_recv=now,
            timestamp_source="hardware",
        )

    def _close_impl(self) -> None:
        sock = getattr(self, "_sock", None)
        if sock is not None:
            try:
                stop = struct.pack("!HHI", 0x1234, 0, 0)
                sock.sendto(stop, self._addr)
            except Exception:
                pass
            sock.close()


@register_sensor_pair(
    "wrist_ft",
    real=ATIFTSensor,
    by_backend={"ros2": ATIFTSensor},
)
class AtiWristFTPair:
    pass

