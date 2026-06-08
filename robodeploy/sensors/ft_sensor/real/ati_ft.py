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
        header, msg_type, _sample_count, status = struct.unpack("!HHII", packet[:12])
        if header != 0x1234:
            raise RuntimeError(f"ATI NetFT invalid header: 0x{header:04x} (expected 0x1234).")
        if msg_type != 2:
            raise RuntimeError(f"ATI NetFT unexpected message type: {msg_type} (expected 2).")
        # Status bit 0x01 = force/torque overload per ATI NetFT RDT specification.
        if status & 0x01:
            raise RuntimeError("ATI NetFT force/torque overflow flag set.")
        counts = np.asarray(struct.unpack("!6i", packet[12:36]), dtype=np.float32)
        wrench = counts * self._scale
        now = time.monotonic()
        return SensorData(
            ft_force=wrench[:3],
            ft_torque=wrench[3:],
            timestamp=now,
            timestamp_hw=now,
            timestamp_recv=now,
            timestamp_source="hardware",
            status="ok",
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
)
class AtiWristFTPair:
    """ATI UDP hardware (``real`` default). ROS2/Gazebo topic FT uses ``Ros2WrenchISensor``."""
    pass

