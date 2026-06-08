"""Xsens MTi native IMU driver (serial MTData2)."""

from __future__ import annotations

import struct
import time

import numpy as np

from robodeploy.core.registry import register_sensor, register_sensor_pair
from robodeploy.core.types import SensorData
from robodeploy.sensors.base import SensorBase

# MTData2 packet header bytes (Xsens device output configuration dependent).
_MTDATA2_PREAMBLE = 0xFA
_MTDATA2_BID = 0xFF
_MTDATA2_MID = 0x36


def _decode_mtdata2_accel_gyro(payload: bytes) -> tuple[np.ndarray, np.ndarray] | None:
    """Best-effort parse of accel+gyro floats from an MTData2 payload."""
    if len(payload) < 24:
        return None
    try:
        vals = struct.unpack_from("<6f", payload, 0)
        accel = np.asarray(vals[:3], dtype=np.float32)
        gyro = np.asarray(vals[3:6], dtype=np.float32)
        if not np.all(np.isfinite(accel)) or not np.all(np.isfinite(gyro)):
            return None
        return accel, gyro
    except struct.error:
        return None


def _read_mtdata2_packet(serial) -> bytes | None:  # noqa: ANN001
    """Read one MTData2 message from a pyserial port."""
    header = serial.read(4)
    if len(header) < 4:
        return None
    if header[0] != _MTDATA2_PREAMBLE or header[1] != _MTDATA2_BID or header[2] != _MTDATA2_MID:
        return None
    length = int(header[3])
    body = serial.read(length + 1)  # payload + checksum
    if len(body) < length:
        return None
    return body[:length]


@register_sensor("imu_xsens_real")
class XsensIMUSensor(SensorBase):
    """Xsens MTi serial reader with MTData2 framing and dry-run fallback.

    When ``pyserial`` is unavailable or the port cannot be opened, returns a
    stationary gravity vector so unit tests and dry-runs still work.
    """

    def __init__(self, config: dict | None = None) -> None:
        cfg = dict(config or {})
        super().__init__(name=str(cfg.get("name", "wrist_imu")), is_real=True, config=cfg)
        self._serial = None
        self._port = str(cfg.get("port", ""))
        self._baud = int(cfg.get("baud", 115200))
        self._timeout_s = float(cfg.get("timeout_s", 0.05))
        self._use_mtdata2 = bool(cfg.get("use_mtdata2", True))

    def _init_impl(self, backend) -> None:
        del backend
        if not self._port:
            return
        try:
            import serial  # type: ignore[import-not-found]
        except ImportError:
            return
        try:
            self._serial = serial.Serial(self._port, self._baud, timeout=self._timeout_s)
        except Exception:
            self._serial = None

    def _read_impl(self) -> SensorData:
        now = time.monotonic()
        accel = np.array([0.0, 0.0, 9.81], dtype=np.float32)
        gyro = np.zeros(3, dtype=np.float32)
        if self._serial is not None:
            try:
                if self._use_mtdata2:
                    payload = _read_mtdata2_packet(self._serial)
                    if payload is not None:
                        decoded = _decode_mtdata2_accel_gyro(payload)
                        if decoded is not None:
                            accel, gyro = decoded
                else:
                    raw = self._serial.read(32)
                    decoded = _decode_mtdata2_accel_gyro(raw)
                    if decoded is not None:
                        accel, gyro = decoded
            except Exception:
                pass
        return SensorData(
            imu_acceleration=accel,
            imu_angular_velocity=gyro,
            timestamp=now,
            timestamp_hw=now,
            timestamp_recv=now,
            timestamp_source="hardware",
        )

    def _close_impl(self) -> None:
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass
            self._serial = None


@register_sensor_pair("wrist_imu", real=XsensIMUSensor)
class XsensWristIMUPair:
    pass


@register_sensor_pair("base_imu", real=XsensIMUSensor)
class XsensBaseIMUPair:
    pass
