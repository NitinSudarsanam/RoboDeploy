"""ContactSensorBase — shared scaffolding for binary contact sensors."""

from __future__ import annotations

from robodeploy.sensors.base import SensorBase


class ContactSensorBase(SensorBase):
    """Binary contact / proximity sensor base class."""
