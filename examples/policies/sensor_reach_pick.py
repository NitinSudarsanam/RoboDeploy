"""Sensor-driven reach policy — thin re-export of packaged demo policy."""

from robodeploy.demos.policies.sensor_reach_pick import SensorReachPickPlacePolicy  # noqa: F401

__all__ = ["SensorReachPickPlacePolicy"]
