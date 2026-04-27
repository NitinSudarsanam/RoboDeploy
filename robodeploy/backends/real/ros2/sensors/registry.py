"""Registry for ROS2 sensors (Open/Closed)."""

from __future__ import annotations

from typing import Callable

from .interfaces import IRos2Sensor, Ros2SensorConfig


_SENSORS: dict[str, Callable[[Ros2SensorConfig, dict], IRos2Sensor]] = {}


def register_ros2_sensor(sensor_type: str):
    def decorator(factory: Callable[[Ros2SensorConfig, dict], IRos2Sensor]):
        if sensor_type in _SENSORS:
            raise KeyError(f"ROS2 sensor '{sensor_type}' already registered.")
        _SENSORS[sensor_type] = factory
        return factory

    return decorator


def make_ros2_sensor(sensor_type: str, cfg: Ros2SensorConfig, backend_config_dict: dict) -> IRos2Sensor:
    if sensor_type not in _SENSORS:
        raise KeyError(f"ROS2 sensor '{sensor_type}' not found. Registered: {list(_SENSORS)}")
    return _SENSORS[sensor_type](cfg, backend_config_dict)

