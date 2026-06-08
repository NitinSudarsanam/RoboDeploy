# Kuka FT + IMU multi-modal real-hardware demo

Demonstrates combined force-torque, IMU, and contact-threshold sensors on a live ROS2
robot (or ATI NetFT + Xsens serial when running without ROS2).

## Requirements

- Franka/Kuka arm with ROS2 controllers **or** standalone ATI NetFT + Xsens MTi
- `pip install -e ".[ros2]"` for ROS2 bridge
- Optional: `pip install pyserial` for native Xsens IMU

## ROS2 preset (recommended)

```bash
python examples/kuka_ft_imu_pick_real/run.py --preset kuka_ft_imu_multimodal_ros2
```

Bridges `/wrench`, `/imu`, and FT-threshold contact from the sensor rig.

## Native hardware (no ROS2)

Set environment variables and use the real preset:

```bash
export ATI_NETFT_HOST=192.168.1.100
export ROBODEPLOY_XSENS_PORT=/dev/ttyUSB0
python examples/kuka_ft_imu_pick_real/run.py --preset kuka_ft_imu_multimodal_real
```

**Hardware blocker**: native preset requires reachable ATI UDP and an open Xsens serial port.
Without hardware the env still constructs but FT/IMU reads return stale status.

## Policy behavior

- `grasp_detection: ft` — grasp confirmed from `obs.ft_force`
- `imu_omega_max` — phase settle waits for IMU stillness
- `halt_on_sensor_failure` — policy holds when `obs.sensor_status` reports critical sensor errors
