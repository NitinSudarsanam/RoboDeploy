# SO-101 follower on real hardware (Feetech / USB)

RoboDeploy runs the same task/policy code as in simulation by using `backend_for_simulator("real_world", ...)`, which selects the **`so101_feetech`** ROS2 controller adapter for the bundled [`SO101Description`](../robodeploy/description/so101/description.py): it talks to the arm over USB via Hugging Face **lerobot**’s `FeetechMotorsBus`, publishes `sensor_msgs/JointState` on `/<robot_id>/joint_states`, and echoes commands on `/<robot_id>/joint_position_commands` for tooling parity with the generic `joint_position` path.

## Prerequisites

- Linux is the tested path (udev, `/dev/ttyACM*`). Windows serial may work but is best-effort.
- ROS 2 Jazzy Python environment with `rclpy`, `sensor_msgs`, `std_msgs`, `tf2_ros` (same as other ROS2 backends).
- Feetech Python SDK (pulled in by lerobot extras):

```bash
pip install "lerobot[feetech]"
```

Tested against lerobot layouts that expose `lerobot.motors.feetech.FeetechMotorsBus` and `lerobot.motors.Motor` / `MotorNormMode` (see source if your install differs).

## udev (recommended)

Allow non-root access to the USB serial device, then replug the arm:

```bash
# Example: ATTRS{idVendor}=="1a86" — adjust to match `udevadm info -a -n /dev/ttyACM0`
SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", MODE="0666", GROUP="dialout"
```

## Calibration (required)

The bundled template [`robodeploy/description/so101/calibration/example.json`](../robodeploy/description/so101/calibration/example.json) is **refused at runtime** unless you pass `robot0.allow_uncalibrated=true` (dry runs only).

1. Move the arm with torque off and record **two** distinct poses; the CLI fits per joint:

   `tick ≈ zero_ticks + q_rad * ticks_per_rad`

2. Write a user JSON (default `~/.robodeploy/so101_calibration.json`):

```bash
python -m examples.so101.calibrate_so101 --port /dev/ttyACM0 --out ~/.robodeploy/so101_calibration.json
```

3. Point RoboDeploy at it:

- Environment: `export ROBODEPLOY_SO101_CALIBRATION=$HOME/.robodeploy/so101_calibration.json`
- Or config: `config_overrides={"robot0.calibration_path": ".../so101_calibration.json"}`

## Run the sinusoid demo on hardware

```bash
export ROBODEPLOY_SO101_PORT=/dev/ttyACM0
export ROBODEPLOY_SO101_CALIBRATION=$HOME/.robodeploy/so101_calibration.json
export ROBODEPLOY_BACKEND=real_world
python -m examples.so101.run_switch_simulator
```

Or:

```bash
python -m examples.so101.run_switch_simulator --backend real_world --port /dev/ttyACM0 --calibration ~/.robodeploy/so101_calibration.json
```

Optional: `--allow-uncalibrated` allows loading the bundled template (unsafe on a real arm).

## Behavior profiles

Use `BehaviorProfile(preset="demo")` or `--profile demo` for a first bring-up: lower `velocity_scale` and softer tracking translate to smaller `max_joint_velocity` and command pacing via existing `backend_for_simulator` wiring.

## Safety features (v1)

| Mechanism | Role |
|-----------|------|
| Torque off on fault | `_hard_stop` disables torque on watchdog, temperature, or limit violations. |
| Watchdog | Arms after the first commanded motion; must receive commands / `get_obs` feeds before timeout. |
| E-stop | SIGINT and console `q` (when TTY) trip a stop flag checked on every command and state read. |
| Joint limits | Soft URDF-derived limits in calibration JSON; position + finite-difference velocity checks. |
| Temperature | Background poll of `Present_Temperature`; exceeds `temperature_max_c` (default 70) → torque off. |

Recover: fix the fault condition, power-cycle if needed, re-run calibration if offsets drifted, then restart the script.

## Troubleshooting

- **`ImportError: ... lerobot`**: install `lerobot[feetech]` in the **same** Python as ROS2.
- **`MissingCalibrationError`**: run `calibrate_so101` or set `ROBODEPLOY_SO101_CALIBRATION`.
- **`Incorrect status packet` / comm errors**: wrong baud, loose cable, duplicate IDs on the bus, or insufficient supply voltage.
- **`real_world` + `ros2_rviz`**: the auto-config uses `joint_position` + `dev_fake_sim` for local RViz demos; `so101_feetech` is selected only for `real_world` unless you override `robot0.controller` explicitly.

## pytest hardware smoke (optional)

With a real port and deps installed:

```bash
export ROBODEPLOY_SO101_PORT=/dev/ttyACM0
pytest tests/test_so101_real.py -k hardware --tb=short
```

Without the env var that test is skipped.
