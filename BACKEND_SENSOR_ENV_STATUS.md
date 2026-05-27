# Backend Sensor Environment Status

This is the implementation status for `BACKEND_SENSOR_ENV_EVALUATION.md` without changing the evaluation source.

## Current Capability

- `MuJoCoBackend`: primary supported path for loaded `WorldSpec` props, generated cameras from `SensorMount`, mounted force/torque sites, named image/depth observations, prop pose edits, and domain randomization hooks.
- `ROS2RealBackend`: supports ROS2 RGBD streams through the compatibility config path and the `ISensor` adapter path. Sensor diagnostics now include topic, timestamp, and skew details.
- `ROS2GazeboBackend`: remains ROS2 transport over Gazebo, with stricter process/topic readiness and generated image bridge rules for configured RGBD topics.
- `IsaacSimBackend`: loads basic `WorldSpec` primitives, lights, and cameras into the USD stage when Isaac USD APIs are available. Prop pose reads/writes use USD prims when possible and otherwise report metadata-only fallback in diagnostics.

## Known Gaps

- MuJoCo IMU/touch sensors are not implemented yet.
- Gazebo does not spawn `WorldSpec` props as Gazebo models yet.
- Isaac physics fidelity for generated props is still basic and should be verified inside Isaac before treating it as parity with MuJoCo.
- Real-world prop pose support still needs a perception source such as AprilTag, ChArUco, or OptiTrack.

## Verification

- Unit coverage lives in `tests/test_sensor_env_capabilities.py` and `tests/test_registry_honesty.py`.
- MuJoCo acceptance coverage lives in `tests/test_sensors_and_scenes_e2e.py` and is skipped when the optional `mujoco` package is unavailable.
- Recommended smoke command:

```shell
python -m compileall robodeploy tests examples
python -m unittest discover -s tests
```
