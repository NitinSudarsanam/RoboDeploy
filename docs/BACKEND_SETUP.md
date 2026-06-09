# Backend setup: MuJoCo, ROS 2 + RViz, Isaac Sim

This guide matches the library layering in [ARCHITECTURE.md](../ARCHITECTURE.md): **user code talks to `RoboEnv` + backends**, not to simulator SDKs directly.

**Windows default**: MuJoCo + ROS 2 + RViz are the primary “golden paths”. Isaac Sim on Windows is documented as **best-effort** (often smooth on Linux).

### Python interpreters (conda / WSL)

RoboDeploy does not require a specific layout, but one working split is:

- **Conda env `ros2_env`**: use one environment for ROS 2 (Jazzy) *and* MuJoCo when you want both in the same place—install `mujoco` with `pip` into that env for `run_mujoco`, and run ROS2-backed examples from the same interpreter so `rclpy` matches your ROS install.
- **WSL + repo venv `.venv-wsl`**: under WSL, activate [`.venv-wsl`](../.venv-wsl) (or recreate a venv there) for ROS2+RViz workflows so Linux ROS packages and `rclpy` line up with your distro; run `python -m examples.user_kuka_sinusoid.run_ros2_rviz` from that shell.

Use whichever Python actually has `mujoco` for MuJoCo-only demos and whichever is bound to your ROS2 install for `run_ros2_rviz` / `run_gazebo`.

**SO-101 on real USB (Feetech)**: for `backend_for_simulator("real_world", robots=[so101_robot], ...)`, the library selects the `so101_feetech` controller (lerobot serial bus + ROS2 topic bridge). Install extras, udev, calibration CLI, and safety notes are in [SO101_REAL.md](SO101_REAL.md).

---

## Shared: how backend configuration works

Backends receive keyword arguments and normalize them into a single `config` mapping.
Prefer flat keyword args:

```python
backend_kwargs={"enable_viewer": True}
```

One-level nesting is still supported for compatibility:

```python
backend_kwargs={"config": {"enable_viewer": True}}
```

Avoid double-nested config. `BackendBase` still flattens it for older examples, but new code should not generate this shape:

```python
backend_kwargs={"config": {"config": {"enable_viewer": True}}}
```

---

## Kuka sinusoid: one robot-centric pattern (MuJoCo / ROS2+RViz / Gazebo)

Use [`backend_for_simulator`](../robodeploy/backends/simulator.py) so the library derives per-robot ROS topics, joint names for `JointState`, RViz defaults, and (for Gazebo) the `sim` launch fragment from each [`RobotDescription`](../robodeploy/description/base.py) (`ros_transport_joint_names`, optional `gazebo_sim_launch_config`, etc.):

- **`"mujoco"`** — `MuJoCoBackend` with viewer + actuator fallback defaults (single robot).
- **`"ros2_rviz"`** — `ROS2RvizBackend`, a simulated ROS2/RViz transport with joint-position drivers under `/<robot_id>/`.
- **`"gazebo"`** — `ROS2GazeboBackend`; requires `RobotDescription.gazebo_sim_launch_config()` and/or `config_overrides["sim"]`.

Optional **`local_ros_graph=True`** (ROS2+RViz only) starts the built-in `dev_fake_sim` joint-position devtool.

Override defaults with **`config_overrides`** (nested dicts merge recursively).

```python
from robodeploy.backends.simulator import backend_for_simulator
from robodeploy.env import RoboEnv

# robot = Robot(... same tasks/policies for every simulator ...)
env = RoboEnv(backend=backend_for_simulator("mujoco", robots=[robot]), robots=[robot])
# env = RoboEnv(backend=backend_for_simulator("ros2_rviz", robots=[robot], local_ros_graph=True), robots=[robot])
# env = RoboEnv(backend=backend_for_simulator("gazebo", robots=[robot]), robots=[robot])

# Optional: e.g. disable MuJoCo viewer
# env = RoboEnv(
#     backend=backend_for_simulator("mujoco", robots=[robot], config_overrides={"enable_viewer": False}),
#     robots=[robot],
# )
```

Thin entrypoints (unchanged module names):

- `python -m examples.user_kuka_sinusoid.run_mujoco`
- `python -m examples.user_kuka_sinusoid.run_ros2_rviz` (add `--fake-sim` for `local_ros_graph`)
- `python -m examples.user_kuka_sinusoid.run_gazebo`

**Smoke checks (from repo root):** MuJoCo needs `pip install mujoco`; without it, `run_mujoco` should still exit 0 after printing the install hint. ROS2 examples need a Jazzy-capable environment with `rclpy` available; use `run_ros2_rviz --fake-sim` when no external robot graph is running. Gazebo path requires `gz` and related packages as in the Gazebo section below.

---

## MuJoCo (primary on Windows)

### Prerequisites

- Python 3.x supported by your environment
- Install MuJoCo Python bindings:

```bash
pip install mujoco
```

### Canonical example

- [examples/user_kuka_sinusoid/run_mujoco.py](../examples/user_kuka_sinusoid/run_mujoco.py)

Run (from repo root):

```bash
python -m examples.user_kuka_sinusoid.run_mujoco
```

### Viewer

MuJoCo viewer is controlled by `enable_viewer` in backend config (see [robodeploy/backends/sim/mujoco/backend.py](../robodeploy/backends/sim/mujoco/backend.py)).

The Kuka sinusoid demo enables the viewer by default via `backend_for_simulator("mujoco", ...)`. To match the older explicit shape:

```python
backend_kwargs={"config": {"enable_viewer": True}}
```

### RViz from MuJoCo (optional)

MuJoCoBackend can publish the same RoboDeploy RViz topics when enabled:

```python
backend_kwargs={"config": {
  "rviz": {"enabled": True, "fixed_frame": "world", "publish_hz": 10.0},
}}
```

### Assets (MJCF or URDF)

MuJoCo first looks for an MJCF model path from the robot description or an override. If only a URDF is available, `MuJoCoBackend` can ask MuJoCo to compile it once, inject position actuators, and run the augmented model. This auto path is convenient for bring-up but less explicit than a hand-authored MJCF.

Useful config keys:

- `asset_overrides`: choose a specific MJCF or URDF per robot.
- `cache_compiled_mjcf`: cache the generated MJCF text for inspection when MuJoCo exposes it.
- `compiled_cache_dir`: directory for generated MJCF cache files.
- `urdf_position_kp`, `urdf_joint_damping`, `urdf_joint_armature`: tuning for the generated actuator and joint defaults.

### Multi-robot MuJoCo

Only use multi-robot `RoboEnv(robots=[...], tasks=[...])` if your chosen backend implements `initialize_multi()`.

If a backend does not implement it, `RoboEnv` will raise a clear `NotImplementedError` telling you to use single-agent mode or pick a different backend.

---

## ROS 2 Jazzy + RViz (primary on Windows)

### Prerequisites

- ROS 2 **Jazzy** Python environment (`rclpy`, `tf2_ros`, `sensor_msgs`, `std_msgs`, `visualization_msgs`, `geometry_msgs`)
- A running robot graph that matches your configured topics (commonly `ros2_control` + `joint_state_broadcaster`)

### Gazebo Harmonic (gz-sim) simulator path

RoboDeploy can **optionally** launch Gazebo Harmonic (`gz sim`) for you when using the `ros2` backend.

Prereqs (high level):

- `gz` available on PATH (Gazebo Harmonic)
- `ros_gz_bridge` installed (for `/clock`, images, IMU, wrench bridges)
- `ros_gz_sim` for URDF spawn (`ros2 run ros_gz_sim create`)
- `gz_ros2_control` + `ros2_control` controllers (`joint_state_broadcaster`, `joint_trajectory_controller`)
- Optional Pinocchio for Cartesian reach on URDF: `pip install -e ".[kinematics]"` (`pin` package on Linux)

Bundled Kuka URDF with `ros2_control` lives at
`robodeploy/description/kuka/assets/urdf/kuka.urdf` (used automatically by `KukaDescription`).

Canonical examples:

- [examples/user_kuka_sinusoid/run_gazebo.py](../examples/user_kuka_sinusoid/run_gazebo.py) — sinusoid joint motion
- [examples/kuka_ft_imu_pick_gazebo/run_gazebo.py](../examples/kuka_ft_imu_pick_gazebo/run_gazebo.py) — multimodal pick-place (RGB-D + FT + IMU + contact + prop_pose)

Live sensor CI gate (Linux):

```bash
ROBODEPLOY_LIVE_GAZEBO=1 pytest tests/test_live_gazebo_sensors.py -q
```

**Multimodal pick-place demo** (Linux + GUI or headless):

```bash
pip install -e ".[kinematics]"   # Pinocchio IK for reach on URDF
python -m examples.kuka_ft_imu_pick_gazebo.run_gazebo
```

Expected `obs` keys after reset: `images`, `ft_forces`, `imu_angular_velocity`, `contact_state`, `objects`.
Controllers from `gz_ros2_control` publish at the **root** namespace (`/joint_states`, `/joint_trajectory_controller/joint_trajectory`).
`KukaDescription.gazebo_ros2_extra_config()` sets absolute topics so drivers do not double-prefix `/robot0//joint_states`.

Verify on Linux:

```bash
ros2 topic list | grep joint
ros2 topic echo /joint_states --once
ros2 control list
```

Troubleshooting:

- **No joint states / arm frozen** — confirm `/joint_states` is publishing (not only `/robot0/joint_states`). Check `joint_state_broadcaster` is active.
- **JTC deaf** — echo `/joint_trajectory_controller/joint_trajectory` while stepping; commands should appear.
- **FT never triggers grasp** — tune `force_threshold` in `kuka_ft_imu_pick_gazebo` preset; arm links need URDF `<collision>` (bundled in `kuka.urdf`).
- **Carry invisible** — Gazebo `follow` mode uses kinematic bookkeeping + `set_pose`; weld grasp is not supported. Cube may clip through gripper in GUI.

**Limitations:** single-robot Gazebo (one URDF spawn); multi-robot Gazebo is not supported yet.

The example uses:

- `config.sim.kind = "gazebo"`
- `config.sim.world = <repo>/examples/user_kuka_sinusoid/assets/gazebo_world.sdf` (replace with your world)
- `config.sim.controllers_to_spawn = [...]` (best-effort via `ros2 control load_controller`)

### Canonical example

- [examples/ros2_rviz_minimal.py](../examples/ros2_rviz_minimal.py)
- Kuka sinusoid (ROS2+RViz): [examples/user_kuka_sinusoid/run_ros2_rviz.py](../examples/user_kuka_sinusoid/run_ros2_rviz.py)

### Simulator option (Windows-friendly)

If you just want a simulator ROS graph (no Gazebo/Isaac needed), the Kuka ROS2+RViz
example can start a tiny fake joint-position simulator that:

- publishes `/robot0/joint_states`
- accepts `/robot0/joint_position_commands` (or your configured `robot0.joint_cmd_topic`)
- broadcasts a minimal TF `base_link -> ee_link` (for the EE pose topic)

Run:

```bash
python -m examples.user_kuka_sinusoid.run_ros2_rviz --fake-sim
```

### Per-robot ROS namespace convention

For `robot_id="robot0"`, drivers typically expect:

- `/<robot0>/joint_states`
- `/<robot0>/<controller_command_topic>` (configurable)

Configure per-robot keys on the backend config, for example:

- `robot0.joint_states_topic`
- `robot0.joint_cmd_topic`
- `robot0.base_frame`, `robot0.ee_frame`

See [robodeploy/backends/real/ros2/backend.py](../robodeploy/backends/real/ros2/backend.py).

### Picking a driver for Kuka (or any non-Franka robot)

If you have a ROS2 bringup that exposes `/joint_states` and accepts a `Float64MultiArray`
of joint targets, use the generic driver:

```python
backend_kwargs={"config": {
  "controller_by_robot_id": {"robot0": "joint_position"},
  "robot0.joint_states_topic": "joint_states",
  "robot0.joint_cmd_topic": "joint_position_commands",
}}
```

If your joint names/order differ from your `RobotDescription`, override them:

```python
backend_kwargs={"config": {
  "robot0.joint_names": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"],
}}
```

### Presets (recommended)

To reduce per-robot boilerplate, you can pick a preset (data-only defaults) and override as needed:

```python
backend_kwargs={"config": {
  "robot0.preset": "kuka_jtc",
  # Overrides:
  # "robot0.joint_cmd_topic": "joint_trajectory_controller/joint_trajectory",
}}
```

### RViz topics published by RoboDeploy

Under `/robodeploy` (default):

- Scene markers: `/robodeploy/scene/markers`
- Task markers: `/robodeploy/tasks/markers`
- EE pose (per robot): `/robodeploy/<robot_id>/ee_pose`
- EE trace (per robot): `/robodeploy/<robot_id>/trace`

Implementation: [robodeploy/backends/real/ros2/rviz.py](../robodeploy/backends/real/ros2/rviz.py).

TF lookup failures are reported in controller diagnostics instead of silently substituting an identity pose. When TF is unavailable, controllers report `ee_pose_valid=False` and the pose fields contain `NaN` unless you explicitly opt into identity fallback for a local demo.

Scene marker scale follows the same convention as MuJoCo geometry: `GeomSpec.size` is treated as half-extents for boxes and radius/half-height style values for primitive geometry, so RViz marker dimensions are doubled where needed.

### RViz display recipe

1. Set **Fixed Frame** to your configured `rviz.fixed_frame` (default `world`).
2. Add **Pose** display → topic `/robodeploy/<robot_id>/ee_pose`.
3. Add **MarkerArray** → `/robodeploy/scene/markers`.
4. Add **MarkerArray** → `/robodeploy/tasks/markers`.
5. Add **Marker** → `/robodeploy/<robot_id>/trace`.
6. Add **RobotModel** → topic **Description** = `/robot_description` (started automatically when `rviz.enabled=true`, unless Gazebo already launched `robot_state_publisher` for `sim.robot_urdf`).

### RViz enablement + rates

Enable RViz publishing:

```python
backend_kwargs={
  "rviz": {
    "enabled": True,
    "fixed_frame": "world",
    "publish_hz": 10.0,
    # default True: runs `robot_state_publisher` from the robot URDF so RViz RobotModel works
    "launch_robot_state_publisher": True,
  },
}
```

Command publish pacing (optional):

- `command_hz` (global default)
- `command_hz_by_robot_id` (per robot)

See [robodeploy/backends/real/ros2/interfaces.py](../robodeploy/backends/real/ros2/interfaces.py).

### Task goal visualization (backend-agnostic)

Tasks expose goals via `ITask.viz_goals()`; `RoboEnv` forwards a JSON-friendly payload to backends that support `set_viz_payload()` (ROS2 backend does).

### Sensor streams (ROS2 backend)

You can attach optional sensor streams via per-robot config:

```python
backend_kwargs={"config": {
  "robot0.sensors": [
    {
      "type": "rgbd",
      "name": "front",
      "rgb": "camera/color/image_raw",
      "depth": "camera/depth/image_raw",
      "info": "camera/color/camera_info",
    }
  ],
}}
```

RGBD diagnostics include configured topics, hardware/wall timestamps, RGB-depth skew, and the last callback error if one occurred.

---

## Isaac Sim (secondary on Windows)

### Why Windows is “best-effort”

Isaac Sim depends on a large native extension graph. Typical Windows issues:

- Missing MSVC runtime causing DLL load failures
- `PYTHONPATH` / conda env contamination causing wrong NumPy / wrong Python ABI
- Kit/experience path resolution issues

### Canonical example

- [examples/user_kuka_sinusoid/run_isaacsim.py](../examples/user_kuka_sinusoid/run_isaacsim.py)

### Recommended run style

Run with Isaac Sim’s own Python launcher (`python.bat`) from a **clean** shell:

- deactivate conda envs that inject incompatible packages
- avoid setting `PYTHONPATH` to a different Python’s site-packages

### URDF requirements

URDFs used for articulation import should include reasonable inertial definitions (otherwise initialization can fail depending on importer settings).

Backend knobs live in [robodeploy/backends/sim/isaacsim/backend.py](../robodeploy/backends/sim/isaacsim/backend.py) (`experience`, `headless`, `renderer`, import tuning, etc.).

### RViz from Isaac Sim (optional)

IsaacSimBackend can publish the same RoboDeploy RViz topics when enabled:

```python
backend_kwargs={"config": {
  "rviz": {"enabled": True, "fixed_frame": "world", "publish_hz": 10.0},
}}
```

---

## Live sensor CI (GitHub Actions)

Two optional jobs validate bridged ROS2 sensor topics end-to-end:

| Job | Env flag | Test module |
|-----|----------|-------------|
| `sensor-live-ros2` | `ROBODEPLOY_LIVE_ROS2=1` | `tests/test_live_ros2_sensors.py` |
| `sensor-live-gazebo` | `ROBODEPLOY_LIVE_GAZEBO=1` | `tests/test_live_gazebo_sensors.py` |

**Local reproduction (Linux + ROS 2 Jazzy sourced):**

```bash
source /opt/ros/jazzy/setup.bash
pip install -e ".[dev,sim]"

# ROS2 RViz transport + test publishers (no external robot graph required for CI fixture)
ROBODEPLOY_LIVE_ROS2=1 python -m pytest tests/test_live_ros2_sensors.py -q

# Headless Gazebo empty world + bridged wrist camera/FT topics
ROBODEPLOY_LIVE_GAZEBO=1 python -m pytest tests/test_live_gazebo_sensors.py -q
```

Runnable demos:

- `python -m examples.kuka_sensor_ros2_rviz.run_ros2_rviz` (requires live `/robot0/joint_states` + sensor topics)
- `python -m examples.kuka_sensor_gazebo.run_gazebo` (requires `gz` on PATH)
- `python -m examples.sensor_diagnostics_demo.run` (no ROS; prints `sensor_status` on simulated fault)

**Isaac Sim** sensors remain mock-only in CI (no NVIDIA runtime on standard `ubuntu-latest` runners). Validate Isaac camera/FT manually on a GPU workstation.

### Multi-sensor showcase (MuJoCo, Linux)

Demonstrates wrist + overhead camera, FT, IMU, and prop-pose sensors in one run:

```bash
pip install -e ".[sim]"
python -m examples.sensor_showcase.run
```

Outputs `examples/sensor_showcase/sensor_showcase.json` and a montage image
(`.png` with Pillow, else `.ppm`). Skips on Windows (headless EGL).

### Gazebo sensor topic wiring

Injected URDF sensors publish on namespaced topics matching ROS2 defaults:
`/wrist_camera/image_raw`, `/wrist_ft/wrench`. `ros_gz_bridge` rules include
`CameraInfo` and `Imu` when sensor rigs declare `info` / IMU topics.

Live CI: `test_live_gazebo_sensors.py` includes a gz-rendered camera assertion
(no synthetic image publisher) using `tests/fixtures/gazebo_camera_ft.sdf`.

### Follow-up sensor modalities (planned)

- Real perception: vision-based `obs.objects` (color blob → `ISensor`, ArUco).
- Tactile, lidar, proximity: extend `Observation`/`SensorData` when hardware
  drivers are added (see `CONTRACTS.md`).

---

## Verification checklist

- **MuJoCo**: example runs N steps; viewer optional; MJCF resolves.
- **ROS2+RViz**: with your ROS graph running, example runs; RViz shows markers + per-robot EE + trace.
- **Isaac Sim**: example launches on Isaac python; articulation initializes; no numpy/ABI mismatch.
- **Sensors**: `kuka_pick_mujoco` uses `prop_pose` sensor → `obs.objects`; live ROS2/Gazebo jobs pass on `main`.
