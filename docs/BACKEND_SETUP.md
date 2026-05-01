# Backend setup: MuJoCo, ROS 2 + RViz, Isaac Sim

This guide matches the library layering in [ARCHITECTURE.md](../ARCHITECTURE.md): **user code talks to `RoboEnv` + backends**, not to simulator SDKs directly.

**Windows default**: MuJoCo + ROS 2 + RViz are the primary “golden paths”. Isaac Sim on Windows is documented as **best-effort** (often smooth on Linux).

### Python interpreters (conda / WSL)

RoboDeploy does not require a specific layout, but one working split is:

- **Conda env `ros2_env`**: use one environment for ROS 2 (Jazzy) *and* MuJoCo when you want both in the same place—install `mujoco` with `pip` into that env for `run_mujoco`, and run ROS2-backed examples from the same interpreter so `rclpy` matches your ROS install.
- **WSL + repo venv `.venv-wsl`**: under WSL, activate [`.venv-wsl`](../.venv-wsl) (or recreate a venv there) for ROS2+RViz workflows so Linux ROS packages and `rclpy` line up with your distro; run `python -m examples.user_kuka_sinusoid.run_ros2_rviz` from that shell.

Use whichever Python actually has `mujoco` for MuJoCo-only demos and whichever is bound to your ROS2 install for `run_ros2_rviz` / `run_gazebo`.

---

## Shared: how backend configuration works

Backends receive a single `config: dict` on construction (`BackendClass(**backend_kwargs)`).

For convenience, you may pass nested settings:

```python
backend_kwargs={"config": {"enable_viewer": True}}
```

`BackendBase` merges an *additional* inner `"config"` mapping into the top-level backend config (nested keys win on collisions). This supports both common shapes:

```python
backend_kwargs={"config": {"enable_viewer": True}}
```

and (less common, but seen in some wrappers):

```python
backend_kwargs={"config": {"config": {"enable_viewer": True}}}
```

---

## Kuka sinusoid: one robot-centric pattern (MuJoCo / ROS2+RViz / Gazebo)

Use [`backend_for_simulator`](../robodeploy/backends/simulator.py) so the library derives per-robot ROS topics, joint names for `JointState`, RViz defaults, and (for Gazebo) the `sim` launch fragment from each [`RobotDescription`](../robodeploy/description/base.py) (`ros_transport_joint_names`, optional `gazebo_sim_launch_config`, etc.):

- **`"mujoco"`** — `MuJoCoBackend` with viewer + actuator fallback defaults (single robot).
- **`"ros2_rviz"`** — `ROS2RealBackend` with RViz + joint-position drivers under `/<robot_id>/`.
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

### Assets (MJCF)

MuJoCo requires an MJCF model path from the robot description (or an override). If you only have URDF, you must supply MJCF (or a conversion pipeline). See error text in `MuJoCoBackend` for override format:

- `asset_overrides` in backend config (documented in backend error messages)

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
- `ros_gz_bridge` installed (for `/clock` and additional bridges)
- a `ros2_control` setup in your world that exposes `/<robot_id>/joint_states` and a controller command topic

Canonical example:

- [examples/user_kuka_sinusoid/run_gazebo.py](../examples/user_kuka_sinusoid/run_gazebo.py)

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

## Verification checklist

- **MuJoCo**: example runs N steps; viewer optional; MJCF resolves.
- **ROS2+RViz**: with your ROS graph running, example runs; RViz shows markers + per-robot EE + trace.
- **Isaac Sim**: example launches on Isaac python; articulation initializes; no numpy/ABI mismatch.
