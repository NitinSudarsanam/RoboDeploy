# Pick-and-Place Demo Runbook

Operational guide for the V1 Track A pick-place demos across MuJoCo, ROS 2 + RViz, and Gazebo. Use this when rehearsing a live demo, recording insurance video, or validating a fresh install.

**What these demos are:** scripted reach-trajectory policies (`example_reach_pick` + YAML phases) with FT-gated grasp engage and kinematic carry assist. They are **not** learned grasping and **not** force-closure grasping. Success is placement tolerance at the goal (plus sensor health gates on multimodal presets).

**Recommended headline preset (MuJoCo):** `kuka_ft_imu_pick_mujoco` — FT + IMU + contact + prop_pose, tuned for reliable success on Windows and Linux CI.

### Windows workflow (RViz / Gazebo)

| Path | When to use |
|------|-------------|
| **MuJoCo native** | Default live demo on Windows (`--viewer` or headless `--json`). |
| **Docker `demo-gazebo-pick`** | ROS 2 + Gazebo on Windows when WSL is Ubuntu **22.04** (no Jazzy apt) or you want CI-parity Linux without a 24.04 distro. Requires **Docker Desktop running** — probe with `powershell -File scripts/check_docker_engine.ps1`. |
| **WSL Ubuntu 24.04 + Jazzy** | Native RViz/Gazebo rehearsal; add distro with `wsl --install -d Ubuntu-24.04` (interactive, may need reboot). Not present on this machine as of 2026-06-11. |

`wsl --install -d Ubuntu-24.04` is **not** reliably non-interactive on Windows (first-time install prompts, optional reboot). Prefer Docker for automated smoke on 22.04-only hosts; migrate to 24.04 when you need native RViz windows.

---

## Prerequisites (all backends)

From the repo root:

```bash
python -m pip install -e ".[sim,dev]"
robodeploy doctor
```

| Check | MuJoCo | ROS 2 + RViz | Gazebo |
|-------|--------|--------------|--------|
| OS | Windows, Linux, macOS | **Linux or WSL2** (`rclpy` unavailable on native Windows) | **Linux or WSL2** |
| Python extras | `[sim]` (`mujoco`) | `[sim]` + ROS 2 Jazzy system packages | `[sim,kinematics]` recommended (Pinocchio IK) |
| Optional | `pip install -e ".[kinematics]"` for Pin IK | RViz2 display | `gz`, `ros_gz_bridge`, `gz_ros2_control`, `ros2-controllers` |

List available presets:

```bash
python -m examples.cli list-presets
```

Resolve a preset to inspect full config:

```bash
robodeploy config resolve --preset kuka_ft_imu_pick_mujoco --json
```

---

## MuJoCo — `kuka_ft_imu_pick_mujoco`

**Platform:** runs natively on Windows (headless or viewer).

### Install

```bash
pip install -e ".[sim,dev]"
```

`robodeploy doctor` should report MuJoCo as available.

### Headless smoke (CI parity)

```bash
python -m examples.cli run-episode --preset kuka_ft_imu_pick_mujoco --seed 0 --steps 1500 --json
```

Pass `--seed 0` for a reproducible demo run. Without `--seed`, episode randomization is nondeterministic and pick success may vary.

### Visual demo

```bash
python -m examples.cli run-episode --preset kuka_ft_imu_pick_mujoco --seed 0 --viewer --steps 1500 --json
```

If GLFW is unavailable, the MuJoCo backend warns and continues headless instead of crashing.

### Expected success signals

- JSON summary contains `"success": true` (typically step **~400–650**, seed 0).
- `failure` is false or absent; episode terminates before `max_episode_steps` (1500).
- No `FORCE_LIMIT` safety e-stop (preset sets `safety.max_force_N: 150` for sim carry artifacts).
- Grasp engage is FT-gated (`force_threshold: 1.5` N averaged over 3 steps); carry uses kinematic follow mode with `grasp_force_loss_threshold: 0.0` (no FT while prop is held clear of gripper).

### Windows / PowerShell notes

- Redirect JSON to a file instead of piping between Python processes (encoding issues on PS 5.1):

```powershell
python -m examples.cli run-episode --preset kuka_ft_imu_pick_mujoco --steps 1500 --json > out.json
Get-Content out.json
```

- Python 3.14 works for this demo if `mujoco` installs; CI targets 3.10–3.12. A 3.12 venv is optional insurance.

### Multi-seed check

```bash
python -c "from examples.env_from_preset import env_from_preset
for seed in range(10):
    env = env_from_preset('kuka_ft_imu_pick_mujoco', max_episode_steps=1500)
    obs, info = env.reset(seed=seed)
    done = False
    steps = 0
    while not done and steps < 1500:
        obs, r, done, info = env.step()
        steps += 1
    print(seed, steps, getattr(info, 'success', False))
    env.close()"
```

Target: **≥8/10** successes locally (CI asserts pick success on Linux).

---

## ROS 2 + RViz — `kuka_pick_ros2_rviz`

**Platform:** Linux or **WSL2** only. Native Windows Python cannot load `rclpy`.

Preset wires embedded `dev_fake_sim` (no external robot graph), RViz markers, wrist FT/IMU/contact, and prop_pose — same reach policy family as MuJoCo.

### WSL2 setup (one-time)

**Requires Ubuntu 24.04 (noble)** for ROS 2 Jazzy apt packages. Ubuntu 22.04 (jammy) only ships Humble — `ros-jazzy-*` is unavailable; upgrade or add a 24.04 distro (`wsl --install -d Ubuntu-24.04`).

**Scripted bootstrap** (preferred; run inside **Ubuntu 24.04** WSL from repo root):

```bash
bash scripts/wsl24-bootstrap.sh
```

One-liner alternative (same steps as the script):

```bash
sudo apt update && sudo apt install -y curl gnupg lsb-release && \
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null && \
sudo apt update && sudo apt install -y ros-jazzy-desktop ros-jazzy-tf2-ros python3-venv && \
cd /mnt/c/Users/<you>/projects/BrownIVL/RoboDeploy && python3 -m venv .venv-wsl && source .venv-wsl/bin/activate && \
pip install -e ".[ros2,kinematics,sim,dev]" && source /opt/ros/jazzy/setup.bash && robodeploy doctor
```

**Ubuntu 22.04 only?** Use Docker for headless pick demos (no WSLg/X11 required):

```powershell
docker compose -f docker/docker-compose.yml --profile ros2 run --rm demo-rviz-pick
docker compose -f docker/docker-compose.yml --profile ros2 run --rm demo-gazebo-pick
```

`demo-rviz-pick` runs `kuka_pick_ros2_rviz` with embedded `dev_fake_sim` (marker publishing only; no RViz GUI). For interactive RViz, use **Ubuntu 24.04** WSL or native Linux with Jazzy + WSLg/X11.

### Run

```bash
source /opt/ros/jazzy/setup.bash
python -m examples.cli run-episode --preset kuka_pick_ros2_rviz --steps 1500 --json
```

RViz should open (preset `rviz.enabled: true`). Fixed frame: `world`. Robot model from bundled Kuka URDF.

### Expected success signals

- RViz shows arm motion through reach → grasp → carry → place phases.
- JSON: `"success": true` when placement tolerance met.
- `obs` includes `ft_forces`, `imu_angular_velocity`, `contact_state`, `objects` after reset.
- Sensor health in `info.extra.sensor_health.overall` should be healthy.

### Fallback (transport smoke only)

If pick IK or FT sim is flaky, verify ROS graph with the sinusoid example:

```bash
python -m examples.user_kuka_sinusoid.run_ros2_rviz --fake-sim
```

Narrative: "RViz transport works; pick preset is `kuka_pick_ros2_rviz`."

### Caveats

- `kuka_sensor_ros2_rviz` is **sensor smoke only** (`example_joint_track`), not pick-place.
- Pinocchio (`[kinematics]`) improves Cartesian reach on URDF; without it, joint-space fallback may reduce success rate.

---

## Gazebo — `kuka_ft_imu_pick_gazebo`

**Platform:** Linux or **WSL2** with Gazebo Harmonic (`gz`).

### WSL2 / Linux packages

```bash
sudo apt install \
  ros-jazzy-ros-gz \
  ros-jazzy-gz-ros2-control \
  ros-jazzy-ros2-controllers \
  ros-jazzy-ros-gz-sim

source /opt/ros/jazzy/setup.bash
pip install -e ".[sim,kinematics,dev]"
```

World file is packaged at `robodeploy/ros2_assets/worlds/pick_minimal.sdf`.

### Run (native script)

```bash
source /opt/ros/jazzy/setup.bash
python -m examples.kuka_ft_imu_pick_gazebo.run_gazebo
```

Headless override for servers:

```bash
ROBODEPLOY_GAZEBO_HEADLESS=1 python -m examples.kuka_ft_imu_pick_gazebo.run_gazebo
```

### Docker (ROS2 image)

Start **Docker Desktop** first (Windows: `docker info` must succeed; engine pipe `dockerDesktopLinuxEngine` required).

```powershell
powershell -File scripts/check_docker_engine.ps1
docker compose -f docker/docker-compose.yml --profile ros2 run --rm demo-gazebo-pick
```

The `demo-gazebo-pick` service runs `docker/demo_gazebo_pick.sh`, which sets `ROBODEPLOY_GAZEBO_HEADLESS=1` and lets `RoboEnv` start the full stack (`gz sim`, `ros_gz_bridge`, URDF spawn via `ros_gz_sim`, `controller_manager`, then the pick episode). Image packages: `gz-harmonic`, `ros-jazzy-ros-gz-sim`, `gz-ros2-control`, `ros2-controllers` (`docker/Dockerfile.ros2`). First run builds `robodeploy/robodeploy:ros2` (~several minutes).

Smoke with a 10-minute cap (PowerShell):

```powershell
$job = Start-Job { docker compose -f docker/docker-compose.yml --profile ros2 run --rm demo-gazebo-pick 2>&1 }
Wait-Job $job -Timeout 600; Receive-Job $job; if ((Get-Job $job).State -eq 'Running') { Stop-Job $job; 'TIMEOUT' }
```

### Expected success signals

- Console: `done at step <N> success= True` and `source_to_goal_distance=<small>`.
- Progress logs every 100 steps show `ft_norm`, `contact=True` during grasp, `sensor_health` overall OK.
- Exit code `0`.

### Measured success rates

| Environment | Gate | Notes |
|-------------|------|-------|
| Linux CI (`sensor-live-gazebo`) | **≥50%** over 10 seeds | `tests/test_live_gazebo_pick_e2e.py` |
| WAVE2 target | **≥70%** over 10 seeds | Pending JTC/IK tuning |
| MuJoCo parity reference | **≥80%** | `tests/test_sensor_mujoco_integration.py` |

Multi-seed rehearsal in WSL2:

```bash
python -c "from examples.kuka_ft_imu_pick_gazebo.pick_episode import run_pick_episodes
r = run_pick_episodes(range(10))
print(r)"
```

### Gazebo-specific caveats

- **Place snap (default on):** `ROBODEPLOY_GAZEBO_PLACE_SNAP=1` (default) snaps the carried prop to the goal when the place phase ends (kinematic carry oracle). Set `ROBODEPLOY_GAZEBO_PLACE_SNAP=0` to measure honest JTC-only placement distance (best observed ~0.085 m in headless Docker; &lt;0.04 m not yet reliable — snap default documents the demo path).
- Carry mode is kinematic bookkeeping (`set_pose`), not a physics weld — cube may clip in GUI.
- Gazebo Kuka URDF gripper geometry differs from MuJoCo MJCF; policy offsets tuned per backend.
- Live run **not possible** on native Windows (no `rclpy` / `gz` in standard Windows Python).

---

## Fallback order (live demo day)

1. **MuJoCo live** — `kuka_ft_imu_pick_mujoco --viewer` (works on Windows).
2. **Recorded video** — capture successful MuJoCo / WSL2 Gazebo / RViz runs as insurance.
3. **Dummy backend** — proves CLI and env wiring only:

```bash
robodeploy run-episode --dummy --steps 10 --json
```

---

## Presets not recommended for live demo

Documented honest failures (do not substitute for headline demo):

| Preset | Issue |
|--------|-------|
| `kuka_pick_mujoco` | Legacy reach policy; grasp often fails — use `kuka_ft_imu_pick_mujoco` instead |
| `kuka_vision_pick_mujoco` | Color-blob source localization; frequently times out |
| `franka_pick_mujoco` | Panda gripper geometry needs separate carry/grasp offsets |
| `kuka_sensor_ros2_rviz` | Joint-track smoke only, not manipulation |
| `kuka_sensor_gazebo` | Sensor smoke only |

---

## Benchmark eval (pip install)

Benchmark presets ship inside the `benchmarks` package (wheel/sdist). After `pip install robodeploy` (no repo checkout required):

```bash
pip install robodeploy pyyaml
robodeploy list-benchmarks --json
robodeploy eval --benchmark manipulation_v1/reach_target --policy scripted --backend dummy --episodes 3 --json
```

Override discovery with `ROBODEPLOY_BENCHMARKS_ROOT=/path/to/benchmarks` or `robodeploy eval --benchmarks-root ...` when using a custom suite checkout.

---

## Quick verification checklist

```bash
# 1. Doctor
robodeploy doctor

# 2. Preset loads
python -m examples.cli list-presets | grep kuka_ft_imu_pick_mujoco

# 3. MuJoCo pick (headless)
python -m examples.cli run-episode --preset kuka_ft_imu_pick_mujoco --steps 1500 --json

# 4. Offline Gazebo pick regression (no gz required)
pytest tests/test_live_gazebo_pick_e2e.py -k offline -q

# 5. Reach DSL unit tests
pytest tests/test_reach_dsl.py -q
```

---

## Related docs

- [Backend setup](BACKEND_SETUP.md) — install details per simulator
- [Platform status](PLATFORM_STATUS.md) — CI maturity and known gaps
- [Tutorial 1 — Getting started](tutorials/01_getting_started.md) — first episode walkthrough
- [plans/V1_PICK_PLACE_DEMO_PLAN.md](../plans/V1_PICK_PLACE_DEMO_PLAN.md) — acceptance criteria and gap analysis
- [plans/INTEGRATION_STATUS.md](../plans/INTEGRATION_STATUS.md) — CI ↔ preset honesty table
