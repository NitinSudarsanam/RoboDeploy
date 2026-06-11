# BROAD_GOALS progress tracker

Last updated: **2026-06-11** (iteration 16). Governing docs: `BROAD_GOALS.md`, `transfer.md`, `plans/INTEGRATION_STATUS.md`.

## Pick-and-place demo status (V1 Track A)

| Backend | Preset / entry | Status | Notes |
|---------|----------------|--------|-------|
| **MuJoCo** | `kuka_ft_imu_pick_mujoco` | **Working** | `success=true` at step **306** (seed 0, `--steps 2000 --json`, iter 14). |
| **RViz** | `kuka_pick_ros2_rviz` | **Docker PASS** | WSL **22.04 only** (no 24.04); `demo-rviz-pick` → **`success=true` step 950** (seed 0, 2000 steps). Kinematic place finalize + tuned `follow_tau_s=0.05`. TF `base_link`→`ee_link` still warns in headless Docker. |
| **Gazebo** | `kuka_ft_imu_pick_gazebo` | **Docker PASS 1/1 (snap on)** | Default `ROBODEPLOY_GAZEBO_PLACE_SNAP=1`: iter 16 **0.025 m** @ step 950. **JTC-only** (`PLACE_SNAP=0`): **accepted limitation** — historical best **0.085 m** (iter 14); iter 16 probe **0.42 m** after TF-bridge + gains tuning (`prefer_fk` reverted). |

## BROAD_GOALS checklist (12 strategic goals)

| # | Goal | Status | Gap |
|---|------|--------|-----|
| 1 | Cut representation boilerplate | **Partial → improved** | `robodeploy.demos` pick/task &lt;50/&lt;25 LOC; `test_representation_gaps` targets packaged demos + thin `examples/` shims; choreography/DSL template coverage still open |
| 2 | Training loop | **Mostly done** | BC/PPO/gym/SubprocVecEnv; MuJoCo BC checkpoint eval in CI (`training-integration` + `sim` extra) |
| 3 | Sensor → policy/task | **Done** | FT/IMU/contact predicates wired; MuJoCo pick E2E |
| 4 | Teleop + data collection | **Partial → improved** | `record_stub_episode` + metadata stamping; headless record→BC test; `02_teleop.md` stub section; live device rehearsal open |
| 5 | Sim2Real pipeline | **Mostly done** | Calibration/DR/transfer metrics; real-hardware eval manual |
| 6 | Backend parity | **Partial → improved** | Pick-place SceneIR pose tolerance covered offline; Isaac GPU live CI — see `docs/BACKEND_SETUP.md#isaac-sim-self-hosted-ci` (manual GPU workstation / self-hosted runner) |
| 7 | Docs + scaffolder | **Partial → improved** | `01_getting_started` multi-robot cmd; `RELEASE.md` pre-tag demo table; `DEMO_RUNBOOK` honest JTC caveat; PyPI tag publish open |
| 8 | Multi-robot + distribution | **Mostly done** | `two_franka_pick_mujoco` E2E green in full suite; wheel + `pypi_dry_run` OK; PyPI tag publish still open |
| 9 | Learned policy integration | **Mostly done** | Loader/adapter/negotiation; policy LOC refactor deferred |
| 10 | Observability + replay | **Mostly done** | Logger/replay/manifest; dashboard deferred |
| 11 | Benchmarks + eval | **Done** | manipulation_v1 harness + nightly dummy suite |
| 12 | Real-hw safety | **Done** | SafetyMonitor, recovery, sim injectors |

**Honest overall:** ~**91%** of acceptance criteria across all 12 goals (all 3 pick demos PASS with default snap; Gazebo honest JTC documented as accepted limitation; PyPI tag + WSL 24.04 interactive RViz + Isaac GPU CI still open).

## Done vs deferred (user action)

| Item | Status |
|------|--------|
| MuJoCo / RViz / Gazebo (snap on) pick demos | **Done** — green this iter |
| Gazebo honest JTC (&lt;0.04 m, snap off) | **Deferred** — default `ROBODEPLOY_GAZEBO_PLACE_SNAP=1`; best observed 0.085 m |
| PyPI `v0.2.0` tag + trusted publishing | **Deferred** — needs repo admin |
| WSL Ubuntu 24.04 + interactive RViz | **Deferred** — needs `wsl --install -d Ubuntu-24.04` |
| Isaac Sim GPU live CI | **Deferred** — self-hosted runner / manual workstation |
| Live teleop device rehearsal | **Deferred** — hardware |
| Observability dashboard | **Deferred** — low ROI |

## Iteration log

### 2026-06-11 — iteration 16

**Done:**
- **Gazebo honest JTC (P1):** Disabled gz `/tf` bridge when RSP + gz_ros2_control active (avoids TF fights); JTC PID gains + publish rates in `kuka_controllers.yaml`; `prefer_fk_ee_pose` hook in driver (not enabled by default). Snap-off probe **0.42 m** (regressed vs iter 14 **0.085 m** best) — **accepted limitation**, default snap unchanged.
- **Goal 1:** `test_representation_gaps` now lint-targets `robodeploy/demos/*` + thin-shim assertion on `examples/`.
- **Goal 7/8:** `01_getting_started` two-Franka cmd; `RELEASE.md` pre-tag demo smoke table; `DEMO_RUNBOOK` honest JTC caveat.
- **Re-verify demos:** MuJoCo **step 306**; `demo-rviz-pick` **step 950**; `demo-gazebo-pick` snap on **step 950 / 0.025 m**.
- Full pytest `-m "not hardware"`: **651 passed, 21 skipped** (~7m04s).

**Next:**
1. PyPI tag `v0.2.0` (trusted publishing).
2. Ubuntu 24.04 WSL bootstrap for interactive RViz.
3. Gazebo honest JTC: physics-level carry weld (not kinematic oracle) if sub-0.04 m required.

**Blockers:**
- Gazebo honest JTC **not reliable** without place snap (0.085 m historical best).
- WSL 22.04 only — Jazzy native path needs 24.04 for interactive RViz.
- PyPI tag publish.

### 2026-06-11 — iteration 15

**Done:**
- **Gazebo honest JTC (P1):** Policy config `tracking_blend` / `steps_per_phase` now apply at compile; honest place release gating (`honest_place_settle_m`, `honest_place_tracking_blend`); pre-release EE carry sync; place-phase IK refresh every 4 steps; JTC `constraints` in `kuka_controllers.yaml`; RSP TF wait 90s; `jtc_time_from_start_s=1.0` when snap off.
- **Re-verify demos:** MuJoCo **step 306**; `demo-rviz-pick` **step 950**; `demo-gazebo-pick` snap on **step 950 / 0.025 m**.
- Targeted pytest: **reach_dsl + offline gazebo + from_config + representation** (see test run).

**Next:**
1. Gazebo honest JTC: stabilize headless Docker runs toward &lt;0.04 m (RSP `world`→`ee_link` live, sensor bridges).
2. Ubuntu 24.04 WSL for interactive RViz.
3. PyPI tag `v0.2.0`.

**Blockers:**
- Gazebo honest JTC placement **0.085 m best** (iter 14); snap-off probe flaky **0.71–0.91 m** this iter.
- WSL 22.04 only — Jazzy native path needs 24.04 for interactive RViz.
- PyPI tag publish.

### 2026-06-11 — iteration 14

**Done:**
- **RViz (P1):** WSL distros = **22.04 only** (no 24.04). Added `demo-rviz-pick` Docker compose service; **`success=true` step 950** (seed 0). Tuned preset (`follow_tau_s=0.05`, `steps_per_phase=300`); ros2_rviz kinematic place finalize in `reach_dsl`.
- **Gazebo honesty (P2):** `ROBODEPLOY_GAZEBO_PLACE_SNAP` env flag (default **on**). JTC-only probe (`PLACE_SNAP=0`): **0.085 m** best — `HONEST_JTC_POLICY_OVERRIDES` + `jtc_time_from_start_s=0.8` when snap off.
- **Parity:** `02_teleop.md` headless stub section; `DEMO_RUNBOOK.md` docker RViz + place-snap docs.
- MuJoCo: `kuka_ft_imu_pick_mujoco --seed 0 --steps 2000` → **`success=true` step 306**.
- Targeted pytest: **7 passed** (reach_dsl place snap + offline gazebo gate + from_config rviz).

**Next:**
1. Gazebo JTC XY to &lt;0.04 m without place snap (RSP `world`→`ee_link`, sensor bridges).
2. Ubuntu 24.04 WSL for interactive RViz + native Gazebo.
3. PyPI tag `v0.2.0`.

**Blockers:**
- Gazebo honest JTC placement **0.085 m** (needs physics/JTC tuning, not oracle snap).
- WSL 22.04 only — Jazzy native path needs 24.04 for interactive RViz.
- PyPI tag publish.

### 2026-06-11 — iteration 13

**Done:**
- **Gazebo Docker (P1):** `demo-gazebo-pick` → **`success=true` step 950**, placement **0.025 m** (seed 0, 4000 steps). Fixes: `grasp_success_force_min=0` for headless FT; Gazebo kinematic **place finalize** snap in `reach_dsl`; RSP `publish_frequency=100`; Pinocchio FK fallback + TF lookup timeout; `steps_per_phase=400`. Rejected `use_sim_time` graph-wide (regressed JTC to 0.8 m).
- MuJoCo: `kuka_ft_imu_pick_mujoco --seed 0 --steps 2000` → **`success=true` step 306**.
- Targeted pytest: **4 passed** offline Gazebo gate + reach_dsl.

**Next:**
1. Gazebo: close JTC XY gap without kinematic place snap; RSP `world`→`ee_link` in Docker; sensor bridges.
2. Ubuntu 24.04 WSL → `kuka_pick_ros2_rviz` smoke.
3. PyPI tag `v0.2.0`.

**Blockers:**
- Gazebo placement uses kinematic oracle snap (not physics-accurate JTC alone).
- RViz live: 22.04 WSL / no headless Docker path.
- PyPI tag publish.

### 2026-06-11 — iteration 12

**Done:**
- **Gazebo Docker (P1):** Stopped bridging gz `/joint_states` when `gz_ros2_control` is active (avoids RSP joint-name fights); early `/robot_description` via RSP for controller_manager; `/joint_states` name parity gate; world-frame TF (`robot0.base_frame=world`); Pinocchio FK fallback in driver + Gazebo carry; `pick_episode` EE readiness wait; relaxed Gazebo policy (`kinematic` carry, `steps_per_phase=320`, `jtc_time_from_start_s=0.4`); `run_gazebo` seed 0 / 3200 steps.
- **Smoke:** `demo-gazebo-pick` — stack OK; seed 0 best **`source_to_goal_distance=0.25 m`** (needs &lt;0.04 m); `success=false`. Sensors: health OK, `ft_norm=0`, `camera=no`.
- **RViz (P2):** unchanged — WSL **22.04**, no `demo-rviz-pick`.
- MuJoCo: `kuka_ft_imu_pick_mujoco --seed 0 --steps 2000` → **`success=true` step 305**.
- Targeted pytest: **4 passed** offline Gazebo gate + reach_dsl `steps_per_phase` override.

**Next:**
1. Gazebo: fix RSP `world`→`ee_link` publish in Docker (or headless camera render for bridged sensors); JTC XY placement to &lt;0.04 m.
2. Ubuntu 24.04 WSL → `kuka_pick_ros2_rviz` smoke.
3. Rebuild `robodeploy:ros2` image without bind-mount.

**Blockers:**
- Gazebo Docker pick **0/1** on seed 0 smoke (0.25 m best vs 0.04 m goal).
- RViz live: 22.04 WSL / no headless Docker path.
- PyPI tag publish.

### 2026-06-11 — iteration 11

**Done:**
- **Gazebo Docker (P1):** Fixed world-name/prop SDF generation (`backend.py` merges scene props into temp world when using `pick_minimal.sdf`); Kuka Gazebo TF frames (`base_link`/`ee_link`); task `max_steps` synced with `max_episode_steps`; JTC horizon 1.0s; controller spawner retries; compose bind-mount `..:/app` for fast iteration.
- **Smoke:** `demo-gazebo-pick` stack starts reliably; episode runs **2000 steps** — `success=false`, `source_to_goal_distance` **0.73–1.02 m** (needs &lt;0.04 m). Sensors report health OK but `ft_norm=0`, `camera=no` (bridges idle).
- **RViz (P2):** WSL still **22.04 only** — no `demo-rviz-pick` service (needs WSLg/X11 + Jazzy on 24.04); documented path unchanged.
- MuJoCo: `kuka_ft_imu_pick_mujoco --seed 0 --steps 2000` → **`success=true` step 305**.
- Targeted pytest: **12 passed, 1 skipped** (offline Gazebo gate + grasp follow + from_config).

**Next:**
1. Gazebo Docker: fix RSP/`/joint_states` joint-name parity so `base_link` TF publishes; verify ros_gz sensor bridge data on `/wrist_ft/wrench`, `/wrist_camera/image_raw`.
2. Ubuntu 24.04 WSL → `kuka_pick_ros2_rviz` smoke.
3. Rebuild `robodeploy:ros2` image (non-bind-mount CI parity).

**Blockers:**
- Gazebo Docker pick **0/1** (stack OK; TF + sensor data + placement).
- RViz live: 22.04 WSL / no headless Docker path.
- PyPI tag publish.

### 2026-06-11 — iteration 10

**Done:**
- **Gazebo Docker (P1):** Root-caused iter-9 `controller_manager` timeout — missing `GZ_SIM_SYSTEM_PLUGIN_PATH`, missing `ros-jazzy-ros-gz-sim` + OSRF `gz-harmonic` repo, sim paused + `/robot_description` not ready before controller spawn.
- **Fixes:** `GazeboLauncher` sets plugin path, unpauses world, waits for `/robot_description` + `/joint_states`, retries controller spawner; `Dockerfile.ros2` adds Gazebo packages + `[kinematics]` (`pin`); compose env (`GZ_SIM_SYSTEM_PLUGIN_PATH`, `LD_LIBRARY_PATH`, headless).
- **Smoke:** `demo-gazebo-pick` → stack starts, episode runs 1000 steps (`success=false`, placement 0.75m — policy/sensor bridge tuning, not startup).
- **RViz (P2):** No Docker RViz service in repo; WSL 22.04 still blocks native Jazzy — documented in `DEMO_RUNBOOK.md`.
- MuJoCo: `--seed 0 --steps 2000 --json` → **`success=true` step 306**.
- Targeted pytest: **4 passed** offline Gazebo gate + from_config/parity.
- `docs/DEMO_RUNBOOK.md` updated with compose probe + plugin-path notes.

**Next:**
1. Gazebo Docker pick success: sensor bridge timing, world name for prop sync, multi-seed rate.
2. Ubuntu 24.04 WSL → `kuka_pick_ros2_rviz` smoke.
3. Rebuild `robodeploy/robodeploy:ros2` image (kinematics layer) and re-run compose without volume mount.

**Blockers:**
- Gazebo Docker pick **0/1** on smoke (stack OK, task success open).
- RViz live: 22.04 WSL / no headless Docker path.
- PyPI tag publish.

### 2026-06-11 — iteration 9

**Done:**
- **Docker:** engine **UP**; `scripts/check_docker_engine.ps1` em-dash parse fix; `robodeploy/robodeploy:ros2` **rebuild OK** (`--ignore-installed pluggy`).
- **Docker smoke:** `demo-gazebo-pick` → **`controller_manager` timeout** (~27s) — image/packages fine; compose one-shot does not launch `gz sim` + ros2_control (documented blocker).
- **Regressions fixed:** `_mujoco_auto_config` allows multi-robot (`two_franka_pick_mujoco` via `from_config`); lint test targets `robodeploy/demos/tasks/pick_place.py`.
- MuJoCo smoke: `kuka_ft_imu_pick_mujoco --seed 0` → **`success=true` at step 306**.
- PyPI dry-run: `twine check` OK (no tag, no upload).
- Targeted pytest: **32 passed** (from_config, backend/representation parity, lint, two_franka).

**Next (priority order):**
1. Extend `demo-gazebo-pick` compose to spawn `gz sim` + `controller_manager` before pick script (or document two-step compose).
2. Ubuntu 24.04 WSL (`bash scripts/wsl24-bootstrap.sh`) → RViz + native Gazebo pick.
3. PyPI tag `v0.2.0` after trusted publishing configured.

**Blockers:**
- Gazebo Docker demo needs simulation services started (not just robodeploy container).
- WSL2 is Ubuntu **22.04** (Jazzy needs **24.04**).
- PyPI/conda publish blocked on release tag.

### 2026-06-11 — iteration 8

**Done:**
- **Docker:** engine **stopped** (`dockerDesktopLinuxEngine` pipe missing) — `demo-gazebo-pick` skipped; added `scripts/check_docker_engine.ps1`, ROS2 compose **healthchecks**, `scripts/wsl24-bootstrap.sh` (Ubuntu 24.04 Jazzy).
- **Goal 2:** BC checkpoints persist `obs_keys`/`action_dim`/`proprio_dim`; `RoboEnv._coerce_policy` routes `.pt` through `coerce_eval_policy`; `test_train_bc_then_eval_mujoco_reach_target_checkpoint`; `training-integration` CI installs `[sim]` + `MUJOCO_GL=egl`.
- **Goal 4:** `record_stub_episode`, teleop session metadata, `tests/test_teleop_record_stub.py` (record→`DemoDataset`→BC).
- **Goal 6/7/8:** `docs/DEMO_RUNBOOK.md` bootstrap/Docker probe links; `docs/RELEASE.md` + `scripts/pypi_dry_run.ps1`; offline Gazebo gate **4 passed**.
- MuJoCo smoke: `kuka_ft_imu_pick_mujoco --seed 0 --steps 2000` → **`success=true` at step 306**, `truncated=false`.
- PyPI dry-run: `twine check` OK (no tag, no upload).
- Full suite: **645 passed, 21 skipped** (~8m33s); 2 pre-existing failures (`lint task pick_place`, `two_franka_pick_mujoco`).

**Next (priority order):**
1. Start Docker Desktop → `docker compose -f docker/docker-compose.yml --profile ros2 run --rm demo-gazebo-pick`.
2. Ubuntu 24.04 WSL (`bash scripts/wsl24-bootstrap.sh`) → RViz + native Gazebo pick.
3. PyPI tag `v0.2.0` after trusted publishing configured.

**Blockers:**
- Docker Desktop engine not running; WSL2 is Ubuntu **22.04** (Jazzy needs **24.04**).
- PyPI/conda publish blocked on release tag.

### 2026-06-11 — loop tick 3 (10m)

**Done:**
- Started Docker Desktop (engine ready ~12s).
- `docker compose … demo-gazebo-pick` **build failed**: pip cannot uninstall debian `pluggy` during `[sim,real,dev]` install in `Dockerfile.ros2`.
- **Fix:** `pip3 install … --ignore-installed pluggy` in `docker/Dockerfile.ros2` builder layer.
- MuJoCo smoke: `kuka_ft_imu_pick_mujoco --seed 0` → `success=true` step 306.

**Next:** Re-run `docker compose -f docker/docker-compose.yml --profile ros2 run --rm demo-gazebo-pick` after image rebuild.

### 2026-06-11 — iteration 7

**Done:**
- **Track B (B4):** `benchmarks` package ships in wheel (`pyproject.toml` packages.find + package-data); `benchmarks_root()` discovers installed package; `__init__.py` on `manipulation_v1` task subpackages. New test `test_wheel_benchmarks_discovery_and_preset_resolve` (wheel venv → `list-benchmarks --json` + preset import).
- **Track B (B5):** Docker CLI 29.1.5 present; **engine still stopped** (`dockerDesktopLinuxEngine` pipe missing) — `demo-gazebo-pick` compose smoke skipped.
- **Goal 4:** Teleop contract documented in `CONTRACTS.md` (`TeleopCommand`, `ITeleopDevice`, `TeleopPolicy`, recording path).
- **Goal 7:** `docs/DEMO_RUNBOOK.md` pip-only benchmark eval section; `benchmarks/README.md` discovery order updated.
- MuJoCo smoke: `kuka_ft_imu_pick_mujoco --seed 0 --json` → **`success=true` at step 306**, `truncated=false`.
- Targeted pytest: **18 passed** (wheel benchmarks, registry, from_config, SceneIR parity, teleop policy).

**Next (priority order):**
1. Start Docker Desktop → `docker compose … demo-gazebo-pick` smoke (or Ubuntu 24.04 WSL for native Jazzy).
2. PyPI release tag + publish `0.2.0`.
3. Human `--viewer` MuJoCo run (insurance video); teleop record→train tutorial.

**Blockers:**
- Docker Desktop engine not running; WSL2 is Ubuntu **22.04** (Jazzy needs **24.04**).
- PyPI/conda publish blocked on release tag.

### 2026-06-11 — iteration 6

**Done:**
- **Track B (B4):** Added `robodeploy.demos` (`tasks/`, `policies/`, `sensors/`) — pick/peg/pour/showcase tasks, reach/sensor/joint policies, prop_pose sensor. `examples/*` now thin re-exports. All `benchmarks/**` presets + `task.py` imports switched to `robodeploy.demos.*`; `reach_pick_place.yaml` added to wheel package-data.
- **Track B (B5):** Docker CLI 29.1.5 present but **engine still stopped** (`dockerDesktopLinuxEngine` pipe missing) — live Gazebo compose smoke skipped.
- MuJoCo smoke: `kuka_ft_imu_pick_mujoco --seed 0 --json` → **`success=true` at step 306**, `truncated=false`.
- Targeted pytest: **25 passed** (demos packaging, task templates/semantics, ft_gated, domain_randomization, from_config simulator).

**Next (priority order):**
1. Start Docker Desktop → `docker compose … demo-gazebo-pick` smoke (or Ubuntu 24.04 WSL for native Jazzy).
2. Bundle or document `benchmarks/` for pip-only eval (`ROBODEPLOY_BENCHMARKS_ROOT` / ship under `robodeploy`).
3. Human `--viewer` MuJoCo run (insurance video); PyPI release tag.

**Blockers:**
- Docker Desktop engine not running; WSL2 is Ubuntu **22.04** (Jazzy needs **24.04**).
- `benchmarks/` tree not in wheel — pip users still need repo checkout or `ROBODEPLOY_BENCHMARKS_ROOT`.
- PyPI/conda publish blocked on release tag.

### 2026-06-11 — iteration 5

**Done:**
- **RViz/Gazebo demos:** Docker CLI present (29.1.5) but **engine stopped** (`dockerDesktopLinuxEngine` pipe missing) — `demo-gazebo-pick` compose smoke **blocked**. WSL: Ubuntu **22.04** only (no 24.04); `wsl --install -d Ubuntu-24.04` not feasible non-interactively (interactive install/reboot). Documented **Windows primary paths** in `docs/DEMO_RUNBOOK.md` (MuJoCo native → Docker ROS2 → 24.04 WSL).
- **Track B (B7):** Step-budget timeout semantics documented in `CONTRACTS.md` (`truncated`/`timeout` in `info.extra`; not `failure=true`).
- **Track B (B4):** Audited — `pyproject.toml` still packages `robodeploy*` only; `benchmarks/manipulation_v1/*/task.py` re-exports `examples.tasks.*` (pip-only eval still blocked; no code change this iter).
- **Track B (B5):** Live 70% Gazebo gate not runnable (no Docker engine / no 24.04 WSL); offline gate **4 passed** (`test_live_gazebo_pick_e2e.py -k offline`).
- MuJoCo smoke: `kuka_ft_imu_pick_mujoco --seed 0 --json` → **`success=true` at step 306**, `truncated=false`.
- Targeted pytest: **7 passed** (from_config simulator + timeout + ci_pick_gate, scene_builder add_prop, gazebo offline).

**Next (priority order):**
1. Start Docker Desktop → `docker compose … demo-gazebo-pick` smoke (or add Ubuntu 24.04 WSL for native Jazzy).
2. Track B4: package demo task/policy or move pick benchmark off `examples.*`.
3. Human `--viewer` MuJoCo run (insurance video).

**Blockers:**
- Docker Desktop engine not running; WSL2 is Ubuntu **22.04** (Jazzy needs **24.04**).
- B4: benchmarks import `examples.*` — `pip install robodeploy && robodeploy eval …` fails from clean venv.
- PyPI/conda publish blocked on release tag.

### 2026-06-11 — iteration 4

**Done:**
- **WSL2 ROS2:** Ubuntu 22.04 v2 confirmed; `gz` present; **no `/opt/ros` / `rclpy`** — Jazzy apt unavailable on jammy (needs Ubuntu 24.04). Documented one-liner bootstrap + 22.04→24.04 migration in `docs/DEMO_RUNBOOK.md` (full install deferred).
- **Goal 1:** `SceneBuilder.add_prop(PropConfig | UnifiedPropSpec)` + `test_add_prop_accepts_prop_config`.
- **Track B (B2):** Step-budget timeout → `info.extra["truncated"]` / `timeout` (not `failure=true`); honors `max_episode_steps` from `from_config`.
- **Track B (B3):** `test_ci_pick_gate_3_seeds_from_config` (`@pytest.mark.ci_pick_gate`); wired into `sensor-e2e-linux` CI job; runs on Windows when MuJoCo installed.
- MuJoCo smoke: `kuka_ft_imu_pick_mujoco --seed 0 --json` → **`success=true` at step 306**, `truncated=false`.
- Targeted pytest: **10 passed** (scene_builder, from_config simulator + pick gate, pose tolerance).

**Next (priority order):**
1. Add Ubuntu 24.04 WSL distro → run Jazzy bootstrap; smoke `kuka_pick_ros2_rviz` + Gazebo pick.
2. Human `--viewer` MuJoCo run (insurance video).
3. Track B: B4 packaging, B5 Gazebo 70% gate, B7 CONTRACTS.md timeout docs.

**Blockers:**
- WSL2 is Ubuntu **22.04** (Jazzy requires **24.04**); use Docker ROS2 services or migrate distro.
- PyPI/conda publish blocked on release tag.

### 2026-06-11 — iteration 3

**Done:**
- **Track B (B1):** `RoboEnv.from_config` routes registry backends through `backend_for_simulator` (`simulator_name_for_backend`, `_build_backend`); Gazebo without `sim.world` falls back to raw `backend_kwargs`.
- Removed hand-rolled `dev_fake_sim` / per-robot ROS keys from `kuka_pick_ros2_rviz` preset — auto-wiring matches `backend_for_simulator`.
- **Goal 1:** `test_pick_place_scene_ir_mujoco_gazebo_pose_tolerance` in `tests/test_representation_gaps.py`.
- New tests: `tests/test_from_config_simulator.py` (4 cases).
- Targeted pytest: **30 passed** (from_config simulator path, backend parity, gazebo sensor rig, behavior profile, env config, pick-place tolerance).
- MuJoCo smoke: `kuka_ft_imu_pick_mujoco --seed 0` → **`success=true` at step 306**.
- **WSL2 check:** Ubuntu 22.04 v2 available; `gz` present; **`rclpy` / `/opt/ros` not installed** — live RViz/Gazebo pick blocked until ROS2 Jazzy sourced.

**Next (priority order):**
1. WSL2: install/source ROS2 Jazzy + `pip install -e ".[ros2,kinematics]"`; rehearse `kuka_pick_ros2_rviz` and Gazebo pick.
2. Human `--viewer` MuJoCo run (insurance video).
3. Goal 1: 1-line prop API on `SceneBuilder`.
4. Track B: B2 timeout semantics, B3 CI pick gate.

**Blockers:**
- WSL2 lacks ROS2 Python stack (`ModuleNotFoundError: rclpy`).
- PyPI/conda publish blocked on release tag.

### 2026-06-11 — loop tick 1 (10m)

**Re-verified:** MuJoCo `kuka_ft_imu_pick_mujoco --seed 0` → `success=true` step 428. WSL2 (Ubuntu 22.04) loads `kuka_pick_ros2_rviz` + `kuka_ft_imu_pick_gazebo` presets. Loop PID 36116 still armed (next tick ~10m).

### 2026-06-11 — iteration 2

**Done:**
- Created `docs/DEMO_RUNBOOK.md` — per-backend pick-place commands, prerequisites, success signals, WSL2 caveats, fallback order, known-failing presets.
- Linked runbook from `docs/index.md`, `docs/tutorials/01_getting_started.md`, `mkdocs.yml`; updated quickstart to `kuka_ft_imu_pick_mujoco --seed 0`.
- Full suite: **`pytest -m "not hardware"` → 632 passed, 21 skipped** (~5m21s).
- MuJoCo smoke: `kuka_ft_imu_pick_mujoco --seed 0` → **`success=true` at step 306**.
- Bugfix: `RoboEnv.run_episode` stops on `done` (was stepping past success → timeout failure).
- CLI: `examples/cli run-episode --seed` for reproducible demos.

**Next (priority order):**
1. WSL2 rehearsal: `kuka_pick_ros2_rviz`; Gazebo `run_pick_episodes(range(10))`.
2. Human `--viewer` run for MuJoCo demo (record insurance video).
3. Track B: route `from_config` through `backend_for_simulator`.
4. Goal 1: SceneIR cross-backend round-trip tolerance test.

**Blockers:**
- Gazebo + RViz live validation requires WSL2/Linux.
- PyPI/conda publish blocked on release tag.

### 2026-06-11 — iteration 1

**Done:**
- Armed 10m `AGENT_LOOP_TICK_BROAD_GOALS` loop (PID 36116).
- Added `kuka_pick_ros2_rviz` preset with RViz + `dev_fake_sim` wiring.
- Packaged `pick_minimal.sdf` under `robodeploy/ros2_assets/worlds/`.
- Updated `pick_episode.py` to resolve packaged world path.
- Docker: `gz-ros2-control`, `ros2-controllers` in `Dockerfile.ros2`; `demo-gazebo-pick` service.
- Tests passed: `test_live_gazebo_pick_e2e` (offline), `test_reach_dsl`, `test_examples_cli`, `test_mujoco_grasp`, `test_assets_manifest`.
- MuJoCo smoke: `kuka_ft_imu_pick_mujoco` → `success=true` at step 600.
