# Cross-simulator pick parity — progress

Plan: `.cursor/plans/cross-simulator_pick_parity_914a780a.plan.md`  
Loop: `AGENT_LOOP_TICK_PICK_PARITY` every 5m — **stopped** (PID 27060 exited after tick **43**, ~3.6h, exit code -1; killed by Claude Code session to avoid concurrent writers).

## Snapshot @ Claude Code session 2026-06-11 (post-IK fix) — ALL THREE PASS HONESTLY

| Backend | Status | Notes |
|---------|--------|-------|
| MuJoCo (Win) | **PASS step 466** | native MuJoCo IK, unchanged |
| RViz (WSL 24.04) | **PASS step 544 — honest, no snap** | was timeout@1500 / failure@1256 |
| Gazebo (WSL 24.04) | **PASS step 1406, 0.0379 m — honest, no snap** | beats old 0.085 m snap-off best |
| Tests | pick_parity + scene_parity + reach_dsl + pin_ik green | pin_ik 6/6 in WSL |

**Root causes fixed (Claude Code session 2026-06-11):**
1. `KinematicsSolver.jacobian` returned **all zeros** (discarded `computeFrameJacobian` return; `getFrameJacobian` reads buffers only `computeJointJacobians` fills). Every pin-IK solve was a no-op → frozen EE, joint-limit clamp spam, timeout/failure on RViz+Gazebo.
2. `PinIkSolver` rewritten to mirror `MujocoIkSolver`: position-only DLS, per-iteration joint-limit clamp, best-effort return (was 6-DOF orientation-coupled + return-q_init-on-failure freeze).
3. `pick_episode._gazebo_place_snap_enabled` called the policy instance method unbound → Gazebo demo died at startup behind the "Requires Linux" banner.
4. `test_ros2_rviz_place_finalize_snaps_source` flipped to `..._does_not_snap_source` (honest placement is the behavior now).

Note: loop ticks 32–43 claimed "WSL no ROS" — wrong distro probed; `Ubuntu` distro = 24.04 + Jazzy + `.venv-wsl` + pinocchio 4.0.0.

## Historical snapshot @ tick 40 (loop)

| Backend | Status | Last verified |
|---------|--------|---------------|
| MuJoCo (Win) | **PASS** step 466 | ticks 35–40 |
| Gazebo (WSL) | PASS step 1397 | tick 25–29 |
| RViz (WSL) | PASS step 1450 | tick 25–29 |
| Tests | 17/17 | tick 40 |
| Docker | blocked | — |
| WSL ROS smokes | blocked (no ROS) | tick 33 |

## Done (iter 43 — loop tick 43)

- [x] **MuJoCo** PASS — step **466** (seed 0)
- [x] Tests: **17/17** `test_pick_parity.py`

## Done (iter 42 — loop tick 42)

- [x] **MuJoCo** PASS — step **466** (seed 0)
- [x] Tests: **17/17** `test_pick_parity.py`

## Done (iter 41 — loop tick 41)

- [x] **MuJoCo** PASS — step **466** (seed 0)
- [x] Tests: **17/17** `test_pick_parity.py`

## Done (iter 40 — loop tick 40)

- [x] **MuJoCo** PASS — step **466** (seed 0)
- [x] Tests: **17/17** `test_pick_parity.py`
- [x] Loop tick **40** — Windows path stable since tick 30 fix; awaiting Docker/ROS for full re-smoke

## Done (iter 39 — loop tick 39)

- [x] **MuJoCo** PASS — step **466** (seed 0)
- [x] Tests: **17/17** `test_pick_parity.py`

## Done (iter 38 — loop tick 38)

- [x] **MuJoCo** PASS — step **466** (seed 0)
- [x] Tests: **17/17** `test_pick_parity.py`

## Done (iter 37 — loop tick 37)

- [x] **MuJoCo** PASS — step **466** (seed 0)
- [x] Tests: **17/17** `test_pick_parity.py`

## Done (iter 36 — loop tick 36)

- [x] **MuJoCo** PASS — step **466** (seed 0); matches tick 35
- [x] Tests: **17/17** `test_pick_parity.py`

## Done (iter 35 — loop tick 35)

- [x] **MuJoCo** PASS — `success=true` step **466** (seed 0; faster than prior 694-step runs)
- [x] Tests: **17/17** `test_pick_parity.py`
- [ ] Docker engine still stopped

## Done (iter 34 — loop tick 34)

- [x] **MuJoCo** PASS — step **694** (seed 0); ticks 30–34 stable on Windows
- [x] Tests: **17/17** `test_pick_parity.py`
- [ ] WSL / Docker unchanged (ROS missing, engine stopped)

## Done (iter 33 — loop tick 33)

- [x] **MuJoCo** PASS seeds **0–2** — all `success=true` step **694**
- [x] Tests: **17/17** `test_pick_parity.py`
- [ ] WSL: default `Ubuntu` + `Ubuntu-22.04` — no ROS install; RViz/Gazebo smokes blocked
- [ ] Docker engine still stopped (`docker-desktop` distro stopped)

## Done (iter 32 — loop tick 32)

- [x] **MuJoCo** PASS — step **694** (seed 0); fix stable across ticks 30–32
- [x] Tests: **17/17** `test_pick_parity.py`
- [ ] WSL RViz/Gazebo smokes skipped — WSL lacks `/opt/ros/jazzy` (ROS not installed in default distro)
- [ ] Docker engine still stopped

## Done (iter 31 — loop tick 31)

- [x] Re-verified **MuJoCo** PASS — `success=true` step **694** (seed 0); tick-30 `_map_mujoco_carry_mode` fix stable
- [x] Tests: **17/17** `test_pick_parity.py`
- [ ] Docker engine still stopped (`dockerDesktopLinuxEngine` pipe missing)

## Done (iter 30 — loop tick 30)

- [x] **MuJoCo regression fixed** — root cause: `_coerce_policy` fix applied preset `carry_mode: follow`; physics follow ejects cube during grasp (step ~245) before FT engage
- [x] **`_map_mujoco_carry_mode`** — `sensor_only` + follow → kinematic at bind (mirrors Gazebo/RViz adapters); **success=true step 694** seed 0
- [x] Tests: **17/17** `test_pick_parity.py`

## Done (iter 25–29 — loop ticks 25–29)

- [x] Re-verified **WSL Gazebo** PASS (step 1397) and **WSL RViz** PASS (step 1450) — no regressions
- [x] **MuJoCo** re-checked (Win, seed 0, `--steps 3200`) — still **truncates @1500**, `success=false`; source on floor (`z≈0.025`, xy off table)
- [x] Tests: **16/16** `test_pick_parity.py`
- [ ] MuJoCo bisect — likely physics/carry in `follow` + `sensor_only`, not Gazebo-only tuning

## Done (iter 20–24 — loop ticks 20–24)

- [x] **`_coerce_policy` fix** — nested `policy_kwargs` unwrapped; `kinematic` carry + place-snap now active at runtime
- [x] **WSL Gazebo smoke PASS** — `success=true` step **1397**, `source_to_goal_distance=0.0250` (after graph cleanup + `recovery_max_retries=12`)
- [x] **Failure root cause (prior runs):** `CONNECTION_LOST` @ ~798 from stale `ros_gz_bridge` / joint_states + joint-4 clamp; not policy logic
- [x] `wsl_gazebo_pick_smoke.sh` — pre-kill bridges/`gz sim` (mirrors RViz smoke)
- [x] Tests: **16/16** `test_pick_parity.py`
- [x] MuJoCo pick regression (fixed iter 30 — `_map_mujoco_carry_mode`)

## Done (iter 9)

- [x] **RViz root cause:** stale `ros_gz_bridge` / `/clock` publishers from concurrent Gazebo smokes → isolated fake-sim adopted sim time → `CONNECTION_LOST`
- [x] **RViz fix:** `ROS2RealBackend` — embedded `dev_fake_sim` without `config.sim` stays on wall time (ignores stray `/clock`); `wsl_rviz_pick_smoke.sh` pre-kills bridges
- [x] **WSL RViz smoke** (clean graph, seed 0): **success=true step 1450** (restored)
- [x] **WSL Gazebo smoke** (snap on, seed 0, `tracking_blend` 0.38→0.42): **success=false** dist **~0.208 m** @ 4000; probe @1800 `kinematic=False` `cache_source` unmoved — place snap not firing
- [x] **pytest:** `test_pick_parity.py` **16/16** (+ `test_rviz_isolated_fake_sim_ignores_stale_graph_clock`)
- [x] **MuJoCo smoke:** **REGRESSED** — seed 0 truncates @1500, cube knocked off table (Win + WSL); was step 373 iter 8
- [x] **Docker `docker info`:** CLI 29.1.5; **engine stopped** (`dockerDesktopLinuxEngine` pipe missing)

## Done (iter 8)

- [x] WSL Gazebo dist ~0.208 m; MuJoCo step 373; RViz regressed (concurrent Gazebo graph)
- [x] TF_OLD_DATA smoke stdout 0; RSP `publish_frequency=0`

## Next (priority)

- [x] **MuJoCo pick regression** — fixed via `_map_mujoco_carry_mode` (tick 30)
- [ ] Start Docker Desktop → `demo-rviz-pick` + `demo-gazebo-pick`
- [ ] Never run Gazebo + RViz WSL smokes concurrently (loop policy)

## Blockers

- **Docker Desktop engine stopped**
## Loop

- **Status:** STOPPED @ 2026-06-11 — PID **27060** killed after tick **43** (exit `4294967295` / -1; likely session/terminal close, not script error)
- **Last agent tick:** 43 — MuJoCo PASS step 466, tests 17/17
- **Restart:** `while ($true) { Start-Sleep -Seconds 300; Write-Output 'AGENT_LOOP_TICK_PICK_PARITY {...}' }` in repo root
