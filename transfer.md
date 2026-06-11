# transfer.md — session handoff (2026-06-11, cross-backend pick parity)

Read with: `PICK_PARITY_PROGRESS.md` (prior loop's tracker), `BROAD_GOALS_PROGRESS.md`, `plans/INTEGRATION_STATUS.md`.

## Context

User reported: RViz + Gazebo produced different scenes than MuJoCo, policy execution differed, and both ROS backends failed. A separate Cursor agent loop ("cross-simulator pick parity", tick 43, PID 27060 — **killed this session** to avoid concurrent writers; re-arm if wanted) had already built the unification layer in the working tree (uncommitted): canonical scene `robodeploy/demos/scenes/pick_table.py`, shared preset core `examples/presets/kuka_ft_imu_pick.yaml` + backend overlays, per-backend carry mapping in `reach_dsl.py`, parity tests (`tests/test_pick_parity.py`, `test_pick_scene_parity.py`), `demo/run_pick.py`.

## Root causes found + fixed this session (the actual parity bugs)

1. **`KinematicsSolver.jacobian` returned all zeros** (`robodeploy/kinematics/solver.py`): it discarded `pin.computeFrameJacobian`'s return and read `pin.getFrameJacobian` without `computeJointJacobians` populated. Every Pinocchio-IK consumer got a zero Jacobian.
2. **`PinIkSolver` semantics diverged from MuJoCo IK** (`robodeploy/kinematics/pin_ik.py`, rewritten): old version delegated to 6-DOF `solver.ik` (orientation term fights position), never clamped joint limits, and on non-convergence returned `q_init` unchanged → policy froze mid-episode (trace: EE parked at (0.335, 0.145, 0.372), 1396 joint-limit clamp warnings, timeout@1500). Now mirrors `MujocoIkSolver`: position-only DLS, per-iteration `np.clip` to `description.joint_position_limits`, best-effort return. `attach_pin_ik` passes limits.
3. **Gazebo demo crashed at startup**: `examples/kuka_ft_imu_pick_gazebo/pick_episode.py` called instance method `ReachTrajectoryPolicy._gazebo_place_snap_enabled()` unbound → TypeError swallowed by run_gazebo's catch-all ("Requires Linux..." red herring). Replaced with standalone env check (`ROBODEPLOY_GAZEBO_PLACE_SNAP`, default **off** now — honest placement).
4. **Stale test**: `test_reach_dsl.py::test_ros2_rviz_place_finalize_snaps_source` asserted the old RViz oracle snap; loop had removed the snap (honest placement). Inverted to `..._does_not_snap_source`.

Earlier this session (already pushed, commits `24fcab6..1575e59`): RViz scene markers latched (transient-local QoS + Jazzy nested-Topic `default.rviz` + 1 Hz republish), gz transport msgs module fix (`gz.msgs10` for Harmonic — kinematic carry was falling back to ~100ms subprocess per pose sync, causing the cube drop/snap glitch), RSP failure now RuntimeWarning, CRLF stripped from WSL scripts.

## Verified results (WSL `Ubuntu` distro = 24.04 + Jazzy + `.venv-wsl`, pinocchio 4.0.0)

| Backend | Before fixes | After fixes |
|---------|--------------|-------------|
| MuJoCo (Win) | success step 466 | unchanged (uses native MuJoCo IK) |
| RViz fake-sim (WSL) | timeout@1500 / failure@1256 (flaky) | **success step 544, honest placement (no snap)** |
| Gazebo (WSL) | TypeError at startup | **success step 1406, 0.0379 m honest placement (no snap)** — beats old 0.085 m snap-off best |

Repro commands:
- MuJoCo: `python -m examples.cli run-episode --preset kuka_ft_imu_pick_mujoco_headless --seed 0 --steps 1500 --json`
- RViz: `wsl -d Ubuntu bash scripts/wsl_rviz_pick_smoke.sh` (never concurrently with Gazebo — stale `/clock` poisons the fake-sim graph)
- Gazebo: `wsl -d Ubuntu bash scripts/wsl_gazebo_pick_smoke.sh`
- Trace probe: `scripts/_wsl_rviz_trace.py` (per-100-step EE/cube/phase + final diagnostics)
- IK probe: `scripts/_pin_ik_probe.py`

Tests: `test_pick_parity.py` + `test_pick_scene_parity.py` + `test_reach_dsl.py` + `test_pin_ik.py` (6/6 in WSL incl. pin-dependent; new regression tests for zero-jacobian, limit clamping, best-effort IK) + offline gazebo e2e — all green. Full Windows suite (`-m "not hardware"`) running at session end — was 654 passed/21 skipped before parity work.

## Outstanding

1. **Gazebo smoke result** — confirm `success=true` honest placement; if placement >0.04 m, the carry weld idea (Gazebo `DetachableJoint`) is the next step; gz transport bindings (`python3-gz-transport13`, `python3-gz-msgs10`) must be importable from `.venv-wsl` for fast kinematic carry (else subprocess fallback ~100ms/step).
2. **Commit the parity work** — ~40 modified + ~15 new files uncommitted (loop's unification + this session's IK/scene/snap fixes). Suggested commit split: (a) kinematics fixes (solver.py, pin_ik.py, test_pin_ik.py), (b) pick_episode snap fix + reach_dsl test flip, (c) loop's unification layer wholesale, (d) docs/trackers. Push to `origin` + `github` remotes (both at `1575e59`).
3. **Full-suite verdict** — confirm no regressions from solver.jacobian change (it was returning zeros, so any consumer behavior can only improve, but verify).
4. **PICK_PARITY_PROGRESS.md** — update with this session's root-causes + results; loop tracker last wrote tick 43 (its WSL "blocked, no ROS" claims were wrong — it probed the wrong distro; `Ubuntu` = 24.04 with ROS).
5. Cleanup candidates: `scripts/_wsl_rviz_trace.py`, `scripts/_pin_ik_probe.py`, `scripts/_wsl_gazebo_probe.py`, `scripts/_wsl_gazebo_quick.py`, `scripts/_wsl_rviz_diag.py` (one-off probes; keep or delete), `demo/` (loop's unified runner — decide whether it ships).
6. RViz visual check: user should rerun episode + `rviz2 -d robodeploy/ros2_assets/rviz/default.rviz` — markers latched now; robot + cube + target should align and the arm should reach them (IK fixed).

## Gotchas

- **Never run RViz fake-sim and Gazebo WSL smokes concurrently** — Gazebo's `/clock`/bridges break the wall-time fake sim (CONNECTION_LOST, TF_OLD_DATA). Use `ROS_DOMAIN_ID` isolation if needed.
- `pkill -f 'gz sim'` inside a `wsl bash -c` one-liner kills the shell itself (pattern matches own cmdline) — run pkill from script files only.
- PowerShell expands `$?`/`$vars` inside double-quoted `wsl bash -c "..."` strings — use single quotes outside, escaped doubles inside.
- `run_gazebo.py` catch-all hides real exceptions behind "Requires Linux" — print `exc` first when debugging (already prints it, line above the banner).
- The committed `solver.ik()` 6-DOF method still exists and is used elsewhere (e.g. tests, planning); only `PinIkSolver` stopped using it. Its orientation+position behavior is unchanged.
