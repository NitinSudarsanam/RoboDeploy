# transfer.md — session handoff (2026-06-11, iteration 17)

Read with: `BROAD_GOALS.md` (strategic index), `BROAD_GOALS_PROGRESS.md` (tracker + iteration log), `plans/INTEGRATION_STATUS.md` (CI honesty), `plans/GOAL_*.md` (per-goal acceptance criteria).

## State at handoff

- Branch `main`, working tree **clean**, everything committed:
  - `e8db35a` — iterations 1–16 work (demo parity, `robodeploy.demos` packaging, Docker/WSL scripts, docs). This had been sitting **uncommitted** across 16 iterations.
  - `7a3d4c4` — goal 9 LOC refactor: `robomimic.py`/`diffusion.py`/`vla.py` = **50/49/43** lines; extractions in `policies/learned/helpers.py` (`ActionSmoother`, `PlanQueue`, `build_plan`, `batch_first_actions`, `vla_packet`, `vla_heuristic_action`, `select_camera_image/depth`); LOC regression test in `tests/test_learned_policy_base.py`.
  - `76f4601` — acceptance closure: tutorial-02 exec test + interface docstring audit (`tests/test_tutorial_02_task.py`); GOAL_01/02/07/09 checkboxes ticked with evidence.
  - `0a67aa9` — leaderboard submit test now writes to tmp (was rewriting `benchmarks/leaderboard/submissions/.../cli_test_*.json` every suite run).
  - Final commit (this one) — tracker iteration 17 + GOAL_04 ticks + this file.
- Full suite `pytest -m "not hardware"`: **654 passed, 21 skipped** (~6m20s) post-refactor.
- MuJoCo demo re-verified: `python -m examples.cli run-episode --preset kuka_ft_imu_pick_mujoco --seed 0 --steps 2000 --json` → `success=true` step **306**.
- Stray `pick_out*.json` deleted and gitignored.

## BROAD_GOALS parity verdict

~**95%** of acceptance criteria. **Zero software-only gaps remain actionable from this machine.** The 16 unchecked boxes across `plans/GOAL_*.md` all need external resources:

| Blocker | Boxes | What unblocks |
|---------|-------|----------------|
| PyPI `v0.2.0` tag + trusted publishing | 3 (goals 7/8) | Repo admin: tag `v0.2.0`; `publish.yml` is ready; `scripts/pypi_dry_run.ps1` + `twine check` already green |
| Isaac Sim GPU Kit runtime | 7 (goal 6) | Self-hosted runner / GPU workstation — `docs/BACKEND_SETUP.md#isaac-sim-self-hosted-ci` |
| Live teleop devices | 5 (goal 4) | Keyboard interactive session, SpaceMouse hardware, live ROS2 twist rehearsal |
| Dashboard | 1 (goal 10) | Deferred by sign-off (`robodeploy/observability/DASHBOARD_DEFERRAL.md`) |

Plus (tracked in tracker "Done vs deferred"): WSL Ubuntu **24.04** install for interactive RViz (current WSL is 22.04; `scripts/wsl24-bootstrap.sh` ready); Gazebo honest JTC sub-0.04 m is an **accepted limitation** (default `ROBODEPLOY_GAZEBO_PLACE_SNAP=1`; best honest 0.085 m, iter 14).

## Next session: pick-up points (priority order)

1. **PyPI release** (user action): configure trusted publishing, push `v0.2.0` tag, then tick the 3 publish boxes and the `pip install robodeploy` quickstart box in GOAL_07/08.
2. **WSL 24.04** (user action, interactive): `wsl --install -d Ubuntu-24.04` → `bash scripts/wsl24-bootstrap.sh` → rehearse `kuka_pick_ros2_rviz` interactively (Docker headless path already PASS at step 950).
3. **Isaac GPU**: on a Kit-capable machine run the goal 6 acceptance list (IMU sensor, capsule prop, `.usd` load, multi-robot, ≤1mm SceneIR pose round-trip).
4. **Gazebo honest JTC** (optional, accepted limitation): if sub-0.04 m without snap is ever required, the remaining idea is a physics-level carry weld (not the kinematic oracle) — see iter 15/16 notes; `prefer_fk_ee_pose` driver hook exists but is off by default.

## Gotchas for the next agent

- The full suite leaves no repo churn anymore, but if you see `benchmarks/leaderboard/submissions/**` modified after a test run, an old checkout of `tests/test_benchmarks.py` is being used — the fix is in `0a67aa9`.
- `train eval` in the robodeploy CLI is **dummy-only by design**; non-dummy checkpoint eval goes through `robodeploy eval --policy ckpt.pt --backend mujoco` (CI: `tests/training/test_train_eval_benchmark_e2e.py`).
- Duplicate task registration raises `KeyError` — tests exec'ing tutorial code substitute a unique registry name (see `tests/test_tutorial_02_task.py`).
- `.gitattributes` pins LF for text assets so `robodeploy/_assets/manifest.json` SHA256 hashes are platform-stable; don't fight it with `core.autocrlf` overrides.
- Windows: MuJoCo env-build tests and PPO smoke are intentionally skipped (`platform.system() != "Windows"` guards); the 21 skips are expected.
