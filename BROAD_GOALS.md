# Broad Goals — strategic index

> **Historical snapshot (2026-06-08).** Many gaps listed below are **partially or fully addressed on `main` v0.2** — e.g. `robodeploy/training/`, `benchmarks/`, `safety/`, sensor-aware policies, scaffold CLI, and Docker/PyPI workflows exist. Do **not** treat the "Signals" bullets as current repo state.
>
> **Use instead:**
> - [plans/INTEGRATION_STATUS.md](plans/INTEGRATION_STATUS.md) — CI ↔ preset honesty
> - [docs/PLATFORM_STATUS.md](docs/PLATFORM_STATUS.md) — user-facing maturity
> - [plans/README.md](plans/README.md) — per-goal plan files with acceptance criteria
> - [SENSOR_INTEGRATION_TODO.md](SENSOR_INTEGRATION_TODO.md) — sensor audit (software ~complete)
> - [REPRESENTATION_UPGRADE_PLAN.md](REPRESENTATION_UPGRADE_PLAN.md) — DSL/builder roadmap

Strategic gaps that originally motivated the v0.2 goal plans. Ranked by ROI.

## Tier 1 — Highest Impact

### 1. Cut Representation Boilerplate
New task = 86 lines. New policy = 261 lines. Scene = per-backend Python. Blocks every new user. Builders + DSLs (`SceneBuilder`, `RewardBuilder`, reach YAML, task templates) cut 60-70% friction. See `REPRESENTATION_UPGRADE_PLAN.md`.
- Signals: `examples/policies/reach_pick_place.py` (261 lines, 8 hardcoded phases); `examples/tasks/pick_place.py` (86 lines) duplicated in `peg_insertion.py`, `pour.py`; per-backend `scene_builder.py` files.

### 2. Build Real Training Loop
Repo is inference + sim env. No RL/IL/BC harness. No batched sim. `SequentialVecEnv` only. Users must bolt on Dreamer/ACT/OpenRL externally. Add `robodeploy/training/` with at minimum BC loop, PPO loop, dataset adapters.
- Signals: `robodeploy/vec_env.py` sequential; no `training/` module; no loss/metric tracking; CI runs no training checks.

### 3. Wire Sensors Into Policies + Tasks
Sensors collected but unused. Policies query `backend.has_prop_contact()`, not `obs.ft_force`. No vision-based termination. No IMU stability gate. Tasks stay oracle-bound. See `SENSOR_INTEGRATION_TODO.md`.
- Signals: `reach_pick_place.py:220-226` backend queries; zero `obs.ft_force` consumers; `tasks/base.py:56` physics query.

## Tier 2 — Adoption Blockers

### 4. Teleop + Data Collection
No keyboard, spacemouse, VR, gamepad, web GUI. Cannot collect real-world demos for IL/BC. Imitation learning story = dead. Add `robodeploy/teleop/` with keyboard + spacemouse + gello at minimum.
- Signals: zero grep hits for "teleop"/"gamepad"/"spacemouse"; `demo_session()` only records hardcoded policies.

### 5. Sim2Real Pipeline
Randomization framework exists. Calibration, DR sweep, transfer-validation metrics missing. No documented "train sim → deploy real" workflow. No sim2real benchmark.
- Signals: `tasks/randomization.py` noise-only; no calibration module; `SO101_REAL.md` manual; no transfer-gap test.

### 6. Backend Parity (IsaacSim, Gazebo)
MuJoCo primary. IsaacSim lacks IMU + live CI. Gazebo lacks procedural terrain, grasp weld parity, lighting presets. Forces MuJoCo dependence.
- Signals: missing `sensors/imu/sim/isaacsim_imu.py`; `backends/sim/gazebo/` no terrain builder; CI mocks Isaac.

## Tier 3 — Production + Ecosystem

### 7. Documentation + Onboarding
Architecture docs exist. No task/policy/scene creation guides. No `robodeploy scaffold` CLI. New user faces 261-line policy as reference.
- Signals: `docs/` has only `BACKEND_SETUP.md` + `SO101_REAL.md`; no creation tutorials; no CLI scaffolder.

### 8. Multi-Robot + Distribution
Multi-robot backend methods exist (`initialize_multi`, `step_multi`). No runnable example. Package 0.1.0, not on PyPI. No Docker, no conda. No plugin entry points.
- Signals: `examples/multiagent_configs.py` structure-only; `pyproject.toml` 0.1.0; no `docker/`; no entry-point plugins.

### 9. Learned Policy Integration
VLAPolicy/DiffusionPolicy/RobomimicPolicy = thin loader shims. No standard checkpoint resolution. No action-space mismatch detection between policy + backend actuators.
- Signals: `policies/learned/*.py` require user `predict_fn`; `Action` dataclass loosely typed; no `validate_action_space()` helper.

### 10. Observability + Reproducibility
`sensor_status` collected, not surfaced. No wandb/tensorboard sinks. No trajectory checkpoint/replay. Seeding partial (env only, not policy). Determinism not tested.
- Signals: `obs_pipeline.py:77` unreported; no `env.get_diagnostics()`; no replay API; no determinism asserts in tests.

### 11. Evaluation + Benchmarking
No standardized task suite, no metric harness, no leaderboard. Cannot compare policies. Research credibility limited.
- Signals: no `benchmarks/` dir; no eval CLI; tasks lack standard difficulty tiers.

### 12. Safety + Error Recovery (Real Hardware)
Real backend lacks watchdog, e-stop hook, joint-limit guard, force-limit cutoff. Production deploy risky.
- Signals: `backends/real/ros2/` no safety module; no force-limit policy wrapper.

---

## Ranked Priority

| Rank | Goal | Effort | ROI |
|---|---|---|---|
| 1 | Cut representation boilerplate | ~100h | Huge — every user |
| 2 | Training loop (BC + PPO + dataset) | ~80h | Unlocks RL/IL story |
| 3 | Sensor → policy/task integration | ~40h | Unlocks contact-aware control |
| 4 | Teleop + data collection | ~50h | Unlocks IL data |
| 5 | Sim2real pipeline + calibration | ~60h | Unlocks real deploy claim |
| 6 | Backend parity (Isaac IMU, Gazebo terrain) | ~40h | Cross-sim credibility |
| 7 | Docs + scaffolder CLI | ~30h | Cuts onboarding |
| 8 | Multi-robot example + PyPI + Docker | ~40h | Ecosystem |
| 9 | Learned policy + action-space validation | ~25h | VLA story |
| 10 | Observability + replay + determinism | ~30h | Research workflows |
| 11 | Benchmarks + eval harness | ~50h | Comparability |
| 12 | Real-hw safety + recovery | ~30h | Production trust |

Total: ~575 hours.

## Suggested Roadmap

**Phase 1 (foundation, ~150h)**: rep boilerplate (1) + sensor wiring (3) + docs (7). Cuts user friction 60%.

**Phase 2 (learning story, ~180h)**: training loop (2) + teleop (4) + sim2real (5). Repo becomes learning platform, not just sim wrapper.

**Phase 3 (ecosystem, ~130h)**: backend parity (6) + multi-robot/PyPI (8) + learned policy (9). External adoption ready.

**Phase 4 (production, ~110h)**: observability (10) + benchmarks (11) + safety (12). Research + prod credibility.
