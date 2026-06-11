# V1 Pick-and-Place Demo Plan (MuJoCo / Gazebo / ROS2)

Written 2026-06-10. Target: working live demo **tomorrow (2026-06-11)** on the Windows demo
machine, followed by a clean, drift-free v1. Every finding below was verified against the
working tree at `main` (7809376); commands and file references are included so each item is
independently checkable.

---

## 1. Verified current state (evidence, not claims)

All of this was measured locally today (Windows 11, Python 3.14.2, mujoco 3.8.0):

| Check | Command | Result |
|---|---|---|
| Test suite | `pytest tests/ -m "not hardware" -q` | **2 failed**, 630 passed, 21 skipped (6m20s) |
| Asset integrity | `robodeploy assets verify` | **FAILS** — `kuka.xml`, `panda.xml` SHA mismatch |
| Headline MuJoCo pick | `python -m examples.cli run-episode --preset kuka_pick_mujoco --steps 1500` | **No success.** Cube knocked to (0.40, −0.14); episode times out; reported `failure: true` |
| FT-gated MuJoCo pick | `kuka_ft_imu_pick_mujoco`, seed 0, 1500-step budget | **E-stop at step 47**: `FORCE_LIMIT: |F|=58.9N > 50.0N` |
| ROS2 RViz pick preset | `kuka_sensor_ros2_rviz` (inspection) | **Cannot work as shipped** — see §2.3 |
| Gazebo pick | requires Linux/ROS2 Jazzy/Gazebo Harmonic | **Not runnable on this machine natively**; Docker image incomplete — see §2.4 |

Bottom line: **none of the three demos currently runs successfully end-to-end on the demo
machine.** The gap is concentrated and fixable, but real.

Important context: all MuJoCo end-to-end tests skip on Windows
(`tests/test_sensor_mujoco_integration.py:24` — blanket `_skip_on_windows`), so the local
green-looking suite proves nothing about the demo path. The CI job that asserts ≥80% pick
success runs only on Linux/py3.11.

---

## 2. Points of failure (ranked)

### 2.1 P0 — MuJoCo pick episodes fail on the demo machine

Two distinct failure modes, both reproduced:

1. **`kuka_ft_imu_pick_mujoco`: safety e-stop.** FT spikes from ~0.7N to 24N (step 40)
   to 58.9N (step 47) as the EE descends onto the cube; the default `SafetyMonitor` force
   guard (50N) trips and ends the episode. Contributing causes:
   - `ReachTrajectoryPolicy` re-solves IK only every 25 steps
     (`robodeploy/policies/reach_dsl.py:107`) and position actuators snap to the new goal
     → large transient velocities (observed up to ~11.5 rad/s) and hard impact.
   - Single-robot MuJoCo reset sets `qpos` to home but **not `ctrl`**
     (`robodeploy/backends/sim/mujoco/backend.py:653-659`); the multi-robot path does set
     `ctrl` (`_set_home_qpos_multi`, line 391). Until the first action arrives, actuators
     pull toward 0 → start-of-episode jerk.
   - No per-preset safety tuning: sim demo presets inherit the real-robot 50N force limit.
2. **`kuka_pick_mujoco`: grasp never engages.** The `example_sensor_reach_pick` policy
   pushes the cube away (final pos (0.40, −0.14) vs target (0.60, 0.20)); the episode runs
   the full budget and ends unsuccessful.

CI never catches either: PR CI runs `--steps 50` smoke only for `kuka_pick_mujoco`
(`.github/workflows/test.yml:48`) and the 80% success test covers only
`kuka_ft_imu_pick_mujoco` on Linux.

Suspected environment drift amplifier: demo machine is Python 3.14 (outside the supported
3.10–3.12 matrix); `jax` is part of the dev extra and the obs builders silently fall back
to numpy when jax is absent (`backend.py:408-411, 680-684`).

### 2.2 P0 — Asset manifest verification fails on Windows (2 test failures)

`git config core.autocrlf` is `true` and there is **no `.gitattributes`**, so Windows
checkouts rewrite `kuka.xml` / `panda.xml` to CRLF. SHA256 of the CRLF file no longer
matches `robodeploy/_assets/manifest.json` (verified: LF-normalized hash matches exactly).
Fails `tests/test_assets_manifest.py`, `tests/test_cli.py::test_assets_verify_sha256`, and
`robodeploy assets verify` (exit 1) on every Windows clone.

### 2.3 P0 — ROS2 (RViz) pick demo cannot work from presets

`RoboEnv.from_config` instantiates the backend from raw `backend_kwargs` only
(`robodeploy/env.py:325-332`). All ROS2 auto-wiring — `rviz.enabled`,
`robot_state_publisher`, `dev_fake_sim` joint publishers, per-robot topic/controller config
— lives **only** in `backend_for_simulator()` (`robodeploy/backends/simulator.py:66-116`).

The shipped ROS2 presets (`kuka_sensor_ros2_rviz`, `kuka_ft_imu_multimodal_ros2`) carry
**no** `backend_kwargs`, so the backend starts with an empty config: RViz disabled
(`_parse_backend_config` defaults `rviz_enabled=False`,
`robodeploy/backends/real/ros2/backend.py:581`), no fake joint sim, no external graph →
joint states never arrive → stale-state recovery → `SafetyError`.

The only working ROS2 path today is `examples/user_kuka_sinusoid/run_ros2_rviz.py
--fake-sim`, which goes through `backend_for_simulator` — and it is a sinusoid wave, not
pick-and-place.

### 2.4 P0 — Gazebo demo not runnable from this machine as shipped

- Requires Linux + ROS2 Jazzy + Gazebo Harmonic + `ros_gz_bridge` + `gz_ros2_control`
  (`examples/kuka_ft_imu_pick_gazebo/run_gazebo.py` docstring). Demo machine is Windows →
  WSL2 or Docker is mandatory.
- `docker/Dockerfile.ros2` installs `ros-jazzy-ros-gz` + bridge but **not**
  `ros-jazzy-gz-ros2-control` or `ros-jazzy-ros2-controllers` → controller spawning fails
  in-container. No service/command in `docker-compose.yml` runs the pick demo; no display
  strategy even though the preset sets `headless: false`.
- The pick world lives at `tests/fixtures/gazebo_pick_minimal.sdf` and is imported by the
  example (`examples/kuka_ft_imu_pick_gazebo/pick_episode.py:13`) — examples depend on the
  tests directory, which is absent from an installed package.
- Reliability: the live CI gate is ≥50% success over 10 seeds **with relaxed thresholds**
  (`RELAXED_POLICY_CONFIG`, `pick_episode.py:20-26`). A single live demo run is a coin flip.

### 2.5 P1 — Demo readout and viewer issues

- Timeout is reported as `failure: true` (`robodeploy/env.py:1242-1244` treats
  `step_count >= max_steps` as failure). A clean run that merely runs out of steps prints
  `success=false failure=true` — terrible optics in a live demo and semantically wrong.
- Every pick preset sets `enable_viewer: false` and `examples.cli run-episode` has **no
  `--viewer` flag** — there is currently nothing to look at in a live demo without editing
  YAML.
- If the viewer fails to open, the backend **raises** instead of degrading to headless
  (`mujoco/backend.py:266-272`).

### 2.6 P1 — Honest-but-easy-to-trip gaps (already documented in repo, listed for completeness)

- `manipulation_v1/pick_place_cube` dummy preset uses a reach policy → success rate ≈0
  (known gap in `plans/INTEGRATION_STATUS.md`); tiers 2–8 are placeholders.
- Gazebo/MuJoCo "grasp" is carry assist (kinematic follow / weld / prop-pose teleport via
  `PropPoseSyncer`), not physical gripper closure. Fine for v1 — must be stated on the
  demo slide, not discovered by the audience.
- `kuka_ft_imu_multimodal_real` preset requires a `perception_source` that no preset
  provides; `get_prop_pose` raises (`ros2/backend.py:388-395`).

---

## 3. Contract drift inventory

| # | Drift | Where | Severity |
|---|---|---|---|
| D1 | **Two construction paths disagree.** `from_config` (presets) skips everything `backend_for_simulator` does: behavior profiles, `control_hz`, RViz config, `dev_fake_sim`, Gazebo world derivation. Same preset semantics produce different runtime wiring depending on entry point. | `env.py:325` vs `backends/simulator.py:153` | High — root cause of §2.3 |
| D2 | Factory requires a Gazebo `world` (`simulator.py:237-241` raises) while the backend (and ARCHITECTURE.md) supports synthesizing a temp SDF when `sim.world` is omitted (`gazebo/backend.py:139-143`). | factory vs backend vs docs | Medium |
| D3 | `CONTRACTS.md` is encoded UTF-16LE without BOM — grep/diff/doc tooling reads it as garbage; the canonical contract file is effectively unreadable to tools. | `CONTRACTS.md` | Medium |
| D4 | Backend naming: registered `ros2_gazebo`, alias `gazebo`; presets use both spellings (`presets.yaml` uses `ros2_gazebo`, benchmark presets use `gazebo`). Plus `ros2` / `real_world` / `ros2_rviz` overlap for the ROS2 family (`registry.py:52, 296-300`). | registry/presets | Medium |
| D5 | Benchmarks depend on `examples/` (`benchmarks/manipulation_v1/pick_place_cube/task.py` re-exports `examples.tasks.pick_place`) but only `robodeploy*` is packaged (`pyproject.toml [tool.setuptools.packages.find]`). A pip-installed user cannot run the flagship benchmark. Contradicts the "library vs examples boundary" table in CONTRACTS.md. | benchmarks/examples/pyproject | High for v1 |
| D6 | Observation contract: `ee_velocity` / `ee_angular_velocity` are documented as required but MuJoCo always returns zeros (`mujoco/backend.py:694-695, 425-426`). | CONTRACTS.md vs backend | Medium |
| D7 | Timeout→`failure=true` semantics undocumented and misleading (§2.5). | `env.py:1242` | Medium |
| D8 | `docs/PLATFORM_STATUS.md` "MuJoCo pick-place → start here: `kuka_pick_mujoco`" recommends a preset whose success is never asserted anywhere and which fails locally (§2.1.2). Also "~620 passed" vs measured 632 (2 failed on Windows). | docs vs reality | Medium |
| D9 | Asset SHA manifest assumes LF; repo has no `.gitattributes` and Windows clones break it (§2.2). | `_assets/manifest.json` | High (Windows) |
| D10 | Examples import test fixtures (`pick_episode.py` → `tests/fixtures/*.sdf`); private task API leaks into examples (`task._placement_goal()` called from `pick_episode.py:100`). | examples/tests | Low–Medium |
| D11 | Local Python 3.14 vs supported/CI matrix 3.10–3.12; jax→numpy silent fallback changes the obs construction path between CI and demo machine. | pyproject/CI vs machine | Medium for demo |
| D12 | Dead/stale artifacts committed: retired `examples/kuka_pick_demo.py` stub, `output.png`, `output1.png`, `MUJOCO_LOG.TXT`, root-level `history.json`, overlapping plan docs (`BROAD_GOALS.md`, `REPRESENTATION_UPGRADE_PLAN.md`, `SENSOR_INTEGRATION_TODO.md`). | repo root | Low |

---

## 4. Clean-code debt (no behavior change, do after the demo)

1. **`env.py` (1308 lines)** mixes config coercion, episode loop, task-state assembly,
   sensor health, manifests, safety transitions. Split into `env_construction.py`
   (the `from_config`/`make`/coercers), the episode loop, and info/diagnostics assembly.
2. **`MuJoCoBackend` dual mode.** Single-robot and multi-robot paths duplicate obs
   building (`_build_obs` vs `_build_obs_for_state`), home-pose logic (with the `ctrl` bug
   divergence noted in §2.1), grasp state, and viewer setup. Make multi the only code path;
   single robot = list of one.
3. **Gazebo backend defensive `getattr` style** (`getattr(self, "_scene_prop_poses", {})`
   et al. throughout `gazebo/backend.py`) hides initialization-order bugs. Initialize all
   attributes in `__init__`/`initialize_multi` and access them directly.
4. **Custom YAML include splicer** (`examples/config/__init__.py:67-109` concatenates raw
   file text so anchors resolve across files) is the most fragile code in the demo path.
   Replace with explicit fragment loading + dict merge (no cross-file YAML anchors).
5. **Duplicate fake joint sims**: `robodeploy/backends/real/ros2/dev/fake_joint_sim.py`,
   `robodeploy/ros2/devtools/fake_jointpos_sim.py`, and
   `examples/user_kuka_sinusoid/ros2_fake_jointpos_sim.py`. Keep one devtool.
6. **`examples/_bootstrap.py` sys.path hack** in every example — replace with documented
   `pip install -e .` requirement plus an `examples` package install or guard message.
7. Root-level clutter and retired stubs (D12) — delete.
8. Nine `cli_*.py` top-level modules — move under a `robodeploy/cli/` package for
   discoverability (mechanical, low risk).

---

## 5. High-level goals for V1

- **G1 — One command, one visible, successful pick-place demo per backend** (MuJoCo,
  Gazebo, ROS2/RViz), each verified ≥9/10 seeds on the actual demo hardware.
- **G2 — Single construction path**: presets and programmatic construction produce
  identical backend wiring; `backend_for_simulator` becomes the only place that knows
  per-backend auto-config.
- **G3 — Tests green on the demo machine**, with MuJoCo e2e no longer blanket-skipped on
  Windows and pick success asserted in PR CI (small seed count).
- **G4 — Contract docs match code** (CONTRACTS.md readable UTF-8, drift items D1–D11
  closed or explicitly documented as limitations).
- **G5 — Clean code**: §4 items done; no module over ~600 lines in the demo path; no
  examples→tests imports; no dead stubs.

---

## 6. Track A — Demo readiness (today, in order)

> Scope discipline: tune configs and add small flags; do not refactor anything today.

### A1. Unbreak Windows test signal (~30 min)
- Add `.gitattributes`: `*.xml -text`, `*.urdf -text`, `*.stl -text`, `*.sdf -text` (at
  minimum for `robodeploy/description/**` and `ros2_assets/**`), then renormalize
  (`git add --renormalize .`).
- Alternative belt-and-braces: hash with `\r\n`→`\n` normalization in
  `robodeploy/assets.py`.
- Acceptance: `robodeploy assets verify` exit 0; the 2 failing tests pass.

### A2. Make MuJoCo pick succeed and look good (core of the day)
- Create a dedicated demo venv on **Python 3.12** (match CI; removes the 3.14/jax
  variable). `py -3.12 -m venv .venv-demo && pip install -e ".[sim,dev]"`.
- Tune until `kuka_ft_imu_pick_mujoco` passes 10/10 seeds locally, in this order of
  preference (stop when green):
  1. Slow the approach: in `reach_pick_place.yaml`, lower `blend`/`tracking_blend`
     (e.g. 0.22 → 0.10) and add an intermediate descend phase (offset 0.10 → 0.04 →
     0.015); re-solve IK every step or every 5 steps instead of 25
     (`reach_dsl.py:107`) — config knob, not refactor, if possible.
  2. Seed `ctrl` with home on single-robot reset (one-line fix in `_set_home_qpos`,
     mirrors the multi path) — small, safe, fixes the start-of-episode jerk.
  3. Per-preset safety config: raise the sim demo force limit (e.g. 150N) via
     `safety` config in the preset rather than weakening the global default.
- Add `--viewer` flag to `examples.cli run-episode` that overrides
  `backend_kwargs.config.enable_viewer=true`; make viewer failure degrade to headless
  with a warning instead of raising (`mujoco/backend.py:266-272`).
- Acceptance: `python -m examples.cli run-episode --preset kuka_ft_imu_pick_mujoco
  --viewer --steps 1500` shows the arm picking and placing the cube, `success=true`,
  10/10 seeds headless.

### A3. ROS2 (RViz) pick demo (preset-only fix)
- The backend already reads `dev_fake_sim` and `rviz` from its config dict — the wiring
  gap is purely that the presets don't provide them. Add a `kuka_pick_ros2_rviz` preset
  with `backend_kwargs.config` containing: `rviz: {enabled: true, fixed_frame: world}`,
  `dev_fake_sim: [{robot_ns: /robot0, joint_names: [...], publish_hz: 100}]`, and the
  joint topic keys (`robot0.joint_names`, `robot0.joint_states_topic`,
  `robot0.joint_cmd_topic`) — copy values from `_ros2_auto_config`
  (`backends/simulator.py:66-116`).
- Run inside WSL2 (rclpy on Windows is not practical). RViz renders via WSLg.
- Acceptance: RViz shows the arm executing the pick trajectory with scene markers;
  episode completes without `SafetyError`; sensor status `ok`.
- Fallback if time runs out: `python -m examples.user_kuka_sinusoid.run_ros2_rviz
  --fake-sim` (known-working) + narrate that pick-place preset lands in v1 — only if the
  preset route genuinely fails.

### A4. Gazebo pick demo (WSL2 preferred over Docker)
- WSL2 Ubuntu 24.04: install ROS2 Jazzy + Gazebo Harmonic + `ros-jazzy-ros-gz`
  + `ros-jazzy-gz-ros2-control` + `ros-jazzy-ros2-controllers`; `pip install -e
  ".[sim,kinematics,dev]"`.
- Also patch `docker/Dockerfile.ros2` (add the two missing packages) so the container
  path works as backup; add a `demo-gazebo-pick` compose service running
  `python -m examples.kuka_ft_imu_pick_gazebo.run_gazebo`.
- Copy `tests/fixtures/gazebo_pick_minimal.sdf` →
  `robodeploy/ros2_assets/worlds/pick_minimal.sdf` and point `pick_episode.py` at the
  packaged path (keep the tests fixture as a thin copy or import).
- Rehearse: `run_pick_episodes(range(10))`; record success rate. If <8/10, apply the
  same approach-speed tuning as A2 (thresholds are already relaxed — prefer slowing the
  trajectory over relaxing further).
- Acceptance: one full successful episode with `headless: false` rendering in WSLg, plus
  a measured local success rate to quote honestly.

### A5. Insurance + runbook (end of day)
- Record screen video of one successful run per backend today. If anything breaks live
  tomorrow, play the recording — never debug live.
- Write `docs/DEMO_RUNBOOK.md`: exact commands per backend, env activation, expected
  output, the honest caveats (scripted reach policy + carry assist, not learned
  grasping; Gazebo success rate X/10), and the fallback order.
- Do **not** demo: benchmarks tiers 2–8, real-hardware presets, Isaac.

---

## 7. Track B — V1 hardening (this week, after the demo)

| # | Item | Closes | Acceptance |
|---|---|---|---|
| B1 | Route `from_config` backend construction through `backend_for_simulator` (preset `backend:` name → simulator name + overrides). Delete per-preset hand-copied wiring added in A3. | D1, §2.3 | Same preset works identically via CLI and API; ROS2 presets need no hand-rolled `dev_fake_sim` |
| B2 | Timeout semantics: report `truncated` (or `timeout: true` in `info.extra`) instead of `failure=true`; document in CONTRACTS.md. | D7, §2.5 | Demo readout clean; tests updated |
| B3 | CI: assert pick success in PR CI (3 seeds MuJoCo); un-skip MuJoCo e2e on Windows (keep vision-only skips, gate on GLFW availability not platform); pin `mujoco` version; add a Windows MuJoCo job. | §2.1 blind spot | A regression in pick success fails PR CI on both OSes |
| B4 | Packaging: move the pick-place demo task+policy YAML into `robodeploy` proper (e.g. `robodeploy.tasks.demo`) or package `examples`; benchmarks must not import `examples.*` or `tests.*`. | D5, D10 | `pip install robodeploy && robodeploy eval --benchmark manipulation_v1/pick_place_cube --backend mujoco` works from a clean venv |
| B5 | Gazebo: hit the WAVE2_01 ≥70%/10-seed gate with **unrelaxed** thresholds; align factory vs backend on world synthesis (drop the factory `ValueError`, D2). | §2.4, D2 | live CI gate raised to 0.7; factory accepts omitted world |
| B6 | Observation contract: populate `ee_velocity`/`ee_angular_velocity` via `mj_objectVelocity`; fix single-robot `ctrl` home properly (if A2 used the safety-limit workaround); document remaining optional-zeros explicitly. | D6 | contract test asserting nonzero ee velocity under motion |
| B7 | Docs/contracts hygiene: convert CONTRACTS.md to UTF-8; update PLATFORM_STATUS numbers and the recommended pick command; document the single backend naming story (`gazebo` canonical, `ros2_gazebo` internal). | D3, D4, D8 | grep-able CONTRACTS.md; status doc matches a fresh `pytest` run |
| B8 | ROS2 real preset honesty: either wire a `perception_source` example (color blob or TF) into `kuka_ft_imu_multimodal_real` or mark it template-only in presets and docs. | §2.6 | preset either runs against fake graph or refuses with actionable message |

## 8. Track C — Clean-code pass (after B1–B3 land)

Execute §4 in this order (each PR independently green):
1. C1: split `env.py`; unify MuJoCoBackend single/multi paths (B6 makes this safer).
2. C2: replace the YAML include splicer with explicit fragment merging; add loader tests
   for include-order and anchor-free behavior.
3. C3: Gazebo backend attribute initialization cleanup (drop `getattr` defaults).
4. C4: consolidate fake joint sims into `robodeploy/ros2/devtools/`.
5. C5: delete dead stubs and root clutter (D12); fold stale root plan docs into `plans/`.
6. C6: move `cli_*.py` into `robodeploy/cli/` package.

---

## 9. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| MuJoCo tuning doesn't converge to 10/10 today | Medium | The `ctrl`-home fix + slower approach almost certainly clears the 50N trip; worst case raise sim force limit per-preset and note it |
| Gazebo live flake during demo | High (gate is 50%) | Rehearsed seed pinned the night before; recorded video fallback; quote measured rate honestly |
| WSL2/Jazzy install eats the day | Medium | Start the WSL install **first** (it downloads in background while doing A2); Docker path as plan B with the two missing apt packages |
| ROS2 preset fix exposes more wiring gaps | Medium | Fallback to `backend_for_simulator`-based script (mirror `user_kuka_sinusoid/run_ros2_rviz.py` but with the pick task) — bypasses `from_config` entirely |
| Python 3.14 oddities resurface | Low after A2 | Demo venv pinned to 3.12; never run the demo from the 3.14 interpreter |

## 10. What the demo honestly shows (say this out loud)

- One task definition (`pick_place`), one policy spec (reach DSL YAML), three backends —
  the actual value proposition.
- Grasping is scripted carry-assist (kinematic follow / weld), not learned manipulation —
  learned policies are a v1.x milestone (`WAVE2_03/04`).
- Benchmarks: tier-1 reach is real; tier-2+ are placeholders being wired to this demo
  path (B4).
