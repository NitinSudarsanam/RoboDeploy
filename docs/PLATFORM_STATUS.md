# Platform status (v0.2)

Honest summary of what RoboDeploy v0.2 delivers today, what CI proves, and what remains planned. Updated for **`main`** (Wave 2 integration merged, 2026-06-10).

For contributor-level job tables, see [plans/INTEGRATION_STATUS.md](../plans/INTEGRATION_STATUS.md).

---

## At a glance

| Metric | Value |
|--------|-------|
| Version | **0.2.0** (not yet on PyPI) |
| Tests | **~620 passed**, ~19 skipped (`pytest -m "not hardware"`) |
| Strategic goals | **~65%** honest completion (12 goal plans) |
| Primary sim | MuJoCo (fully exercised in CI) |
| Live Gazebo | Linux sensor + pick E2E (relaxed thresholds) |

---

## What works well

### Core runtime

- `RoboEnv` reset/step/close with multi-robot MuJoCo
- Registry-based wiring (`from_config`, presets, `use()`)
- Action-space negotiation (e.g. DELTA_EE → JOINT_POS)
- Safety monitor + filters on step and reset paths
- Obs pipeline with sync, noise, color-blob → `obs.objects`

### Backends

| Backend | Dev ready | CI |
|---------|-----------|-----|
| MuJoCo | Yes | Full sensor e2e, pick presets, benchmarks |
| Dummy | Yes | All platforms |
| Gazebo | Linux smoke + pick E2E | `sensor-live-gazebo` |
| ROS2 RViz | Yes | Live sensor test (Linux) |
| Isaac Sim | Partial | Mock import smoke |
| Real ROS2 | Partial | Hardware markers only |

### Learning stack

- Gymnasium adapter + registered envs
- BC and PPO trainers, `SubprocVecEnv`
- `robodeploy train bc|ppo|eval` CLI
- `examples/train_ppo_reach.py`, `train_ppo_kuka_pick.py`
- Train → benchmark eval E2E test (dummy BC)

### Benchmarks

- `manipulation_v1` harness (HTML, video, leaderboard schema)
- Tier 1 `reach_target` on dummy (and short MuJoCo PR smoke)
- Placeholder tiers 2–8 labeled honestly in `spec.json`

### Distribution

- `package-build` CI (sdist/wheel + `twine check`)
- Docker CPU smoke
- Conda recipe metadata smoke
- Plugin entry-point discovery test

---

## What is partial or planned

| Capability | Gap |
|------------|-----|
| **PyPI** | `publish.yml` ready; no `v0.2.0` tag published |
| **Teleop / datasets** | Contract + keyboard stub; no full record→train path |
| **Gazebo pick** | CI ≥50% over 10 seeds; production target 70% |
| **500k PPO** | Script exists; nightly runs 50k proxy (`continue-on-error`) |
| **Isaac GPU** | No org GPU runner; obs parity unchecked live |
| **Real hardware eval** | `sim2real` presets manual |
| **Benchmark tiers 2–8** | Reach scripted placeholders, not real manipulation |
| **Dashboard** | Deferred per `DASHBOARD_DEFERRAL.md` |

---

## CI you can trust

| Job | Proves |
|-----|--------|
| `unittest` matrix | Core library on Win/macOS/Linux, Py 3.10–3.12 |
| `sensor-e2e-linux` | MuJoCo sensors, vision integration, MuJoCo sensor 80% pick |
| `eval-mujoco-smoke` | 3-episode MuJoCo `reach_target` eval |
| `sensor-live-gazebo` | Multimodal obs + pick E2E (`live_gazebo`) |
| `package-build` | Wheel installs and CLI runs |
| `benchmark.yml` nightly | Dummy suite N=5, schema validation, Pages deploy |

**Does not prove:** full 100-episode MuJoCo baselines nightly, Gazebo manipulation suite, Isaac live physics, PyPI consumer install without cloning repo.

---

## Recommended paths by goal

| I want to… | Start here |
|------------|------------|
| Try without sim | `robodeploy run-episode --dummy --steps 10` |
| MuJoCo pick-place | `python -m examples.cli run-episode --preset kuka_pick_mujoco` |
| Train PPO | `python examples/train_ppo_reach.py --backend dummy` |
| Run benchmarks | `robodeploy eval --benchmark manipulation_v1/reach_target --backend dummy` |
| Gazebo sensors | [BACKEND_SETUP.md](BACKEND_SETUP.md) + `kuka_ft_imu_pick_gazebo` |
| Ship a release | [RELEASE.md](RELEASE.md) |
| Understand architecture | [PROJECT_GUIDE.md](PROJECT_GUIDE.md) |

---

## Roadmap

**Strategic goals (12):** [plans/README.md](../plans/README.md)

**Wave 2 follow-ups:**

| Plan | Focus |
|------|-------|
| [WAVE2_01](../plans/WAVE2_01_GAZEBO_LIVE_E2E.md) | Gazebo pick ≥70%, tuning |
| [WAVE2_02](../plans/WAVE2_02_RELEASE_PYPI_CONDA.md) | PyPI + conda-forge |
| [WAVE2_03](../plans/WAVE2_03_VISION_AND_REAL_HARDWARE.md) | Vision pick E2E, real smoke |
| [WAVE2_04](../plans/WAVE2_04_TRAINING_PRODUCTION.md) | 500k PPO gates |
| [WAVE2_05](../plans/WAVE2_05_POLISH.md) | Policy LOC, Isaac GPU playbook |
