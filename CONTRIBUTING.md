# Contributing

RoboDeploy is in active development (v0.2 beta on `main`). Keep changes small enough to review and preserve the main boundary: user code builds a `Robot` and `RoboEnv`; backends adapt that contract to a simulator or real hardware.

**Read first:**

| Doc | Purpose |
|-----|---------|
| [docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md) | Platform overview |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Layer diagram and principles |
| [CONTRACTS.md](CONTRACTS.md) | Public API contracts |
| [plans/INTEGRATION_STATUS.md](plans/INTEGRATION_STATUS.md) | What CI proves vs claims |
| [history.json](history.json) | Machine-readable gap list |

## Repository layout

```text
robodeploy/       Installable library — see module table below
examples/         Presets, demo tasks/policies (not shipped on PyPI)
benchmarks/       manipulation_v1, sim2real, leaderboard schema
tests/            pytest suite (~620 tests, `not hardware`)
docs/             MkDocs source (guides, tutorials)
plans/            GOAL_0N + WAVE2 strategic plans
.github/workflows/  CI: test.yml, benchmark.yml, docs.yml, publish.yml
docker/           Dockerfile.cpu for smoke
conda-recipe/     conda-forge metadata
```

### `robodeploy/` module layout

| Directory | Add code here when… |
|-----------|---------------------|
| `backends/` | Integrating a simulator or ROS2 hardware stack |
| `core/` | Shared types, registry, robot model, sensor rig |
| `description/` | New robot URDF/MJCF assets and metadata |
| `tasks/` | Reusable task base, templates, predicates (not one-off demos) |
| `policies/` | Reusable policy base, learned loaders, reach DSL |
| `sensors/` | New sensor modality with sim + real pair |
| `obs_pipeline/` | Observation transforms shared across backends |
| `training/` | BC/PPO, datasets, Gym registration |
| `evaluation/` | Benchmark harness, metrics, reports |
| `safety/` | Guards and monitor wiring |
| `sim2real/`, `calibration/` | Transfer and calibration tooling |
| `teleop/` | Input devices and recording (IL path) |
| `observability/` | Replay, manifests, seeding |
| `kinematics/` | IK solvers attached via `policy_ik` |
| `perception/` | Vision helpers consumed by tasks/policies |
| `ros2/` | Shared ROS2 runtime (not backend-specific I/O) |
| `testing/` | Dummy backend for smoke tests only |

**Demo tasks and policies** belong under `examples/`, registered via `custom_modules` in preset YAML.

## Architecture rules

- Robot descriptions live under `robodeploy/description/*`. They should describe assets, joint names, limits, frames, and optional launch metadata. They should not open simulators, ROS nodes, or serial devices.
- Backends live under `robodeploy/backends/*` and own simulator or hardware integration.
- Sensors implement the shared sensor interface and can be paired across sim and real implementations through the registry.
- Policies return `Action` objects. If a policy emits a non-joint action, set `Action.action_space` or the task's policy `action_space` explicitly.
- Multi-robot backend methods should be implemented intentionally. Do not add fallback shims that silently call single-robot methods unless the backend really supports that behavior.

## Code style

- Prefer local patterns over new abstractions.
- Use type hints on new public functions and on internal helpers where the types clarify behavior.
- Use readable names. Unit suffixes such as `_s`, `_ms`, `_hz`, and `_rad` are encouraged where ambiguity is likely, but the repo does not enforce a blanket naming rule.
- Large files are acceptable when they model one cohesive adapter. Split files when it makes behavior easier to test or review.
- Keep assets in asset files. Avoid embedding long XML/URDF/MJCF strings in Python unless a test fixture needs a tiny inline model.

## Arrays and interop

The codebase uses NumPy, JAX arrays, and optional PyTorch depending on the boundary. Convert at boundaries and keep the conversion visible. `robodeploy/core/interop.py` currently copies JAX arrays through NumPy; it does not implement a DLPack zero-copy path.

## Adding components

- Register importable components with the relevant `register_*` decorator when they are meant for `RoboEnv.from_config()` (canonical) or minimal `RoboEnv.make()` smoke paths.
- Example/demo YAML presets belong under `examples/config/`, not `robodeploy/`.
- Registered placeholder policies and controllers should use names that make their stub status clear.
- If a component needs optional dependencies such as `rclpy`, `mujoco`, `torch`, or Isaac Sim, import them lazily or raise an actionable `ImportError`.

Entry points in `pyproject.toml`: `robodeploy.backends`, `robodeploy.robots`, `robodeploy.tasks`, `robodeploy.policies`, `robodeploy.sensors`.

## Tests

Run the focused tests that cover your change:

```bash
python -m compileall robodeploy tests examples
python -m pytest tests/ -q -m "not hardware"    # full suite (~620 tests)
python -m pytest tests/path/to/test_foo.py -q   # narrow
python -m pytest tests/test_cli.py tests/test_presets.py -q   # quick CLI/preset smoke
git diff --check
```

**Markers** (`pyproject.toml` → `[tool.pytest.ini_options]`):

| Marker | Meaning |
|--------|---------|
| `hardware` | Real robot / lab — skipped by default |
| `slow` | PPO convergence — minutes |
| `live_gazebo` | Linux + Gazebo Harmonic (`ROBODEPLOY_LIVE_GAZEBO=1`) |
| `optional_nightly` | 50k PPO proxy — mirrors `ppo-nightly.yml` |

Hardware tests must skip unless the required environment variables and devices are present ([tests/HARDWARE_TESTS.md](tests/HARDWARE_TESTS.md)). Avoid sleep-only assertions in threaded tests; prefer `threading.Event` or another explicit synchronization point.

**CI:** see [plans/INTEGRATION_STATUS.md](plans/INTEGRATION_STATUS.md) for job → claim mapping.

## Documentation

User-facing docs live under `docs/` and are built with MkDocs:

```bash
pip install -e ".[docs]"
mkdocs serve
mkdocs build          # omit --strict if external GitHub links warn
```

When adding features, update:

- [README.md](README.md) for install/quickstart changes
- [docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md) for architectural or workflow changes
- [docs/PLATFORM_STATUS.md](docs/PLATFORM_STATUS.md) if CI coverage or maturity changes
- [ARCHITECTURE.md](ARCHITECTURE.md) / [CONTRACTS.md](CONTRACTS.md) if public API behavior changes
- Relevant guide (`docs/SENSOR_INTEGRATION.md`, `docs/TRAINING.md`, etc.)
- [plans/INTEGRATION_STATUS.md](plans/INTEGRATION_STATUS.md) if presets or CI jobs change
- [examples/README.md](examples/README.md) if presets or runnable demos change

Do not commit `MUJOCO_LOG.TXT` or ad-hoc benchmark submission artifacts under `benchmarks/leaderboard/submissions/` unless intentionally contributing scores.
