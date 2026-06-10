# Contributing

RoboDeploy is in active development (v0.2 beta). Keep changes small enough to review and preserve the main boundary: user code builds a `Robot` and `RoboEnv`; backends adapt that contract to a simulator or real hardware.

**Read first:**

| Doc | Purpose |
|-----|---------|
| [docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md) | Platform overview |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Layer diagram and principles |
| [CONTRACTS.md](CONTRACTS.md) | Public API contracts |
| [plans/INTEGRATION_STATUS.md](plans/INTEGRATION_STATUS.md) | What CI proves vs claims |
| [history.json](history.json) | Machine-readable gap list |

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

## Tests

Run the focused tests that cover your change:

```bash
python -m compileall robodeploy tests examples
python -m pytest tests/ -q -m "not hardware"    # full suite (~620 tests)
python -m pytest tests/path/to/test_foo.py -q   # narrow
git diff --check
```

**Markers:**

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
mkdocs build --strict
```

When adding features, update:

- [README.md](README.md) for install/quickstart changes
- [docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md) for architectural or workflow changes
- [docs/PLATFORM_STATUS.md](docs/PLATFORM_STATUS.md) if CI coverage or maturity changes
- Relevant guide (`docs/SENSOR_INTEGRATION.md`, `docs/TRAINING.md`, etc.)
- [plans/INTEGRATION_STATUS.md](plans/INTEGRATION_STATUS.md) if presets or CI jobs change