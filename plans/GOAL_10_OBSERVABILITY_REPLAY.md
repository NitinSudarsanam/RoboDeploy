# Goal 10 — Observability, Replay, Determinism

**Priority**: Tier 3. **Effort**: ~30h. **Touches**: research workflows, debugging.

## Problem

- `sensor_status` collected (`obs_pipeline.py:77`) but no consumer surfaces it.
- No wandb / tensorboard / mlflow sinks.
- No trajectory checkpoint or rollback. Only forward replay.
- Seeding partial: domain randomizer seeded but policies + env-level RNG not consistently seeded. No `env.reset(seed=)`.
- Determinism untested.
- No `get_diagnostics()` consumer.
- ROS2 controllers implement diagnostics protocol but values not logged.

## Current State (Audit)

### Logging
- Zero references to wandb / tensorboard / mlflow.
- Only `--quiet` flag (`cli.py:142`) toggles request logs.
- Basic `print()` / JSON via `cli_helpers.py:23-33`.

### Diagnostics
- `SupportsDiagnostics` protocol in `backends/capabilities.py:20-23` with `get_diagnostics()`.
- ROS2 backend populates `_diagnostics` dict (`backends/real/ros2/backend.py:145`).
- `obs_pipeline.py:68-77` merges per-sensor status into `Observation.sensor_status`.
- `EpisodeInfo.extra["diagnostics"]` populated via `build_diagnostics_extra()` (`core/extra_schemas.py:25-26`).
- Test: `tests/test_env_diagnostics.py` validates failed-import diagnostics surface.

### Replay
- `demo_recording.py:34-59` — DemoRecorder.
- `demo_recording.py:86-88` — `iter_replay_actions()` yields actions.
- HDF5 export referenced (`cli.py:72-75`) but not all paths implemented.
- No state snapshot. No rollback. No trajectory diff.

### Seeding
- `tasks/randomization.py` seeds noise generators.
- `backends/sim/isaacsim/backend.py` references seed.
- No `RoboEnv.reset(seed=...)`. No backend.seed(). No policy.seed().

---

## Deliverables

### D1. Structured Logging — `robodeploy/observability/logger.py` (NEW, ~250 lines)

```python
class RoboDeployLogger:
    """Single entry point. Multi-sink dispatch."""

    def __init__(self, *, sinks: list["LogSink"] | None = None,
                 run_name: str | None = None, config: dict | None = None):
        self._sinks = sinks or []
        self._step = 0
        self._run_name = run_name or _generate_run_name()
        self._meta = {"run_name": self._run_name, "start_time": time.time(), "config": config or {}}

    def log_step(self, payload: dict, *, step: int | None = None):
        s = step if step is not None else self._step
        for sink in self._sinks: sink.write(s, payload, kind="step")
        self._step = s + 1

    def log_episode(self, payload: dict): ...
    def log_diagnostic(self, payload: dict, *, level: Literal["info","warn","error"] = "info"): ...
    def log_artifact(self, name: str, path: Path): ...
    def close(self):
        for sink in self._sinks: sink.close()

class LogSink(Protocol):
    def write(self, step: int, payload: dict, *, kind: str): ...
    def close(self): ...

class WandbSink(LogSink): ...
class TensorBoardSink(LogSink): ...
class JsonlSink(LogSink): ...     # local file
class MlflowSink(LogSink): ...    # optional
class StdoutSink(LogSink): ...    # human-readable
```

### D2. Diagnostic Streaming — `robodeploy/env.py` (EXTEND)

```python
class RoboEnv:
    def __init__(self, ..., logger: RoboDeployLogger | None = None): ...

    def step(self, action):
        obs, reward, done, info = ...
        info.extra["sensor_status"] = obs.sensor_status or {}
        info.extra["policy_diagnostics"] = self._policy_diag.summary() if self._policy_diag else {}
        info.extra["backend_diagnostics"] = self._backend.get_diagnostics() if isinstance(self._backend, SupportsDiagnostics) else {}
        info.extra["sensor_health"] = _summarize_health(obs.sensor_status)
        if self._logger:
            self._logger.log_step({
                "reward": reward, "done": done,
                "reward_components": info.extra.get("reward_components", {}),
                "sensor_health": info.extra["sensor_health"],
                "diagnostics": info.extra["backend_diagnostics"],
            })
        return obs, reward, done, info
```

### D3. Episode Checkpoint / State Snapshot — `robodeploy/observability/snapshot.py` (NEW, ~300 lines)

```python
@dataclass
class StateSnapshot:
    timestamp: float
    step: int
    episode_id: str
    sim_state: dict | None      # backend-specific (MuJoCo qpos/qvel/qfrc, Isaac articulation state, etc.)
    obs: Observation
    last_action: Action | None
    policy_state: dict | None   # phase, internal counters
    rng_state: dict             # numpy + python random + torch RNGs

class SnapshotManager:
    """Save/load environment state for rollback + diff."""

    def __init__(self, *, env: RoboEnv, max_keep: int = 100):
        self._env = env
        self._snapshots: list[StateSnapshot] = []

    def capture(self) -> StateSnapshot: ...
    def restore(self, snapshot: StateSnapshot) -> None: ...
    def save(self, path: Path): pickle.dump(self._snapshots, path.open("wb"))
    def load(self, path: Path): ...

    def rollback(self, n_steps: int = 1) -> None:
        if len(self._snapshots) < n_steps: raise ValueError(...)
        snap = self._snapshots[-n_steps]
        self.restore(snap)
        self._snapshots = self._snapshots[:-n_steps]
```

Backend extension required: each `IBackend` adds `get_sim_state() -> dict` + `set_sim_state(state)`. MuJoCo: serialize qpos/qvel/act. IsaacSim: ArticulationView state. Gazebo: not feasible (server-owned) — document as MuJoCo-only.

### D4. Deterministic Seeding — `robodeploy/env.py`, `core/seeding.py` (NEW)

```python
@dataclass
class SeedSet:
    env_seed: int
    policy_seed: int
    randomizer_seed: int
    sensor_noise_seed: int
    obs_pipeline_seed: int

def derive_seeds(master_seed: int) -> SeedSet:
    """Stable derivation: each child seeded by hash(master_seed, child_name)."""
    rng = np.random.default_rng(master_seed)
    return SeedSet(
        env_seed=int(rng.integers(0, 2**31)),
        policy_seed=int(rng.integers(0, 2**31)),
        randomizer_seed=int(rng.integers(0, 2**31)),
        sensor_noise_seed=int(rng.integers(0, 2**31)),
        obs_pipeline_seed=int(rng.integers(0, 2**31)),
    )

class RoboEnv:
    def reset(self, *, seed: int | None = None):
        if seed is not None:
            self._seeds = derive_seeds(seed)
            self._seed_components()
        obs, info = ...

    def _seed_components(self):
        np.random.seed(self._seeds.env_seed)
        random.seed(self._seeds.env_seed)
        try: import torch; torch.manual_seed(self._seeds.env_seed)
        except ImportError: pass
        self._policy.reset(seed=self._seeds.policy_seed)
        self._task.randomizer.seed(self._seeds.randomizer_seed)
        for sensor in self._sensors: sensor.seed(self._seeds.sensor_noise_seed)
        self._backend.seed(self._seeds.env_seed)
```

Extend `IPolicy.reset(seed=None)`, `ISensor.seed(seed)`, `IBackend.seed(seed)`.

### D5. Determinism Tests — `tests/test_determinism.py` (NEW)

```python
def test_two_seeded_rollouts_identical():
    env1 = make_env(seed=42)
    env2 = make_env(seed=42)
    traj1 = rollout(env1, n_steps=200)
    traj2 = rollout(env2, n_steps=200)
    np.testing.assert_array_equal(traj1.obs, traj2.obs)
    np.testing.assert_array_equal(traj1.actions, traj2.actions)
    assert traj1.rewards == traj2.rewards
```

Run for each backend.

### D6. Trajectory Replay + Diff — `robodeploy/observability/replay.py` (NEW, ~250 lines)

```python
class TrajectoryReplayer:
    """Replay a recorded trajectory. Optional: divergence detection."""

    def __init__(self, *, env: RoboEnv, recording: DemoRecorder | Path,
                 divergence_threshold: dict | None = None,
                 on_divergence: Literal["warn","halt","record"] = "warn"):
        self._env = env
        self._frames = recording if isinstance(recording, list) else DemoRecorder.load(recording).frames
        self._threshold = divergence_threshold or {"joint_pos": 0.01, "ee_pos": 0.005}

    def play(self) -> ReplayReport:
        obs, info = self._env.reset(seed=self._frames[0].metadata.get("seed"))
        report = ReplayReport()
        for i, frame in enumerate(self._frames):
            action = frame.action
            obs, reward, done, info = self._env.step(action)
            div = self._compute_divergence(obs, frame.observation)
            report.add(i, div, obs, frame.observation)
            if max(div.values()) > max(self._threshold.values()):
                if self._on_divergence == "halt": break
                elif self._on_divergence == "warn": warnings.warn(...)
        return report

@dataclass
class ReplayReport:
    divergences: list[dict]     # per-step divergence metrics
    max_divergence: dict
    diverged_steps: list[int]
    def render_plot(self, out: Path): ...
```

CLI:
```bash
robodeploy replay demo.jsonl --preset kuka_pick_mujoco --diff --output report.json
```

### D7. Run Manifest — `robodeploy/observability/manifest.py` (NEW, ~150 lines)

Per-run metadata bundle for reproducibility.

```python
@dataclass
class RunManifest:
    run_name: str
    started_at: float
    finished_at: float | None
    seed: int
    env_config: dict
    backend: str
    backend_version: str | None
    robot: str
    task: str
    policy: str
    policy_checkpoint: str | None
    git_hash: str | None
    git_dirty: bool
    python_version: str
    package_version: str
    asset_manifest_hash: str | None     # from Goal 8 D15
    sensor_rig: list[str]

    def save(self, path: Path): ...
    @classmethod
    def load(cls, path: Path) -> "RunManifest": ...

class ManifestRecorder:
    def __init__(self, env: RoboEnv): ...
    def write(self, out_dir: Path): ...
```

Saved alongside every recorded demo + checkpoint. Replay re-uses to recreate exact env.

### D8. Reward Component Logging — extend `RewardBuilder` (Goal 1 D9)

`RewardBuilder.build_components()` returns per-term breakdown. Surface via `info.extra["reward_components"]`. RoboDeployLogger writes per-term scalars.

### D9. Hot-Reload Dashboard (stretch) — `robodeploy/observability/dashboard.py` (NEW, ~400 lines)

FastAPI + WebSocket live dashboard. Charts: reward, success rate, sensor health, action stats. Reads from JSONL sink or wandb run.

```bash
robodeploy dashboard --logs runs/2026-06-08-1410/
```

**Status: DEFERRED** — see `robodeploy/observability/DASHBOARD_DEFERRAL.md`. Use JSONL + `robodeploy logs tail/summary` and optional W&B/TensorBoard/MLflow sinks instead.

### D10. Health Check Integration — `robodeploy/observability/health.py` (NEW)

```python
class HealthMonitor:
    """Watch sensor_status, backend diagnostics; raise on degraded modes."""

    def __init__(self, *, fail_threshold_per_sensor: int = 5,
                 on_failure: Callable[[str, dict], None] | None = None):
        self._fail_counts: dict[str, int] = {}

    def observe(self, sensor_status: dict[str, str]) -> Literal["ok","degraded","failed"]:
        for name, status in sensor_status.items():
            if status != "ok":
                self._fail_counts[name] = self._fail_counts.get(name, 0) + 1
                if self._fail_counts[name] > self._fail_threshold:
                    if self._on_failure: self._on_failure(name, sensor_status)
                    return "failed"
            else:
                self._fail_counts[name] = 0
        if any(s != "ok" for s in sensor_status.values()): return "degraded"
        return "ok"
```

Integrated into `env.step()` (D2).

### D11. CLI — `robodeploy/cli.py` (EXTEND)

```bash
robodeploy logs tail runs/<name>            # follow JSONL in real time
robodeploy logs summary runs/<name>          # statistics
robodeploy snapshot save snapshot.pkl --preset X
robodeploy snapshot restore snapshot.pkl
robodeploy replay demo.jsonl --preset X --diff
robodeploy dashboard --logs runs/<name>
robodeploy manifest show runs/<name>/manifest.json
```

### D12. Tests
- `tests/test_logger_sinks.py` — wandb (mocked), tb, jsonl.
- `tests/test_snapshot.py` — capture / restore / rollback for MuJoCo.
- `tests/test_determinism.py` — see D5.
- `tests/test_replay_diff.py` — divergence detection.
- `tests/test_manifest.py` — round-trip + git state.
- `tests/test_health_monitor.py` — fail count + callback.

---

## Phased Rollout

### Phase 10.1 — Determinism (~6h)
- D4 SeedSet + reset(seed=) + per-component seeding.
- D5 determinism tests for MuJoCo + IsaacSim.
- Update `IPolicy`, `ISensor`, `IBackend` with `.seed()` methods.

### Phase 10.2 — Logging + diagnostics surfacing (~8h)
- D1 RoboDeployLogger + sinks (StdoutSink, JsonlSink, WandbSink, TensorBoardSink).
- D2 env.step diagnostic streaming.
- D8 reward component logging.
- D10 HealthMonitor.
- `tests/test_logger_sinks.py`, `tests/test_health_monitor.py`.

### Phase 10.3 — Snapshot + replay (~8h)
- D3 SnapshotManager + backend get/set_sim_state (MuJoCo + IsaacSim).
- D6 TrajectoryReplayer + divergence report.
- D11 CLI subcommands.
- `tests/test_snapshot.py`, `tests/test_replay_diff.py`.

### Phase 10.4 — Run manifest (~4h)
- D7 RunManifest + ManifestRecorder.
- Auto-write on `RoboEnv.close()`.
- `tests/test_manifest.py`.

### Phase 10.5 (stretch) — Dashboard (~4h)
- D9 FastAPI/WebSocket dashboard.

---

## Acceptance Criteria

- [x] `env.reset(seed=42)` produces identical trajectory across two runs (MuJoCo + Dummy; see `tests/test_determinism.py`).
- [x] `RoboDeployLogger(sinks=[WandbSink, JsonlSink])` writes to both (`tests/test_logger_sinks.py`).
- [x] `info.extra["sensor_status"]` populated every step (`tests/test_observability_env.py`).
- [x] `info.extra["backend_diagnostics"]` populated when backend implements `SupportsDiagnostics`.
- [x] `SnapshotManager.capture()` + `restore()` round-trip state (`tests/test_snapshot.py`; MuJoCo via backend `get/set_sim_state`).
- [x] `TrajectoryReplayer.play()` reports `max_divergence < 1e-6` on noiseless replay (`tests/test_replay_diff.py`).
- [x] `RunManifest.save()` writes git hash + dirty flag + python/package versions (`tests/test_manifest.py`).
- [x] `HealthMonitor` triggers callback after N consecutive failures on a sensor (`tests/test_health_monitor.py`).
- [x] `robodeploy logs tail` updates as JSONL grows (`robodeploy/cli_observability.py`).
- [x] `robodeploy replay demo.jsonl --diff` produces JSON report (`tests/test_replay_cli.py`).
- [x] `robodeploy manifest show runs/X/manifest.json` displays human-readable run.
- [x] Trajectory checkpoint format for Goal 11 (`robodeploy/observability/trajectory_checkpoint.py`, `tests/test_trajectory_checkpoint.py`).
- [x] EvalReport uses Goal 10 `RunManifest` — no duplicate type (`tests/test_eval_runmanifest.py`).
- [ ] D9 Dashboard — **deferred** (`robodeploy/observability/DASHBOARD_DEFERRAL.md`).

## Dependencies

- `wandb>=0.16` (optional sink).
- `tensorboard>=2.15` (optional sink).
- `mlflow>=2.9` (optional sink).
- `gitpython` (manifest git hash).
- `fastapi`, `uvicorn`, `websockets` (dashboard).

Add `[project.optional-dependencies] obs = [wandb, tensorboard, mlflow, gitpython]`.

## Risks

- **Snapshot incompleteness**: MuJoCo `mj_step` advances state but some side-effects (RNG buffers, OpenGL state) not captured. Mitigation: document limits + audit known side-effects.
- **Seeded MuJoCo nondeterminism on GPU**: MuJoCo GPU path not deterministic. Mitigation: gate determinism guarantees to CPU MuJoCo + JAX-deterministic mode.
- **Wandb requires network**: CI without internet → tests fail. Mitigation: mock sink; gate live wandb tests on env var.
- **Replay divergence due to backend nondeterminism**: noisy physics step. Mitigation: replay uses snapshot restore instead of pure forward simulation when divergence > threshold.

## Out of Scope

- Distributed tracing (OpenTelemetry, Jaeger). Future.
- Online anomaly detection (outlier policies). Future.
- Continuous profiling. External tools (py-spy, scalene).
