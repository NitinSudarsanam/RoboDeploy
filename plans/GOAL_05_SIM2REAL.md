# Goal 5 — Sim2Real Pipeline

**Priority**: Tier 2. **Effort**: ~60h. **Touches**: real-deploy claim.

## Problem

Architectural foundation solid (`RoboBridge`, multi-backend, shared contracts). Operational pipeline incomplete:
- DR exists but no sweep automation. Manual hyperparameter search.
- Calibration = SO-101-specific (kinematic only). No generic kinematic/extrinsic/system-ID.
- No transfer-validation metrics. Can't measure sim-vs-real gap.
- Gaussian-only sensor noise. No dropouts, latency, colored noise, bias drift.
- No action noise / disturbances.
- "Train sim → deploy real" workflow undocumented.
- No reality-gap benchmark.

## Current State (Audit)

### Domain Randomization
- `robodeploy/tasks/randomization.py:66-78` — 3-level enum (NONE/LIGHT/FULL).
- Randomizes: object poses (uniform pos/yaw ranges), physics (gravity noise, friction scale, mass scale), sensor noise (Gaussian per modality).
- `DomainRandomizerConfig` + `ObjectRandomConfig` + `PhysicsRandomConfig` + `SensorNoiseConfig`.
- Real backends raise `NotImplementedError` on physics randomization.
- No sweep over noise levels / parameter distributions.

### Calibration (SO-101 only)
- `robodeploy/description/so101/calibration.py:1-273` — linear motor-tick ↔ radian fit.
- `examples/so101/calibrate_so101.py` — two-pose manual capture.
- Storage: JSON at `~/.robodeploy/so101_calibration.json` or env `ROBODEPLOY_SO101_CALIBRATION`.
- `robodeploy/backends/real/ros2/sensors/tf_extrinsics.py` — live TF lookup for camera extrinsics (no automated calibration).
- **No generic IKinematicCalibration interface.**
- **No hand-eye / checkerboard / ArUco extrinsic calibration tool.**
- **No payload mass / TCP offset identification.**

### Real-Time Bridge
- `robodeploy/bridge.py:1-275` — `RoboBridge` decouples control (fixed-Hz subprocess) from inference (variable Hz).
- `ActionTrajectory` seqlock shared mem; `EStopFlag` multiprocessing event.
- No latency model. No command buffering for variable inference rates.

### Transfer Validation
- **None.** `tests/test_so101_real.py` tests calibration round-trip + safety. No sim-vs-real policy comparison.

---

## Deliverables

### D1. Generic Calibration Framework — `robodeploy/calibration/` (NEW directory)

```
robodeploy/calibration/
├── __init__.py
├── base.py              # IKinematicCalibration, IExtrinsicCalibration
├── kinematic/
│   ├── linear.py        # Multi-pose linear fit (generalizes SO-101)
│   ├── nonlinear.py     # Nonlinear DH-param fit
│   └── motor_bus.py     # Motor-encoder ↔ radians
├── extrinsic/
│   ├── checkerboard.py  # OpenCV checkerboard detection
│   ├── aruco.py         # ArUco marker hand-eye
│   ├── handeye.py       # Tsai-Lenz / Park-Martin / Daniilidis
│   └── tf_lookup.py     # Existing ROS TF (refactored)
├── system_id/
│   ├── friction.py      # Coulomb + viscous friction estimation
│   ├── mass.py          # Payload mass from gravity-loaded torque
│   ├── dh.py            # Denavit-Hartenberg parameter fit
│   └── pipeline.py      # Orchestrator
├── store.py             # CalibrationStore — JSON + env var resolution
└── cli.py               # CLI subcommands
```

```python
class IKinematicCalibration(Protocol):
    def fit(self, raw_to_canonical_pairs: list[tuple[Any, Any]]) -> "IKinematicCalibration": ...
    def to_canonical(self, raw_value): ...
    def to_raw(self, canonical_value): ...
    def save(self, path: str | Path) -> None: ...
    @classmethod
    def load(cls, path: str | Path) -> "IKinematicCalibration": ...

class IExtrinsicCalibration(Protocol):
    def fit(self, observations: list["Pose3D"]) -> Pose3D: ...
    def fit_handeye(self, robot_poses: list[Pose3D], marker_poses: list[Pose3D]) -> Pose3D: ...
```

Refactor SO-101 calibration to inherit from `LinearKinematicCalibration` (or wrap it). No behavior change for existing SO-101 workflow.

### D2. CalibrationStore — `robodeploy/calibration/store.py`

Unified storage for all calibration artifacts:

```python
class CalibrationStore:
    def __init__(self, root: Path | None = None):
        self._root = root or Path.home() / ".robodeploy" / "calibration"

    def save(self, name: str, payload: dict, *, robot_id: str | None = None) -> Path:
        path = self._root / (robot_id or "default") / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2))
        return path

    def load(self, name: str, *, robot_id: str | None = None) -> dict: ...
    def list_all(self) -> list[dict]: ...  # [{name, robot_id, schema_version, modified}]
```

Migration: `SO101Calibration.locate()` delegates to `CalibrationStore`.

### D3. Extrinsic Calibration Tool — `robodeploy/calibration/extrinsic/checkerboard.py`

```python
class CheckerboardExtrinsicCalibrator:
    """Detect checkerboard corners in N frames; solve PnP for camera-in-world pose."""
    def __init__(self, *, board_size=(7,5), square_size_m=0.025): ...
    def capture(self, env: RoboEnv, n_frames: int = 20, *, prompt_each_pose: bool = True) -> list[CheckerboardSample]: ...
    def fit(self, samples: list[CheckerboardSample], intrinsics: CameraIntrinsics) -> Pose3D: ...
    def save(self, store: CalibrationStore, *, name: str, robot_id: str): ...
```

ArUco variant:

```python
class ArUcoExtrinsicCalibrator:
    def __init__(self, *, dictionary="DICT_4X4_50", marker_size_m=0.05): ...
    def fit(self, frames: list[np.ndarray], intrinsics: CameraIntrinsics) -> dict[int, Pose3D]: ...
```

Hand-eye:

```python
class HandEyeCalibrator:
    """Tsai-Lenz hand-eye calibration."""
    def fit(self, robot_poses: list[Pose3D], marker_poses: list[Pose3D],
            method: Literal["tsai","park","daniilidis"] = "park") -> Pose3D:
        return cv2.calibrateHandEye(...)
```

### D4. System Identification — `robodeploy/calibration/system_id/`

**friction.py**: drive each joint at constant velocity, measure steady-state torque → fit Coulomb (static) + viscous (proportional) terms.

```python
class FrictionEstimator:
    def collect_data(self, env, joint_idx: int, *, velocities: list[float], steady_state_steps: int = 100): ...
    def fit(self, samples) -> FrictionParams: ...
```

**mass.py**: hold joint at static pose under gravity; torque = m*g*r_cm. Solve for m given known CoM.

```python
class PayloadMassEstimator:
    def estimate(self, env, *, joint_idx: int = -2, pose_test_q: np.ndarray) -> float: ...
```

**pipeline.py**: orchestrator that runs friction + mass + DH back-to-back and writes to `CalibrationStore`.

### D5. DR Sweep Framework — `robodeploy/training/dr_sweep.py` (NEW, ~400 lines)

Parameter sweep over DR levels + ranges. Evaluates policy under each sample, returns sensitivity report.

```python
@dataclass
class DRSweepConfig:
    n_seeds: int = 5
    n_episodes_per_seed: int = 20
    levels: list[RandomLevel] = field(default_factory=lambda: [RandomLevel.NONE, RandomLevel.LIGHT, RandomLevel.FULL])
    object_position_ranges: list[tuple[float, float]] = field(default_factory=lambda: [(0.0, 0.0), (0.02, 0.02), (0.05, 0.05)])
    physics_friction_ranges: list[tuple[float, float]] = field(default_factory=lambda: [(1.0, 1.0), (0.7, 1.3), (0.5, 1.5)])
    sensor_noise_scales: list[float] = field(default_factory=lambda: [0.0, 0.5, 1.0, 2.0])

class DRSweep:
    def __init__(self, *, env_fn, policy_fn, config: DRSweepConfig): ...
    def run(self) -> DRSweepReport: ...   # parallel via SubprocVecEnv
    def report(self) -> dict: ...

@dataclass
class DRSweepReport:
    cells: list[dict]   # [{params, success_rate, mean_reward, std}]
    sensitivity: dict   # {param_name: spearman_corr(success_rate, value)}
    robust_params: dict  # max-success cell

    def plot_heatmap(self, x_param, y_param, *, metric="success_rate", out_path: str): ...
```

CLI:
```bash
robodeploy dr-sweep --preset kuka_pick_mujoco --policy checkpoint.pt --output reports/dr_sweep_001/
```

### D6. Realistic Noise Models — extend `robodeploy/core/transforms.py`

```python
class ColoredNoiseTransform(ObsTransform):
    """Brownian / 1-f / OU process noise."""
    def __init__(self, *, kind: Literal["gaussian","ou","brownian","one_over_f"], sigma: float, dt: float, tau: float = 1.0): ...

class DropoutTransform(ObsTransform):
    """Drop frames at probability p; previous frame held for stale_count steps."""
    def __init__(self, *, p: float = 0.01, max_stale_steps: int = 5): ...

class LatencyTransform(ObsTransform):
    """Delay obs by N steps (latency injection)."""
    def __init__(self, *, latency_steps: int = 1, jitter_steps: int = 0): ...

class QuantizationTransform(ObsTransform):
    """Encoder quantization (round to nearest tick)."""
    def __init__(self, *, ticks_per_unit: dict[str, float]): ...

class BiasDriftTransform(ObsTransform):
    """Slowly drifting bias (e.g., IMU gyro drift)."""
    def __init__(self, *, drift_rate: float, max_drift: float): ...
```

### D7. Action Noise + Disturbances — `robodeploy/tasks/action_noise.py` (NEW, ~150 lines)

```python
class ActionNoiseInjector:
    """Inject noise into actions during sim training."""
    def __init__(self, *, joint_noise_std: float = 0.001, command_dropout_p: float = 0.0,
                 slip_probability: float = 0.0): ...
    def __call__(self, action: Action) -> Action: ...

class ExternalDisturbanceInjector:
    """Random external forces on gripper / prop during sim."""
    def __init__(self, *, force_range_N: tuple[float, float] = (0.0, 1.0),
                 duration_steps_range: tuple[int, int] = (1, 5),
                 probability_per_step: float = 0.001): ...
    def inject(self, backend) -> None: ...
```

Integrate into `TaskBase.reset_fn` + `step` hook.

### D8. Transfer Validation Metrics — `robodeploy/evaluation/transfer_metrics.py` (NEW, ~300 lines)

Record rollouts on both sim + real with same policy + seed. Compute distance metrics.

```python
@dataclass
class TransferMetrics:
    sim_success_rate: float
    real_success_rate: float
    success_gap: float                      # sim - real

    trajectory_distance: dict[str, float]   # {"joint_pos_l2": ..., "ee_pos_l2": ..., "ee_quat_geodesic": ...}
    obs_distribution_kl: dict[str, float]   # KL between sim/real obs marginals per modality
    action_distribution_l2: float

    per_episode_breakdown: list[dict]

class TransferEvaluator:
    def __init__(self, *, sim_env_fn, real_env_fn, policy_fn, n_episodes: int = 20): ...
    def run(self) -> TransferMetrics: ...
    def render_report(self, out_dir: str): ...  # writes JSON + plots
```

CLI:
```bash
robodeploy transfer-eval --preset-sim kuka_pick_mujoco --preset-real kuka_pick_real \
    --policy checkpoint.pt --episodes 20 --output transfer_report/
```

### D9. Latency Compensation in RoboBridge — extend `robodeploy/bridge.py`

```python
class LatencyModel:
    """Models comm + execution delay. Used by control process for action interpolation."""
    def __init__(self, *, mean_delay_s: float, jitter_std_s: float, max_buffer: int = 32): ...
    def predict_execution_time(self, command_time: float) -> float: ...
    def interpolate_command(self, buffer: list[Action], now: float) -> Action: ...

# In RoboBridge:
self._latency_model = LatencyModel(mean_delay_s=0.02, jitter_std_s=0.005)
# control process interpolates between buffered actions to absorb jitter
```

### D10. Sim2Real Documentation — `docs/SIM2REAL.md` (NEW)

End-to-end workflow:
1. Train sim policy with DR sweep → pick robust params.
2. Calibrate real robot (kinematic + extrinsic + payload).
3. Deploy via `RoboBridge` with `LatencyModel`.
4. Run `TransferEvaluator` → produce report.
5. Iterate: tune DR params based on transfer gap.

Include checklist + example command sequence + troubleshooting (e.g., "policy works in sim but trembles on real → increase IMU noise / lower action_hz").

### D11. Reality-Gap Benchmark — `benchmarks/sim2real/` (NEW)

Standardized tasks with sim + real reference policies + expected transfer rates. Allows tracking sim2real performance over releases.

- `benchmarks/sim2real/reach_to_target/` — minimal reach task.
- `benchmarks/sim2real/pick_place_cube/` — standard pick-place.
- `benchmarks/sim2real/peg_insert/` — contact-rich.

Each provides: sim preset, real preset, calibration template, reference checkpoint, expected sim/real success rates.

---

## Phased Rollout

### Phase 5.1 — Calibration framework (~15h)
- D1 directory + interfaces.
- D2 CalibrationStore.
- Refactor SO-101 calibration into linear kinematic plugin (no behavior change).
- D3 CheckerboardExtrinsicCalibrator (single-camera).
- `tests/test_calibration_store.py`, `tests/test_checkerboard_extrinsic.py`.

### Phase 5.2 — System ID (~12h)
- D4 friction + payload mass + DH-param estimation.
- D11 reach_to_target sim2real benchmark + reference data.
- `tests/test_system_id.py`.

### Phase 5.3 — DR sweep + Noise models (~15h)
- D5 DRSweep framework + CLI subcommand.
- D6 ColoredNoise + Dropout + Latency + Quantization + BiasDrift transforms.
- D7 ActionNoiseInjector + ExternalDisturbanceInjector.
- `tests/test_dr_sweep.py`, `tests/test_realistic_noise.py`.

### Phase 5.4 — Transfer validation (~12h)
- D8 TransferEvaluator + metrics.
- D9 LatencyModel in RoboBridge.
- `tests/test_transfer_metrics.py` (sim-vs-sim with injected noise as proxy).

### Phase 5.5 — Docs + Benchmarks (~6h)
- D10 SIM2REAL.md walkthrough.
- D11 pick_place_cube + peg_insert benchmarks.
- Reference checkpoints (small BC nets trained for benchmark).

---

## Acceptance Criteria

- [x] `robodeploy calibrate kinematic --robot so101 --port /dev/ttyACM0` works (dry-run CLI; `tests/test_calibration_live.py`).
- [x] `robodeploy calibrate extrinsic --camera wrist --pattern checkerboard --board 7x5x0.025` solves PnP from N frames (mocked detection; `tests/test_calibration_live.py`).
- [x] `robodeploy calibrate handeye --robot franka --pattern aruco` outputs `T_camera_to_ee` (dry-run CLI; `tests/test_calibration_live.py`).
- [x] `robodeploy calibrate system-id --robot franka --joint 4` outputs friction + mass (dummy backend; `tests/test_calibration_live.py`).
- [x] `robodeploy dr-sweep --preset X` produces heatmap of success vs DR params (dummy env; `tests/test_dr_sweep.py::test_dr_sweep_produces_report`).
- [x] `LatencyTransform(latency_steps=2)` produces 2-step delayed obs; integration test passes (`tests/test_transfer_metrics.py::test_latency_transform_delays_by_n_steps`).
- [x] `TransferEvaluator` outputs JSON + plots comparing sim vs real rollouts (sim-vs-noisy-sim proxy; `tests/test_transfer_metrics.py`).
- [x] `reach_to_target` benchmark reproduces published expected success rates (sim target on dummy backend; `tests/test_sim2real_benchmarks.py`; real=80% not automated).
- [x] `docs/SIM2REAL.md` describes end-to-end workflow with example commands.
- [x] `CalibrationStore` round-trips kinematic + extrinsic + system-id data (`tests/test_calibration_store.py`).

## Dependencies

- `opencv-python>=4.8` — checkerboard, ArUco, hand-eye.
- `scipy` — least squares for sys-id.
- `matplotlib` — heatmap plots.
- (Optional) `colored-noise` package or implement OU process inline.

## Risks

- **Calibration data variance**: small datasets → overfit. Mitigation: report fit residuals + require minimum-pose-count threshold.
- **Hand-eye singularity**: insufficient rotation diversity → degenerate solve. Mitigation: enforce min-rotation between samples + report condition number.
- **System ID requires safe motion**: friction sweep moves joint past limits. Mitigation: hard joint-range guard + low velocity mode.
- **Transfer eval needs real hardware**: most users have only sim. Mitigation: sim-vs-sim-with-noise as proxy for CI tests.
- **DR over-randomization**: too much noise → no policy learns. Mitigation: report sensitivity early in sweep, suggest cutoff.

## Out of Scope

- Learning-based sim-to-real (CycleGAN, GAIL adaptation). External research.
- Multi-camera extrinsic graph optimization (bundle adjustment). Use external tool (COLMAP / Kalibr).
- Time-synchronization across distributed sensors. Use ROS2 TF + chrony.
- Real-time SLAM. External.
