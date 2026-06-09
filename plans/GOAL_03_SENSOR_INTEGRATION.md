# Goal 3 — Sensor → Policy/Task Integration

**Priority**: Tier 1. **Effort**: ~40h. **Touches**: contact-aware control.

## Problem

Sensor data collected but unused. Policies + tasks rely on oracle queries:
- `examples/policies/reach_pick_place.py:220-226` uses `backend.prop_near_ee()` (physics query), not `obs.ft_force`.
- `robodeploy/tasks/base.py:56` uses `backend.has_prop_contact()`, not sensor data.
- No policy reads `obs.ft_force` / `obs.ft_torque` for grasp confidence.
- No task uses force threshold for contact-gated success/termination.
- No IMU stability check anywhere.
- No vision-based termination (segmentation, blob tracking, learned pose).
- `obs_pipeline.py:77` stores `sensor_status` dict — no consumer surfaces it.

Backend coupling = bad. Real-hw policies need sensor signals, not `backend.has_prop_contact()` (sim-only).

## Current State (Audit)

### Sensor wiring
- `robodeploy/obs_pipeline.py:43-77` — `SensorSampleBuffer` merges multi-sensor reads with timestamps.
- Fields populated: `obs.rgb`, `obs.depth`, `obs.ft_force` [3], `obs.ft_torque` [3], `obs.ft_forces` dict, `obs.imu_acceleration` [3], `obs.imu_angular_velocity` [3], `obs.objects` dict, `obs.camera_intrinsics`, `obs.camera_extrinsics`.
- Domain randomization noise injected at `tasks/randomization.py:66-78`.

### Policy consumption (current)
- `examples/policies/sensor_reach_pick.py:20-29` — reads `obs.objects` for prop_pose oracle. **Only sensor consumer.**
- `examples/policies/reach_pick_place.py` — reads `obs.objects` as fallback; primary = scene waypoints.

### Task consumption (current)
- `robodeploy/tasks/base.py` `object_pose()` — tries `obs.objects` first, falls back to `backend.get_prop_pose()`.
- All success/reward functions = oracle distance to backend-resolved pose.

### Missing
- No `obs.ft_force` consumers. Zero grep hits in policies/tasks (excluding pipeline + randomizer).
- No `obs.imu_acceleration` consumers.
- No vision-based predicate (blob detector wired but only for color-blob example).

---

## Deliverables

### D1. ContactSensor Wrapper — `robodeploy/sensors/contact/` (NEW directory)

Decouple grasp detection from backend physics. Sensor exposes binary touch state via `obs.contact_state`.

**Files**:
- `robodeploy/sensors/contact/__init__.py`
- `robodeploy/sensors/contact/base.py` — `ContactSensorBase(SensorBase)`.
- `robodeploy/sensors/contact/sim/mujoco_contact.py` — queries `backend.has_prop_contact()` + `prop_near_ee()`, exposes as `SensorData`.
- `robodeploy/sensors/contact/sim/gazebo_contact.py` — subscribes to `/contacts` topic (Gazebo contact plugin).
- `robodeploy/sensors/contact/real/ft_threshold.py` — wraps FT sensor, returns binary above N-Newton threshold.

```python
@register_sensor_pair("wrist_contact",
                      sim=MuJoCoContactSensor, real=FTThresholdContactSensor)
class MuJoCoContactSensor(ContactSensorBase):
    def __init__(self, prop_name: str, ee_distance_threshold: float = 0.04):
        ...
    def read(self) -> SensorData:
        contact = self._backend.has_prop_contact(self._prop_name)
        near = self._backend.prop_near_ee(self._prop_name, self._ee_dist)
        return SensorData(payload={"contact": bool(contact), "near_ee": bool(near)}, timestamp=time.time())
```

Add `Observation.contact_state: dict[str, bool] | None` to `core/types.py` + populate in `obs_pipeline.py`.

### D2. FT-Based Grasp State Machine — `examples/policies/ft_reach_pick.py` (NEW)

Replaces `backend.has_prop_contact()` queries with sensor-driven detection:

```python
@register_policy("ft_reach_pick")
class FTReachPickPolicy(ReachTrajectoryPolicy):
    """Reach + pick using FT sensor for grasp confirmation."""

    GRASP_FORCE_THRESHOLD_N = 2.0
    LIFT_FORCE_LOSS_THRESHOLD_N = 0.5  # drop detection
    FORCE_WINDOW_STEPS = 5

    def get_action(self, obs: Observation) -> Action:
        if self._current_phase == "close_gripper":
            force_norm = np.linalg.norm(obs.ft_force) if obs.ft_force is not None else 0.0
            self._force_history.append(force_norm)
            if len(self._force_history) >= self.FORCE_WINDOW_STEPS:
                avg_force = np.mean(self._force_history[-self.FORCE_WINDOW_STEPS:])
                if avg_force >= self.GRASP_FORCE_THRESHOLD_N:
                    self._advance_phase()  # grasp confirmed → lift
        elif self._current_phase == "transit":
            force_norm = np.linalg.norm(obs.ft_force) if obs.ft_force is not None else 0.0
            if force_norm < self.LIFT_FORCE_LOSS_THRESHOLD_N:
                self._rewind_to("close_gripper")  # dropped → regrasp
        return super().get_action(obs)
```

### D3. Reward + Success Predicates Using Sensors — extend `robodeploy/tasks/success_predicates.py`

Add (depends on Goal 1 D10 if delivered, else standalone):

```python
@register_success("grasp_force_min")
def grasp_force_min(obs, *, threshold_N: float = 2.0, window: int = 5) -> bool:
    """FT-based grasp confirmation."""
    if obs.ft_force is None: return False
    return float(np.linalg.norm(obs.ft_force)) >= threshold_N

@register_success("contact_held")
def contact_held(obs, *, sensor_name: str = "wrist_contact") -> bool:
    if obs.contact_state is None: return False
    return obs.contact_state.get(sensor_name, False)

@register_success("imu_stable")
def imu_stable(obs, *, max_angular_velocity: float = 0.3, max_acceleration: float = 2.0) -> bool:
    """Stability check using IMU. Useful for ‘hold pose' success."""
    if obs.imu_angular_velocity is None: return False
    omega = float(np.linalg.norm(obs.imu_angular_velocity))
    acc_excess = (float(np.linalg.norm(obs.imu_acceleration)) - 9.81) if obs.imu_acceleration is not None else 0.0
    return omega <= max_angular_velocity and abs(acc_excess) <= max_acceleration

@register_success("vision_target_in_view")
def vision_target_in_view(obs, *, target_color_hsv_range, min_pixels: int = 100) -> bool:
    """Color-blob based vision target check."""
    if obs.rgb is None: return False
    hsv = cv2.cvtColor(obs.rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, *target_color_hsv_range)
    return int(np.sum(mask > 0)) >= min_pixels
```

### D4. Reward Components from Sensors — extend `RewardBuilder` (Goal 1 D9)

```python
class RewardBuilder:
    def penalty_excessive_force(self, *, threshold_N: float = 20.0, scale: float = 0.1) -> "RewardBuilder":
        """Soft penalty if FT force exceeds threshold (collision avoidance)."""
        def term(obs, action):
            if obs.ft_force is None: return 0.0
            excess = max(0.0, float(np.linalg.norm(obs.ft_force)) - threshold_N)
            return -scale * excess
        ...

    def bonus_grasp_force(self, *, min_N: float = 1.0, max_N: float = 5.0, scale: float = 0.05) -> "RewardBuilder": ...
    def penalty_jerk_imu(self, *, scale: float = 0.01) -> "RewardBuilder": ...
    def bonus_visual_alignment(self, *, target_hsv_range, scale: float = 0.1) -> "RewardBuilder": ...
```

### D5. Vision Perception Transform — `robodeploy/perception/vision_predicates.py` (NEW, ~250 lines)

Lightweight perception module — segments + blob tracks + classical pose from `obs.rgb` + `obs.depth` + `obs.camera_extrinsics`.

```python
class ColorBlobTracker:
    """Detects colored blob; outputs SE(3) pose by unprojecting centroid."""
    def __init__(self, hsv_range, min_pixels=200): ...
    def detect(self, rgb, depth, intrinsics, extrinsics) -> Pose3D | None: ...

class ArUcoTracker:
    """OpenCV ArUco marker detection."""
    def __init__(self, marker_size_m=0.04, dictionary="DICT_4X4_50"): ...
    def detect(self, rgb, intrinsics, extrinsics) -> dict[int, Pose3D]: ...

class LearnedPoseEstimator:
    """Wrapper for user-injected nn.Module predicting object pose from RGB-D."""
    def __init__(self, model_fn: Callable[[np.ndarray, np.ndarray], dict[str, Pose3D]]): ...
```

Register as obs_pipeline transform — populates `obs.objects` from vision (replaces oracle).

### D6. Sensor Health Surfaced in Step Info — `robodeploy/env.py`

Modify `env.step()` so `info.extra["sensor_status"]` carries the per-frame `sensor_status` dict.

```python
# robodeploy/env.py step():
obs, reward, done, info = ...
info.extra["sensor_status"] = obs.sensor_status or {}
info.extra["sensor_health"] = _summarize_health(obs.sensor_status)  # "ok" | "degraded" | "failed"
```

Policies + monitoring can react to stale frames (e.g., trigger e-stop on FT failure during grasp).

### D7. IMU Stability Gate in Reach Policy — extend `ReachTrajectoryPolicy`

Phase advancement waits for IMU settle in addition to position settle:

```yaml
phases:
  - name: pregrasp
    kind: reach
    target: source
    offset: [0, 0, 0.10]
    settle:
      position_threshold: 0.025
      imu_omega_max: 0.3   # rad/s, NEW
      hold_steps: 5
```

### D8. Multi-Modal Fusion Transform — `robodeploy/obs_pipeline/transforms/fusion.py` (NEW)

Cross-modal predicates layered atop raw sensors:

```python
class GraspStabilityFusion(ObsTransform):
    """Combines FT magnitude + IMU stillness + contact state into a scalar [0,1]."""
    def __call__(self, obs: Observation) -> Observation:
        ft_score = self._ft_in_range(obs.ft_force) if obs.ft_force is not None else 0.0
        imu_score = self._imu_still(obs.imu_angular_velocity) if obs.imu_angular_velocity is not None else 0.0
        contact_score = 1.0 if (obs.contact_state and obs.contact_state.get("wrist_contact")) else 0.0
        score = 0.4 * ft_score + 0.3 * imu_score + 0.3 * contact_score
        obs.metadata["grasp_stability"] = float(score)
        return obs
```

### D9. Example: kuka_ft_imu_pick — `examples/kuka_ft_imu_pick_mujoco/` (NEW)

Demonstrates full sensor-driven pipeline:
- Scene with FT + IMU + camera + contact sensors.
- `FTReachPickPolicy` from D2.
- Task uses `grasp_force_min` + `imu_stable` success predicates.
- Reward uses `penalty_excessive_force` + `bonus_grasp_force` + `bonus_lift`.
- README with explanation of each sensor's role.

### D10. Tests — `tests/test_sensor_policy_integration.py`, `tests/test_sensor_task_integration.py`, `tests/test_contact_sensor.py`, `tests/test_vision_predicates.py`, `tests/test_grasp_fusion.py`

Each tests sensor consumption against synthetic obs streams (no live sim required for unit tests).

---

## Phased Rollout

### Phase 3.1 — Foundation (~10h)
- D1 ContactSensor (MuJoCo + Gazebo + Real-FT-threshold).
- Add `Observation.contact_state` field + obs_pipeline population.
- D6 sensor health in step info.
- `tests/test_contact_sensor.py`.

### Phase 3.2 — Predicates + RewardBuilder (~10h)
- D3 success predicates (grasp_force_min, contact_held, imu_stable, vision_target_in_view).
- D4 reward terms (penalty_excessive_force, bonus_grasp_force, etc.).
- `tests/test_sensor_task_integration.py`.

### Phase 3.3 — FT Policy + IMU gate (~10h)
- D2 FTReachPickPolicy with sensor-driven grasp state machine.
- D7 IMU stability gate option in reach DSL.
- `tests/test_sensor_policy_integration.py`.

### Phase 3.4 — Vision + Fusion (~7h)
- D5 ColorBlobTracker + ArUcoTracker.
- D8 GraspStabilityFusion transform.
- `tests/test_vision_predicates.py`, `tests/test_grasp_fusion.py`.

### Phase 3.5 — Example + Docs (~3h)
- D9 kuka_ft_imu_pick example.
- README explaining sensor consumption patterns.

---

## Acceptance Criteria

- [ ] FTReachPickPolicy successfully grasps with sensor-driven force threshold (not `backend.has_prop_contact`).
- [ ] Task fails when FT force exceeds 20 N (collision avoidance reward).
- [ ] `imu_stable` predicate distinguishes settled vs swinging EE.
- [ ] `obs.contact_state` populated by ContactSensor in MuJoCo + Gazebo backends.
- [ ] `obs.objects` populated by ColorBlobTracker from camera RGB-D + extrinsics.
- [ ] `info.extra["sensor_status"]` shows per-sensor health each step.
- [x] Drop-detection regrasp loop works (force loss → rewind to grasp phase).
- [ ] kuka_ft_imu_pick example completes ≥80% trials.
- [ ] All sensor predicates covered by unit tests with synthetic Observations.

## Risks

- **Threshold tuning**: GRASP_FORCE_THRESHOLD_N depends on object mass + gripper compliance. Mitigation: per-task config override + auto-calibration helper.
- **Real FT noise**: 2 N threshold may trigger on settling jitter. Mitigation: median filter window in `FTReachPickPolicy._force_history`.
- **Vision compute cost**: ColorBlob @ 480p ≥10ms per frame; may slow real-time loop. Mitigation: subsample, run async in `obs_pipeline`.
- **Sim FT calibration**: MuJoCo FT site may not match real ATI scaling. Mitigation: sim-side scale factor in DomainRandomizer.

## Out of Scope

- Learned vision (NN-based pose estimation). Goal 9.
- Multi-camera fusion (3D reconstruction). External tool.
- Tactile array sensors. Future hardware support.
