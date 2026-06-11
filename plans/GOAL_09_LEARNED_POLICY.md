# Goal 9 — Learned Policy Integration

**Priority**: Tier 3. **Effort**: ~25h. **Touches**: VLA / diffusion / BC story.

## Problem

`VLAPolicy` / `DiffusionPolicy` / `RobomimicPolicy` = thin shims requiring user-injected `predict_fn`. No standard checkpoint resolution, no model config validation, no action-space inference. Users wire adapters by hand. SafetyFilter class referenced (`core/types.py:102-103`) but not implemented.

## Current State (Audit)

### Learned policy classes
- `robodeploy/policies/learned/robomimic.py:23-123` — accepts `predict_fn: Callable[[dict[str, np.ndarray]], np.ndarray]`. JOINT_POS only. Exponential smoothing.
- `robodeploy/policies/learned/diffusion.py:21-177` — accepts `predict_plan_fn`. Queues `replan_interval` plan horizon. DELTA_EE default; CARTESIAN_POSE / JOINT_POS / JOINT_VEL via config.
- `robodeploy/policies/learned/vla.py:21-203` — accepts `predict_fn` + `predict_batch_fn`. Selects RGB/depth by camera name. Keyword + image-com heuristic fallback.

### Spaces + Validation
- `core/spaces.py` — ActionSpace enum (JOINT_POS, JOINT_VEL, JOINT_TORQUE, CARTESIAN_POSE, DELTA_EE).
- `core/spaces.py:41-60` — `infer_action_space(action)` checks populated fields.
- No SafetyFilter class anywhere — referenced in comment but missing.
- Action numeric bounds: no per-joint clip enforced at ingestion.
- Backend validates `supported_action_spaces` at construction; no shape/dtype runtime checks.

### Action adapter
- `robodeploy/action_adapter.py` — exists; converts actions across spaces (DELTA_EE → JOINT_POS via IK, etc.).
- No standard injection point for learned policies.

### Remote inference
- `robodeploy/policies/remote/server.py` — PolicyServer (ZMQ/gRPC/HTTP).
- `robodeploy/policies/remote/http_client.py` — HTTP JSON client.

---

## Deliverables

### D1. ModelLoader + Checkpoint Resolution — `robodeploy/policies/learned/loader.py` (NEW, ~250 lines)

Centralizes model-checkpoint discovery, validation, lazy load.

```python
class ModelSpec(TypedDict):
    framework: Literal["robomimic","diffusion","openvla","pi0","custom"]
    checkpoint: str | Path        # filesystem path | s3://... | hf://repo/path
    config_path: str | None
    expected_action_space: ActionSpace
    expected_action_dim: int
    expected_obs_keys: list[str]
    obs_normalization: dict | None    # mean/std per-key
    action_normalization: dict | None

class ModelLoader:
    """Resolve checkpoint, instantiate model, validate I/O contracts."""

    def __init__(self, *, search_paths: list[Path] | None = None,
                 hf_cache: Path | None = None, s3_client: Any = None):
        self._paths = search_paths or [Path.home() / ".robodeploy" / "models"]
        ...

    def resolve(self, ref: str | Path) -> Path:
        if str(ref).startswith("hf://"):
            return self._download_hf(ref)
        elif str(ref).startswith("s3://"):
            return self._download_s3(ref)
        for root in self._paths:
            candidate = root / ref
            if candidate.exists(): return candidate
        raise FileNotFoundError(f"Model {ref} not found in {self._paths}")

    def load(self, spec: ModelSpec) -> "LoadedModel":
        ckpt_path = self.resolve(spec["checkpoint"])
        framework = spec["framework"]
        if framework == "robomimic":
            return self._load_robomimic(ckpt_path, spec)
        elif framework == "diffusion":
            return self._load_diffusion(ckpt_path, spec)
        elif framework == "openvla":
            return self._load_openvla(ckpt_path, spec)
        elif framework == "pi0":
            return self._load_pi0(ckpt_path, spec)
        elif framework == "custom":
            return self._load_custom(ckpt_path, spec)
        raise ValueError(f"Unknown framework: {framework}")

    def _validate(self, model: "LoadedModel", spec: ModelSpec):
        if model.action_dim != spec["expected_action_dim"]:
            raise ModelContractError(...)
        if set(model.required_obs_keys) - set(spec["expected_obs_keys"]):
            raise ModelContractError(...)

@dataclass
class LoadedModel:
    predict_fn: Callable[[dict[str, np.ndarray]], np.ndarray]
    predict_batch_fn: Callable[[list[dict]], np.ndarray] | None
    action_space: ActionSpace
    action_dim: int
    required_obs_keys: list[str]
    framework: str
    metadata: dict
```

### D2. SafetyFilter — `robodeploy/kinematics/safety.py` (NEW, ~300 lines)

Implements the SafetyFilter class referenced in `core/types.py:102-103`. Hard joint position/velocity clamps + slew + workspace bounds.

```python
@dataclass
class SafetyLimits:
    joint_position_min: np.ndarray | None = None       # [dof] rad
    joint_position_max: np.ndarray | None = None       # [dof] rad
    joint_velocity_max: np.ndarray | None = None       # [dof] rad/s
    joint_acceleration_max: np.ndarray | None = None   # [dof] rad/s^2
    workspace_box: tuple[np.ndarray, np.ndarray] | None = None  # (low_xyz, high_xyz)
    ee_velocity_max: float | None = None               # m/s
    force_max: float | None = None                     # N (FT threshold for halt)
    torque_max: float | None = None                    # N*m
    control_hz: float = 100.0

class SafetyFilter:
    """Clamps actions to safe envelope. Raises SafetyError on hard violations."""

    def __init__(self, *, limits: SafetyLimits, description: RobotDescription,
                 on_violation: Literal["clamp","halt","raise"] = "clamp",
                 verbose: bool = False):
        self._limits = limits or limits_from_description(description)
        self._mode = on_violation
        self._last_q = None

    def filter(self, action: Action, obs: Observation) -> Action:
        # 1. Clamp joint positions to limits.
        # 2. Clamp joint velocity (slew).
        # 3. Clamp joint acceleration.
        # 4. Project EE pose to workspace box.
        # 5. Check force/torque threshold (halt if exceeded).
        ...

    def violations(self) -> list[ViolationRecord]: ...

@dataclass
class ViolationRecord:
    kind: Literal["joint_position","joint_velocity","ee_workspace","force"]
    value: float
    limit: float
    timestamp: float
```

Wire into `RoboEnv`:
- `RoboEnv.step()` calls `self._safety_filter.filter(action, last_obs)` before backend step.
- Configurable via `EnvConfig.safety = SafetyLimits(...)`.
- Default limits derived from `RobotDescription.joint_limits`.

### D3. ActionSpace Auto-Adaptation — `robodeploy/policies/learned/adapter.py` (NEW, ~250 lines)

Auto-bridge model's action output to backend's required ActionSpace.

```python
class LearnedActionAdapter:
    """Converts model output to backend-compatible Action.

    Examples:
        Model outputs DELTA_EE; backend supports JOINT_POS → IK solve.
        Model outputs JOINT_POS [dof=7]; backend supports JOINT_POS [dof=8 with gripper] → append gripper.
        Model outputs normalized [-1,1]; un-normalize using action_normalization.
    """

    def __init__(self, *, source_space: ActionSpace, target_space: ActionSpace,
                 source_dim: int, target_dim: int,
                 ik_solver: IKSolver | None = None,
                 normalization: dict | None = None,
                 gripper_index_source: int | None = None,
                 gripper_index_target: int | None = None):
        ...

    def __call__(self, model_output: np.ndarray, obs: Observation) -> Action:
        # 1. Unnormalize.
        # 2. Extract gripper if separate.
        # 3. Convert space (DELTA_EE → JOINT_POS via IK, etc.).
        # 4. Validate output dim matches target.
        # 5. Return Action dataclass.
        ...
```

### D4. Refactor Learned Policy Classes — `policies/learned/robomimic.py`, `diffusion.py`, `vla.py`

Reduce duplication by inheriting from `LearnedPolicyBase`.

```python
class LearnedPolicyBase(PolicyBase):
    """Common scaffold: ModelLoader + ActionAdapter + obs preprocessing."""

    def __init__(self, *, model_spec: ModelSpec, action_space: ActionSpace,
                 obs_keys: list[str] | None = None, config: dict | None = None,
                 adapter_kwargs: dict | None = None, loader: ModelLoader | None = None):
        super().__init__(action_space=action_space, config=config or {})
        self._loader = loader or ModelLoader()
        self._model = self._loader.load(model_spec)
        self._adapter = LearnedActionAdapter(
            source_space=self._model.action_space, target_space=action_space,
            source_dim=self._model.action_dim, target_dim=self._infer_target_dim(),
            **(adapter_kwargs or {}),
        )
        self._obs_preprocess = self._build_preprocess(obs_keys or self._model.required_obs_keys)

    def get_action(self, obs: Observation) -> Action:
        obs_dict = self._obs_preprocess(obs)
        model_output = self._model.predict_fn(obs_dict)
        return self._adapter(model_output, obs)

    def get_action_batch(self, obs_list: list[Observation]) -> list[Action]:
        if self._model.predict_batch_fn is None:
            return [self.get_action(o) for o in obs_list]
        batch = [self._obs_preprocess(o) for o in obs_list]
        outputs = self._model.predict_batch_fn(batch)
        return [self._adapter(out, obs) for out, obs in zip(outputs, obs_list)]
```

`RobomimicPolicy`, `DiffusionPolicy`, `VLAPolicy` become 30-50 line subclasses adding framework-specific quirks (queueing for diffusion, camera selection for VLA).

### D5. Hugging Face Model Registry — `robodeploy/policies/learned/hf_hub.py` (NEW, ~150 lines)

```python
class HFModelRegistry:
    """Pull pre-trained policies from Hugging Face Hub."""

    KNOWN_MODELS = {
        "openvla-7b": ModelSpec(framework="openvla", checkpoint="hf://openvla/openvla-7b",
                               expected_action_space=ActionSpace.DELTA_EE, expected_action_dim=7,
                               expected_obs_keys=["rgb","instruction"]),
        "octo-base": ModelSpec(framework="custom", checkpoint="hf://rail-berkeley/octo-base", ...),
        "pi0-base":  ModelSpec(framework="pi0", checkpoint="hf://physical-intelligence/pi0", ...),
    }

    @classmethod
    def from_name(cls, name: str, *, action_space: ActionSpace) -> LearnedPolicyBase:
        spec = cls.KNOWN_MODELS.get(name)
        if spec is None: raise ValueError(f"Unknown model {name}; try {list(cls.KNOWN_MODELS)}")
        return LearnedPolicyBase(model_spec=spec, action_space=action_space)
```

CLI:
```bash
robodeploy models list                       # show known models
robodeploy models download openvla-7b        # cache locally
robodeploy run-episode --preset X --policy hf:openvla-7b
```

### D6. Action-Space Compatibility Negotiation — `robodeploy/env.py`

At env construction, detect mismatch + auto-insert adapter:

```python
def _negotiate_action_space(policy: IPolicy, backend: IBackend) -> tuple[IPolicy, ActionSpace]:
    if policy.action_space in backend.supported_action_spaces:
        return policy, policy.action_space
    # Find adapter path: e.g., DELTA_EE → JOINT_POS via IK
    target = backend.supported_action_spaces[0]
    if can_adapt(policy.action_space, target):
        adapter = ActionSpaceAdapter(policy.action_space, target)
        return AdaptedPolicy(policy, adapter), target
    raise ActionSpaceIncompatibility(
        f"Policy outputs {policy.action_space} but backend supports {backend.supported_action_spaces}. "
        f"No adapter available."
    )
```

### D7. Policy Server Streaming — `robodeploy/policies/remote/server.py` (EXTEND)

For VLA / diffusion that take >100ms inference: streaming chunked plan output + async client. Lets policy server start emitting first action while later actions still computing.

```python
class StreamingPolicyServer:
    async def predict_stream(self, obs):
        for chunk in self._model.predict_chunked(obs, chunk_size=4):
            yield chunk
```

Async HTTP client picks up first chunk → starts executing.

### D8. Replan-Aware ReachTrajectoryPolicy Integration

Allow learned policy to take over mid-trajectory of `ReachTrajectoryPolicy`:

```yaml
phases:
  - {name: pregrasp, kind: reach, target: source, offset: [0,0,0.10]}
  - {name: handoff, kind: learned, policy: hf:openvla-7b, instruction: "Pick up the cube",
                    fallback_to: reach, fallback_target: source, max_steps: 200}
```

Allows hybrid scripted + learned trajectories.

### D9. Action Distribution Reporting — `robodeploy/policies/diagnostics.py` (NEW, ~150 lines)

Track action distribution online; warn on suspicious behavior (always-zero gripper, NaN actions, dim mismatch).

```python
class PolicyDiagnostics:
    def record(self, action: Action): ...
    def summary(self) -> dict:
        return {
            "action_mean": ..., "action_std": ...,
            "action_min": ..., "action_max": ...,
            "nan_count": ..., "clipped_count": ...,
        }
```

Surface in `info.extra["policy_diagnostics"]` (used by Goal 10).

### D10. Tests
- `tests/test_safety_filter.py` — joint position/velocity/workspace clamping.
- `tests/test_action_adapter.py` — DELTA_EE → JOINT_POS via IK.
- `tests/test_model_loader.py` — hf:// resolution (mocked), checkpoint validation, contract errors.
- `tests/test_learned_policy_base.py` — predict + adapt + filter pipeline.
- `tests/test_action_space_negotiation.py` — auto-adapter insertion.

---

## Phased Rollout

### Phase 9.1 — SafetyFilter (~8h)
- D2 SafetyFilter + SafetyLimits.
- Wire into RoboEnv.step.
- `tests/test_safety_filter.py`.

### Phase 9.2 — ModelLoader + Adapter (~8h)
- D1 ModelLoader (filesystem + hf + s3).
- D3 LearnedActionAdapter.
- D6 negotiation logic.
- `tests/test_model_loader.py`, `tests/test_action_adapter.py`.

### Phase 9.3 — LearnedPolicyBase refactor (~5h)
- D4 LearnedPolicyBase + refactor existing three classes.
- `tests/test_learned_policy_base.py`.

### Phase 9.4 — HF registry + streaming + diagnostics (~4h)
- D5 HFModelRegistry + CLI.
- D7 streaming server.
- D8 replan-aware hybrid policy.
- D9 PolicyDiagnostics.

---

## Acceptance Criteria

- [x] `SafetyFilter` clamps action exceeding joint limits to nearest valid value.
- [x] `SafetyFilter(on_violation="raise")` raises `SafetyError` on workspace exit.
- [x] `RoboEnv` applies `SafetyFilter` to all policy actions before backend.step.
- [x] `ModelLoader.load(spec)` validates `action_dim` mismatch (raises `ModelContractError`).
- [x] `LearnedActionAdapter` converts DELTA_EE → JOINT_POS via IK; round-trip error <1mm.
- [x] `RoboEnv` with DELTA_EE policy + JOINT_POS backend auto-inserts adapter without code change.
- [x] `HFModelRegistry.from_name("openvla-7b")` downloads from HF Hub on first call.
- [x] `RobomimicPolicy`, `DiffusionPolicy`, `VLAPolicy` ≤ 50 lines each after refactor — done 2026-06-11: 50 / 49 / 43 lines (`robomimic.py` / `diffusion.py` / `vla.py`); smoothing (`ActionSmoother`, `arm_gripper_action`), plan queueing (`PlanQueue`, `build_plan`, `batch_first_actions`), and camera/heuristic paths (`vla_packet`, `vla_heuristic_action`, `select_camera_image/depth`) extracted to `helpers.py`; LOC regression test `tests/test_learned_policy_base.py::test_learned_policy_files_at_most_50_lines`.
- [x] `info.extra["policy_diagnostics"]` shows action stats per step.
- [x] Streaming HTTP client receives first action chunk within 50ms of large-VLA call.

## Dependencies

- `huggingface_hub>=0.20` (for hf:// resolution).
- `boto3` (optional, for s3://).
- `torch>=2.0` (most learned policies).
- `openvla`, `octo`, `pi_zero` (optional model packages).

Add to `[project.optional-dependencies] learned = [torch, huggingface_hub, transformers]`.

## Risks

- **HF model API churn**: OpenVLA / Octo / Pi0 expose different `predict()` signatures. Mitigation: framework-specific wrapper in `_load_<framework>`.
- **IK adapter fails at singularity**: DELTA_EE → JOINT_POS round-trip drifts. Mitigation: damped least squares + fallback to last valid q.
- **SafetyFilter false positives**: aggressive limits clip valid demos during BC. Mitigation: per-policy override + verbose logging.
- **Action normalization mismatch**: model trained with different scale than deployed. Mitigation: read normalization from checkpoint metadata when present; warn if absent.

## Out of Scope

- LoRA fine-tuning of VLAs. Goal 2 extension.
- Multi-modal policy fusion (vision + language + tactile). Future.
- On-device quantization (int8 / fp16). External tools (TensorRT, ONNX Runtime).
- Online policy distillation. Future.
