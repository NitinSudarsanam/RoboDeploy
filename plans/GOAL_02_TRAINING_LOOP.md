# Goal 2 — Build Training Loop

**Priority**: Tier 1. **Effort**: ~80h. **Touches**: every learning workflow.

## Problem

RoboDeploy = inference + sim env runtime. Zero training code:
- `robodeploy/vec_env.py` is `SequentialVecEnv` only (no subprocess, no GPU).
- No loss, optimizer, gradient, dataset loader, replay buffer.
- Learned policy classes (`VLAPolicy`, `DiffusionPolicy`, `RobomimicPolicy`) only load pre-trained checkpoints via user-injected `predict_fn`.
- `env.step()` returns 4-tuple `(obs, reward, done, info)` — **not gymnasium 5-tuple** (no `terminated`/`truncated` split).
- No `gym.Space` for action/observation.
- Demo recording exists (`demo_recording.py`) but no dataset loader/batcher/collator.
- No HuggingFace / LeRobot / RLDS / Robomimic dataset adapters.

## Current State (Audit)

### VecEnv
- `robodeploy/vec_env.py:10-47` — `SequentialVecEnv` sequential loop over instances.
- No process pool, no async, no GPU sim hook.

### Demo Recording
- `robodeploy/demo_recording.py:34-59` — `DemoRecorder` collects `DemoFrame`.
- `robodeploy/demo_recording.py:70-89` — `DemoSession` auto-records `env.step()`.
- Export: `export_demo_jsonl()`, `export_demo_hdf5()`.
- Replay: `iter_replay_actions()`.

### Learned Policies
- `robodeploy/policies/learned/robomimic.py:23-123` — accepts `predict_fn: Callable[[dict], np.ndarray]`. JOINT_POS only.
- `robodeploy/policies/learned/diffusion.py:21-177` — accepts `predict_plan_fn: Callable[[dict], list[Action]]`. Queues plan horizon.
- `robodeploy/policies/learned/vla.py:21-203` — accepts `predict_fn` + `predict_batch_fn`.
- All three: inference-only wrappers.

### Spaces
- `robodeploy/core/spaces.py` — `ActionSpace` enum (JOINT_POS, JOINT_VEL, JOINT_TORQUE, CARTESIAN_POSE, DELTA_EE).
- `Action` + `Observation` = dataclasses, **not `gym.spaces.Box`**.
- `RoboEnv` has no `action_space` / `observation_space` properties.

### Env Lifecycle
- `robodeploy/env.py:565-681` — `reset()` returns `(obs, info)`, `step()` returns `(obs, reward, done, info)`.
- `done = success or failure` — no truncation.

---

## Deliverables

### D1. Gymnasium-Compatible Env Adapter — `robodeploy/training/gym_adapter.py` (NEW, ~250 lines)

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from robodeploy.env import RoboEnv
from robodeploy.core.types import Action, Observation

class GymRoboEnv(gym.Env):
    """Adapts RoboEnv to gymnasium API: 5-tuple step, Box spaces."""
    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(self, robo_env: RoboEnv, *, max_episode_steps: int | None = None):
        self._env = robo_env
        self._action_space = self._build_action_space()
        self._observation_space = self._build_observation_space()
        self._max_steps = max_episode_steps or robo_env.max_episode_steps or 1000
        self._step_count = 0

    @property
    def action_space(self) -> spaces.Space: return self._action_space
    @property
    def observation_space(self) -> spaces.Space: return self._observation_space

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        obs, info = self._env.reset(seed=seed)
        self._step_count = 0
        return self._obs_to_array(obs), self._info_dict(info)

    def step(self, action_array: np.ndarray):
        action = self._array_to_action(action_array)
        obs, reward, done, info = self._env.step(action)
        self._step_count += 1
        truncated = self._step_count >= self._max_steps
        terminated = done and not truncated
        return self._obs_to_array(obs), float(reward), terminated, truncated, self._info_dict(info)

    def _build_action_space(self) -> spaces.Box:
        # Derive low/high from robot joint limits (description.joint_limits)
        # Construct flat Box matching robot.action_space
        ...

    def _build_observation_space(self) -> spaces.Dict:
        # Dict space: proprio (Box), rgb (Box uint8), depth (Box float32), ee (Box), ft (Box)
        # Conditional on task.obs_spec()
        ...

    def _obs_to_array(self, obs: Observation) -> dict: ...
    def _array_to_action(self, arr: np.ndarray) -> Action: ...
```

### D2. ParallelVecEnv — `robodeploy/training/parallel_vec_env.py` (NEW, ~350 lines)

```python
class SubprocVecEnv:
    """Each env in subprocess. Mirrors Stable-Baselines3 SubprocVecEnv."""
    def __init__(self, env_fns: list[Callable[[], GymRoboEnv]], *, start_method: str = "spawn"):
        ctx = mp.get_context(start_method)
        self._parent_conns, self._child_conns = zip(*[ctx.Pipe() for _ in env_fns])
        self._workers = [ctx.Process(target=_worker, args=(child, fn)) for child, fn in zip(self._child_conns, env_fns)]
        for w in self._workers: w.daemon = True; w.start()

    def reset(self, seeds: list[int] | None = None): ...
    def step(self, actions: np.ndarray): ...
    def close(self): ...

def _worker(conn, env_fn):
    env = env_fn()
    while True:
        cmd, data = conn.recv()
        if cmd == "step": conn.send(env.step(data))
        elif cmd == "reset": conn.send(env.reset(seed=data))
        elif cmd == "close": env.close(); conn.close(); break
```

Plus `AsyncVecEnv` variant using `asyncio` for I/O-bound real backends.

### D3. Dataset Pipeline — `robodeploy/training/dataset.py` (NEW, ~400 lines)

```python
class DemoDataset(torch.utils.data.Dataset):
    """Loads recorded demos (JSONL/HDF5/LeRobot/RLDS)."""

    @classmethod
    def from_jsonl(cls, path: str | Path, *, obs_keys: list[str] | None = None) -> "DemoDataset": ...
    @classmethod
    def from_hdf5(cls, path: str | Path) -> "DemoDataset": ...
    @classmethod
    def from_lerobot(cls, repo_id: str) -> "DemoDataset": ...  # HuggingFace LeRobot format
    @classmethod
    def from_robomimic(cls, path: str | Path) -> "DemoDataset": ...
    @classmethod
    def from_rlds(cls, builder, split: str = "train") -> "DemoDataset": ...

    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> dict[str, Tensor]: ...

class SequenceDataset(DemoDataset):
    """Windowed trajectories for diffusion / sequence policies."""
    def __init__(self, base: DemoDataset, *, horizon: int, pad_strategy: str = "last"): ...

class DemoCollator:
    """Pads variable-length trajectories. Handles dict obs."""
    def __call__(self, batch: list[dict]) -> dict[str, Tensor]: ...
```

### D4. Trainer Framework — `robodeploy/training/trainer.py` (NEW, ~500 lines)

```python
@dataclass
class TrainerConfig:
    lr: float = 1e-4
    batch_size: int = 64
    epochs: int = 100
    grad_clip: float = 1.0
    eval_interval: int = 1000
    checkpoint_interval: int = 5000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_dir: str = "./runs"
    seed: int = 42

class Trainer:
    def __init__(self, *, policy_module: nn.Module, dataset: DemoDataset,
                 loss_fn: Callable, optimizer_fn: Callable, config: TrainerConfig,
                 eval_env: GymRoboEnv | None = None, callbacks: list["TrainerCallback"] = ()):
        self.policy = policy_module.to(config.device)
        self.optim = optimizer_fn(self.policy.parameters())
        self.loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, collate_fn=DemoCollator())
        ...

    def fit(self): ...
    def step(self, batch) -> dict[str, float]: ...
    def evaluate(self) -> dict[str, float]: ...
    def save_checkpoint(self, path: str): ...
    def load_checkpoint(self, path: str): ...

class TrainerCallback:
    def on_step_end(self, trainer, metrics): ...
    def on_eval_end(self, trainer, metrics): ...
    def on_checkpoint(self, trainer, path): ...
```

### D5. BC Loss + Policy — `robodeploy/training/bc.py` (NEW, ~200 lines)

```python
class BCPolicyModule(nn.Module):
    """MLP or CNN+MLP policy. Outputs ActionSpace-shaped tensor."""
    def __init__(self, obs_keys: list[str], action_dim: int, *, hidden=(256, 256), encoder: str = "mlp"):
        super().__init__()
        self.encoder = build_encoder(obs_keys, encoder)
        self.head = MLP(self.encoder.out_dim, hidden, action_dim)

    def forward(self, obs_dict) -> Tensor: ...

def bc_mse_loss(pred: Tensor, target: Tensor, mask: Tensor | None = None) -> Tensor:
    if mask is not None: return ((pred - target) ** 2 * mask).sum() / mask.sum()
    return F.mse_loss(pred, target)

def bc_gaussian_nll_loss(mu: Tensor, log_std: Tensor, target: Tensor) -> Tensor: ...

# Convenience:
def train_bc(*, dataset, obs_keys, action_dim, config, eval_env=None) -> BCPolicyModule: ...
```

### D6. PPO Loop — `robodeploy/training/ppo.py` (NEW, ~600 lines)

```python
@dataclass
class PPOConfig:
    rollout_steps: int = 2048
    n_envs: int = 8
    n_epochs: int = 10
    minibatch_size: int = 256
    clip_range: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    gae_lambda: float = 0.95
    gamma: float = 0.99
    target_kl: float | None = 0.03
    lr: float = 3e-4
    total_steps: int = 1_000_000

class ActorCritic(nn.Module):
    def __init__(self, obs_keys, action_dim, *, hidden=(256, 256)): ...
    def forward(self, obs) -> tuple[Distribution, Tensor]: ...   # (policy_dist, value)

class PPOTrainer:
    def __init__(self, *, env: SubprocVecEnv, model: ActorCritic, config: PPOConfig, callbacks=()): ...
    def collect_rollouts(self) -> RolloutBuffer: ...
    def compute_gae(self, buffer): ...
    def update(self, buffer) -> dict[str, float]: ...
    def fit(self): ...

class RolloutBuffer:
    obs: dict[str, Tensor]; actions: Tensor; rewards: Tensor; values: Tensor
    log_probs: Tensor; advantages: Tensor; returns: Tensor; dones: Tensor
    def add(self, ...): ...
    def get(self, minibatch_size) -> Iterator: ...
```

### D7. Trainable Policy Wrappers — `robodeploy/policies/trainable_base.py` (NEW, ~150 lines)

```python
class TrainablePolicyBase(PolicyBase):
    """PolicyBase subclass that wraps a torch.nn.Module + handles train/eval modes."""
    def __init__(self, *, module: nn.Module, action_space: ActionSpace, config: dict | None = None):
        super().__init__(action_space=action_space, config=config or {})
        self._module = module
        self._device = next(module.parameters()).device

    def get_action(self, obs: Observation) -> Action:
        with torch.no_grad():
            obs_dict = self._obs_to_dict(obs)
            action_tensor = self._module(obs_dict)
            return self._tensor_to_action(action_tensor)

    def train_mode(self): self._module.train()
    def eval_mode(self): self._module.eval()
    def state_dict(self) -> dict: return self._module.state_dict()
    def load_state_dict(self, sd): self._module.load_state_dict(sd)
    def save_checkpoint(self, path: str | Path): ...
    @classmethod
    def from_checkpoint(cls, path: str | Path, action_space: ActionSpace) -> "TrainablePolicyBase": ...
```

### D8. Callbacks — `robodeploy/training/callbacks.py` (NEW, ~200 lines)

```python
class WandbCallback(TrainerCallback): ...
class TensorBoardCallback(TrainerCallback): ...
class CheckpointCallback(TrainerCallback):
    def __init__(self, *, save_dir, every_n_steps, keep_top_k=5, metric="eval/success_rate", mode="max"): ...
class EvalCallback(TrainerCallback):
    def __init__(self, *, eval_env, n_episodes=10, deterministic=True): ...
class EarlyStoppingCallback(TrainerCallback): ...
```

### D9. CLI — `robodeploy/cli.py` (EXTEND)

```bash
robodeploy train bc --dataset demos.jsonl --policy mlp --obs proprio,rgb --action-dim 8 --epochs 100
robodeploy train ppo --preset kuka_pick_mujoco --n-envs 16 --total-steps 1000000 --log wandb
robodeploy eval --checkpoint runs/best.pt --preset kuka_pick_mujoco --episodes 100
robodeploy convert-dataset --from lerobot://lerobot/aloha --to demos.hdf5
```

### D10. Reward Logging in env.step()

Modify `robodeploy/env.py:782` so `info["reward_components"]` carries per-term reward breakdown (from `RewardBuilder.build_components()`). Required for diagnostic training plots.

### D11. Replay Buffer (for off-policy RL) — `robodeploy/training/replay_buffer.py` (NEW, ~250 lines)

```python
class ReplayBuffer:
    """Uniform sample. For SAC/TD3 extension."""
    def __init__(self, capacity: int, obs_shape, action_dim, device="cpu"): ...
    def add(self, obs, action, reward, next_obs, done): ...
    def sample(self, batch_size) -> tuple[Tensor, ...]: ...

class PrioritizedReplayBuffer(ReplayBuffer): ...
```

---

## Phased Rollout

### Phase 2.1 — Gym adapter + ParallelVecEnv (~15h)
- D1 GymRoboEnv. Verify `gym.make()` + Stable-Baselines3 PPO runs out-of-box.
- D2 SubprocVecEnv. `tests/test_subproc_vec_env.py`: 4-env throughput >3× sequential.

### Phase 2.2 — Dataset pipeline (~15h)
- D3 DemoDataset (JSONL, HDF5 first; LeRobot/Robomimic in 2.6).
- D10 reward component logging.
- `tests/test_dataset.py`: load+iterate JSONL demo, batch shapes correct.

### Phase 2.3 — BC trainer (~15h)
- D4 Trainer base + D5 BC loss + D7 TrainablePolicyBase + D8 callbacks (wandb, tb, checkpoint, eval).
- `examples/train_bc_pick_place.py` end-to-end: record 50 demos → train BC → eval ≥50% success.
- `tests/test_bc_training.py`: overfit small dataset in 100 steps.

### Phase 2.4 — PPO trainer (~20h)
- D6 PPO + GAE + ActorCritic.
- D11 ReplayBuffer (for future SAC).
- `examples/train_ppo_reach.py`: PPO 16-env on simple reach task, ≥80% success in 500k steps.
- `tests/test_ppo_components.py`: GAE math, PPO clip, value loss.

### Phase 2.5 — CLI (~10h)
- D9 `robodeploy train/eval/convert-dataset` subcommands.
- Smoke tests via subprocess.

### Phase 2.6 — Dataset adapters (~5h)
- `DemoDataset.from_lerobot()` (depends on `lerobot` package).
- `DemoDataset.from_robomimic()` (uses existing robomimic).
- `DemoDataset.from_rlds()` (depends on `rlds` / `tensorflow_datasets`).

---

## Acceptance Criteria

- [x] `gym.make("robodeploy/kuka_pick_mujoco-v0")` works (`tests/training/test_gym_register.py`; MuJoCo CI in `sensor-e2e-linux`).
- [x] `stable_baselines3.PPO("MultiInputPolicy", env).learn(100k)` runs without error (`tests/training/test_sb3_smoke.py`, `@pytest.mark.slow`, `robodeploy/Tiny-v0`).
- [x] `robodeploy train bc --dataset demos.jsonl --epochs 100` writes checkpoint (`tests/training/test_bc_training.py`; wandb only with `--log wandb`, not asserted in CI).
- [x] `robodeploy train ppo --preset kuka_pick_mujoco --n-envs 16` parallel env path — **throughput honestly scoped**: ≥3× claim only for `dummy_gym_env_factory` + `fork` on Linux (`test_subproc_vec_env`, `test_ppo_preset_throughput`); CLI uses `spawn` by default; preset factory spawn smoke in `test_ppo_preset_throughput` (MuJoCo optional); documented in `docs/tutorials/03_training.md`.
- [x] `robodeploy train eval --checkpoint X.pt --episodes 100` outputs success_rate, mean_reward, time_to_success — `train eval` is dummy-only by CLI design (preset runs live in `examples.cli`); non-dummy checkpoint eval covered by `robodeploy eval --policy X.pt --backend mujoco` (`RoboEnv._coerce_policy` → `coerce_eval_policy`) with MuJoCo CI coverage in `tests/training/test_train_eval_benchmark_e2e.py::test_train_bc_then_eval_mujoco_reach_target_checkpoint` (`training-integration` job, `[sim]` extra).
- [x] BC overfit test passes (loss < 1e-4 on 10-sample dataset in 500 steps) (`tests/training/test_bc_training.py`).
- [x] PPO converges on `reach_target` toy task ≥80% success (`tests/training/test_ppo_reach_target.py`, 10k steps + small net, `@pytest.mark.slow`; full 500k in `examples/train_ppo_reach.py`).
- [x] Dataset loaders: JSONL, HDF5 (`tests/training/test_dataset.py`), LeRobot, Robomimic (`tests/test_lerobot_export.py`; RLDS local bundle in same file).
- [x] `tests/training/` adds: gym_adapter, subproc, dataset, bc, ppo, callbacks (14 modules under `tests/training/`).

## Dependencies

- `torch>=2.0` (training extra)
- `gymnasium>=0.29` (training extra)
- `stable-baselines3>=2.1` (optional, for smoke test)
- `wandb`, `tensorboard` (optional callbacks)
- `lerobot`, `robomimic`, `rlds` (optional dataset adapters)

Add `[project.optional-dependencies] training = [torch, gymnasium, wandb, tensorboard, h5py]`.

## Risks

- **Action space heterogeneity**: BC trained on JOINT_POS data won't apply to CARTESIAN_POSE policy. Mitigation: `ActionAdapter` in dataset pipeline normalizes target action space.
- **Multi-modal observation**: vision + proprio + FT mixed shapes. Mitigation: explicit `obs_keys` parameter on dataset/encoder.
- **Real-time training infeasible**: PPO with real backend = slow (1-2 Hz). Mitigation: clearly mark PPO as sim-only; document real-hw fine-tuning workflow (BC + offline RL only).
- **Determinism on multi-process**: subprocess RNG state. Mitigation: seed per-worker, log per-worker streams.

## Out of Scope

- Offline RL (CQL, IQL). Phase 2 extension.
- Distributed multi-node training. Use Ray/PyTorch DDP separately.
- Foundation-model fine-tuning (LoRA on VLA). Goal 9.
- AutoRL / HPO. External tools (Optuna).
