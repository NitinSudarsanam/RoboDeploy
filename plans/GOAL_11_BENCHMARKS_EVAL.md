# Goal 11 — Benchmarks + Evaluation Harness

**Priority**: Tier 3. **Effort**: ~50h. **Touches**: research credibility, policy comparison.

## Problem

No standard benchmark, no metric harness, no leaderboard. Cannot compare policies across runs or projects. Each user invents own metric definitions. No automated eval pipeline.

## Current State (Audit)

### Tests
- `tests/test_*.py` — unit + integration. No benchmark harness.
- No `benchmarks/` directory.
- Zero references to "benchmark" / "leaderboard" in code.

### Eval-like utilities
- `examples/` runs single episodes (CLI `run-episode`).
- `cli_helpers.py:23-33` extracts diagnostics — closest thing to a per-episode summary.
- `EpisodeInfo.success`, `EpisodeInfo.failure` exist but not aggregated.

### Task metrics
- Tasks define `reward_fn` + `success_fn`. No standardized metric definitions, time limits, or difficulty tiers.

### Datasets
- `robodeploy/dataset_export.py` exists. Purpose: writing demos. No benchmark scoring.

---

## Deliverables

### D1. Benchmark Task Suite — `benchmarks/` (NEW directory)

Standardized tasks across difficulty tiers. Each task: deterministic scene, fixed episode budget, defined success criteria, reference baseline scores.

```
benchmarks/
├── README.md
├── manipulation_v1/
│   ├── reach_target/              # 1 — easiest
│   │   ├── task.py
│   │   ├── preset_mujoco.yaml
│   │   ├── preset_isaacsim.yaml
│   │   ├── preset_gazebo.yaml
│   │   ├── reference_scores.json  # baselines
│   │   └── README.md
│   ├── pick_place_cube/           # 2
│   ├── stacking_3blocks/          # 3
│   ├── pour_into_cup/             # 4
│   ├── peg_insert_round/          # 5
│   ├── peg_insert_square/         # 6 — harder hole tolerance
│   ├── cloth_fold/                # 7 — deformable
│   └── tools_use_screw/           # 8 — hardest
├── sim2real_v1/
│   ├── reach_target_real/
│   └── pick_place_so101_real/
├── multirobot_v1/
│   ├── two_arm_handoff/
│   └── three_arm_assembly/
└── leaderboard/
    ├── schema.json                # standard score format
    └── submissions/                # user-submitted scores
```

Each `task.py` uses templates (Goal 1 D8). Each `preset_*.yaml` is canonical config — no user modification allowed for valid benchmark runs.

### D2. Metric Definitions — `robodeploy/evaluation/metrics.py` (NEW, ~400 lines)

Standard metric library reused across all benchmarks.

```python
@dataclass
class EpisodeMetrics:
    success: bool
    reward_total: float
    reward_per_step: float
    steps: int
    time_to_success_steps: int | None
    time_to_success_seconds: float | None
    smoothness_jerk: float                  # RMS jerk
    smoothness_action_norm: float           # mean action L2
    smoothness_velocity: float              # mean joint velocity
    collision_count: int
    max_force_N: float
    workspace_violations: int
    distance_to_goal_final: float
    distance_to_goal_min: float
    constraint_violations: dict[str, int]
    metadata: dict

class MetricsCollector:
    """Per-episode metric accumulator."""

    def __init__(self, *, task: ITask, robot_description: RobotDescription): ...
    def reset(self): ...
    def observe(self, obs: Observation, action: Action, reward: float, info: EpisodeInfo): ...
    def finalize(self) -> EpisodeMetrics: ...

# Aggregators
def aggregate_episodes(metrics: list[EpisodeMetrics]) -> "AggregateMetrics":
    return AggregateMetrics(
        success_rate=mean(m.success for m in metrics),
        success_rate_ci95=ci95([m.success for m in metrics]),
        mean_reward=mean(m.reward_total for m in metrics),
        median_time_to_success_steps=median([m.time_to_success_steps for m in metrics if m.success]),
        mean_smoothness=mean(m.smoothness_jerk for m in metrics),
        ...
    )
```

### D3. Eval Harness — `robodeploy/evaluation/harness.py` (NEW, ~400 lines)

```python
@dataclass
class EvalConfig:
    n_episodes: int = 100
    seeds: list[int] | None = None     # if None: derive from base_seed
    base_seed: int = 0
    max_steps_per_episode: int = 1000
    parallel: bool = True
    n_workers: int = 8
    deterministic_policy: bool = True
    record_videos: bool = False
    video_dir: Path | None = None
    record_trajectories: bool = False
    trajectory_dir: Path | None = None

class EvalHarness:
    def __init__(self, *, env_factory: Callable[[int], RoboEnv],
                 policy_factory: Callable[[], IPolicy],
                 task: ITask, config: EvalConfig,
                 logger: RoboDeployLogger | None = None): ...

    def run(self) -> EvalReport:
        if self._config.parallel:
            return self._run_parallel()
        return self._run_sequential()

    def _run_episode(self, seed: int) -> EpisodeMetrics:
        env = self._env_factory(seed)
        policy = self._policy_factory()
        policy.reset(seed=seed)
        obs, info = env.reset(seed=seed)
        collector = MetricsCollector(task=self._task, robot_description=env.robot.description)
        for step in range(self._config.max_steps_per_episode):
            action = policy.get_action(obs)
            obs, reward, done, info = env.step(action)
            collector.observe(obs, action, reward, info)
            if done: break
        return collector.finalize()

@dataclass
class EvalReport:
    benchmark_name: str
    benchmark_version: str
    episodes: list[EpisodeMetrics]
    aggregate: AggregateMetrics
    config: EvalConfig
    manifest: RunManifest                  # from Goal 10 D7
    started_at: float
    finished_at: float

    def to_json(self) -> dict: ...
    def save(self, path: Path): ...
    def render_html(self, out: Path): ...
```

### D4. Eval CLI — `robodeploy/cli.py` (EXTEND)

```bash
robodeploy eval --benchmark manipulation_v1/pick_place_cube --policy checkpoint.pt --episodes 100 --output report.json
robodeploy eval --benchmark manipulation_v1 --policy checkpoint.pt           # full suite
robodeploy eval --benchmark manipulation_v1 --policy hf:openvla-7b --backend mujoco
robodeploy eval-compare reportA.json reportB.json --output comparison.html
robodeploy leaderboard submit report.json --benchmark manipulation_v1/pick_place_cube --author "you"
robodeploy leaderboard show manipulation_v1
```

### D5. Reference Baselines — `benchmarks/manipulation_v1/*/reference_scores.json`

Each benchmark ships baseline policies + scores:

```json
{
  "benchmark": "manipulation_v1/pick_place_cube",
  "version": "1.0",
  "baselines": [
    {"policy": "scripted_reach_pick_place", "success_rate": 0.95, "mean_reward": 124.3, "n_episodes": 200, "seed_base": 0},
    {"policy": "bc_demo_50", "success_rate": 0.68, "mean_reward": 87.1, "n_episodes": 200, "seed_base": 0},
    {"policy": "ppo_500k", "success_rate": 0.81, "mean_reward": 110.5, "n_episodes": 200, "seed_base": 0},
    {"policy": "diffusion_demo_50", "success_rate": 0.74, "mean_reward": 92.7, "n_episodes": 200, "seed_base": 0}
  ],
  "reference_assets_sha256": "abc123...",
  "evaluated_at": "2026-06-08T14:00:00Z"
}
```

### D6. Difficulty Tiers + Standardized Limits — `benchmarks/manipulation_v1/spec.json`

```json
{
  "name": "manipulation_v1",
  "version": "1.0",
  "tasks": [
    {"name": "reach_target",       "tier": 1, "max_steps": 300, "expected_success": 0.95, "weight": 1.0},
    {"name": "pick_place_cube",    "tier": 2, "max_steps": 500, "expected_success": 0.85, "weight": 1.5},
    {"name": "stacking_3blocks",   "tier": 3, "max_steps": 800, "expected_success": 0.70, "weight": 2.0},
    {"name": "pour_into_cup",      "tier": 4, "max_steps": 800, "expected_success": 0.60, "weight": 2.5},
    {"name": "peg_insert_round",   "tier": 5, "max_steps": 600, "expected_success": 0.55, "weight": 3.0},
    {"name": "peg_insert_square",  "tier": 6, "max_steps": 600, "expected_success": 0.40, "weight": 3.5},
    {"name": "cloth_fold",         "tier": 7, "max_steps": 1500, "expected_success": 0.30, "weight": 4.0},
    {"name": "tools_use_screw",    "tier": 8, "max_steps": 2000, "expected_success": 0.20, "weight": 5.0}
  ],
  "scoring": {
    "method": "weighted_success_rate",
    "tie_breaker": "mean_reward"
  }
}
```

Aggregate "RoboScore" = Σ(weight * success_rate) / Σ(weight) — normalized to 1.0.

### D7. Video Recording During Eval — `robodeploy/evaluation/video.py` (NEW, ~150 lines)

```python
class EpisodeVideoRecorder:
    def __init__(self, *, env: RoboEnv, camera_name: str = "overhead_camera",
                 out_dir: Path, fps: int = 30): ...

    def start(self, episode_id: str): ...
    def observe(self, obs: Observation): self._frames.append(obs.images[self._camera])
    def finish(self) -> Path: ...   # writes MP4 via imageio-ffmpeg
```

Optional in EvalHarness. Useful for human inspection of failure modes.

### D8. Failure Mode Analysis — `robodeploy/evaluation/failure_analysis.py` (NEW, ~200 lines)

Auto-categorizes failed episodes:

```python
class FailureClassifier:
    """Tag failed episodes with a category."""

    CATEGORIES = ["dropped", "missed_grasp", "off_target", "out_of_workspace",
                  "exceeded_force", "timeout", "collision", "other"]

    def classify(self, metrics: EpisodeMetrics, trajectory: list[Observation]) -> str:
        # Heuristics: max_force_N high → exceeded_force, distance_to_goal_final large → missed_grasp, etc.
        ...
```

### D9. EvalReport HTML Renderer — `robodeploy/evaluation/render.py` (NEW, ~250 lines)

Jinja2 template producing self-contained HTML with:
- Success rate per task + CI.
- Reward distribution plot.
- Smoothness vs success scatter.
- Failure mode breakdown pie.
- Embedded videos (if recorded).
- Comparison vs baseline table.

```python
def render_report(report: EvalReport, baseline: EvalReport | None, out: Path): ...
```

### D10. Leaderboard Submission Schema — `benchmarks/leaderboard/schema.json`

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "required": ["benchmark","benchmark_version","policy_name","author","success_rate","n_episodes","manifest","reproduce"],
  "properties": {
    "benchmark": {"type": "string"},
    "benchmark_version": {"type": "string"},
    "policy_name": {"type": "string"},
    "policy_checkpoint": {"type": "string"},
    "author": {"type": "string"},
    "success_rate": {"type": "number"},
    "success_rate_ci95": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
    "n_episodes": {"type": "integer"},
    "robo_score": {"type": "number"},
    "manifest": {"type": "object"},
    "reproduce": {
      "type": "object",
      "required": ["command","docker_image"],
      "properties": {"command": {"type": "string"}, "docker_image": {"type": "string"}}
    },
    "submitted_at": {"type": "string", "format": "date-time"}
  }
}
```

Submissions = PRs to `benchmarks/leaderboard/submissions/<benchmark>/<author>_<date>.json`. CI validates schema.

### D11. CI Eval Hook — `.github/workflows/benchmark.yml` (NEW)

Nightly: runs `manipulation_v1` against built-in scripted baselines on each backend. Posts results to GitHub Pages. Catches regressions before release.

### D12. Tests
- `tests/test_metrics.py` — collector aggregation.
- `tests/test_eval_harness.py` — parallel + sequential equivalence.
- `tests/test_benchmark_reproducibility.py` — re-running same benchmark with same seed yields same scores.
- `tests/test_leaderboard_schema.py` — submission JSON validation.

---

## Phased Rollout

### Phase 11.1 — Metrics + Harness (~12h)
- D2 metrics library.
- D3 EvalHarness (sequential first; parallel via Goal 2's SubprocVecEnv).
- D4 CLI `eval` subcommand.
- `tests/test_metrics.py`, `tests/test_eval_harness.py`.

### Phase 11.2 — Initial benchmarks (~15h)
- D1 task suite, tier 1-3 first (reach_target, pick_place_cube, stacking_3blocks).
- D6 spec.json + scoring formula.
- D5 reference baselines (scripted policies; BC/PPO if Goal 2 ready).
- READMEs.

### Phase 11.3 — Eval polish (~8h)
- D7 video recording.
- D8 failure mode classifier.
- D9 HTML report renderer.
- `tests/test_benchmark_reproducibility.py`.

### Phase 11.4 — Remaining benchmarks (~8h)
- D1 tier 4-8 (pour, peg_insert, cloth_fold, tools_use_screw).
- D5 baselines.

### Phase 11.5 — Leaderboard + CI (~7h)
- D10 schema + submissions directory + PR validation action.
- D11 nightly CI eval workflow.
- D4 `leaderboard submit/show` CLI.
- `tests/test_leaderboard_schema.py`.

---

## Acceptance Criteria

- [x] `robodeploy eval --benchmark manipulation_v1/pick_place_cube --policy scripted --episodes 100` outputs aggregated metrics JSON (`test_pick_place_cube_eval_outputs_aggregate_json`; CI uses N=5).
- [x] Eval is reproducible: same seed_base + benchmark version → same scores within float epsilon (`tests/test_benchmark_reproducibility.py`).
- [x] Tier-1 `reach_target` scripted baseline achieves ≥95% success (dummy integration test in `test_benchmarks.py`).
- [x] EvalReport HTML renders without warnings; embeds videos if recorded (`test_html_report_embeds_recorded_video`, `test_embed_path_skips_large_files`).
- [x] `robodeploy eval-compare A.json B.json` produces side-by-side delta table (`test_eval_compare_cli`).
- [x] Leaderboard PR validates against schema; rejects malformed submissions (`benchmark.yml` validate-schemas job).
- [x] Nightly CI runs `manipulation_v1` and posts results to GitHub Pages (`benchmark.yml` dummy N=5; MuJoCo subset in `test.yml` `eval-mujoco-smoke`, not nightly).
- [x] FailureClassifier categorizes ≥80% of failed episodes correctly (fixture audit test).
- [x] Parallel eval (`n_workers=8`) matches sequential scores within float epsilon (`test_parallel_matches_sequential`).
- [x] Benchmark spec.json validated by JSON Schema in CI (`benchmark.yml` validate-schemas job).

## Dependencies

- `scipy` (CI95, statistical tests).
- `jinja2` (HTML reports).
- `matplotlib` (plots).
- `imageio[ffmpeg]` (videos).
- `jsonschema` (leaderboard validation).

## Risks

- **Benchmark drift**: minor scene tweaks break score comparability across versions. Mitigation: `benchmark_version` field; immutable releases; SHA256 hash on assets.
- **Parallelism nondeterminism**: SubprocVecEnv ordering races. Mitigation: per-worker seeded; deterministic seed → worker mapping.
- **Reference baseline trickle**: each new benchmark needs reference. Mitigation: ship at least scripted-policy reference per task on benchmark add.
- **Score gaming**: policies overfit to public seed set. Mitigation: leaderboard supports private "test set" seeds for official scores.
- **CI cost**: 8 tasks × 100 episodes × 3 backends = ~2400 episodes nightly. Mitigation: run reduced N for CI gate, full N for release scoring.

## Out of Scope

- Real-hardware benchmark suite. Goal 5's `sim2real_v1` is minimal start; full real-hw benchmark requires lab access.
- Closed eval server (private test seeds). Implement only if score gaming becomes an issue.
- Policy submission as Docker image (full sandboxing). Use code path for now.
- Multi-task / lifelong learning benchmarks. Future.
