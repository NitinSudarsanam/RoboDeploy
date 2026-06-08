# Goal 8 — Multi-Robot + Distribution

**Priority**: Tier 3. **Effort**: ~40h. **Touches**: ecosystem.

## Problem

Architecture supports multi-robot but no runnable examples, no executable multi-robot training. Package = 0.1.0, not on PyPI, no Docker, no plugin entry-points.

## Current State (Audit)

### Multi-robot capability
- ROS2 backend fully implements `initialize_multi`, `reset_multi`, `step_multi`, `get_obs_multi` (`backends/real/ros2/backend.py:124-307`).
- MuJoCo + IsaacSim use shims that reject N > 1 (`backends/sim/mujoco/backend.py:34-51`).
- Gazebo inherits ROS2 backend multi-robot (`backends/sim/gazebo/backend.py:26`).
- `Robot.task_action_resolver()` callback merges concurrent task actions (e.g., `average_joint_position_actions` in `examples/multiagent_configs.py:18-24`).
- `examples/multiagent_configs.py` defines four config patterns: many-robots-many-tasks, many-robots-shared-policy, one-robot-sequential, one-robot-concurrent. **All structure-only — no `main()` block runs them.**

### Distribution
- `pyproject.toml` version `0.1.0`. CLI entry-point: `robodeploy = "robodeploy.cli:main"`.
- Optional extras: `sim`, `real`, `isaacsim`, `dev`.
- Package data: URDF, MJCF, STL, DAE, OBJ, JSON, RViz, SDF (`pyproject.toml:41-53`).
- No `Dockerfile`. No `docker-compose.yml`.
- No PyPI release workflow. No CHANGELOG. No versioning policy.
- No `[project.entry-points]` for third-party plugins.
- `core/registry.py` references entry-point format in ARCHITECTURE.md but auto-discovery not implemented.

---

## Deliverables

### A. Multi-Robot Sim Support (~15h)

### D1. MuJoCo Multi-Robot — `backends/sim/mujoco/backend.py`

Replace shim. Implement proper multi-robot.

Approach: single shared MJCF with per-robot `<body name="robot_<id>">` namespaces (configured via `MultiRobotMjcfBuilder`). Single `mj_step` for all robots.

```python
class MuJoCoBackend(IBackend):
    def initialize_multi(self, robots: list[RobotInit], scene: SceneSpec, shared_sensors: list[ISensor]):
        builder = MultiRobotMjcfBuilder(scene)
        for r in robots:
            builder.add_robot(r.robot_id, r.description, base_pose=r.base_pose)
        for s in shared_sensors:
            builder.add_sensor(s)
        self._model = mujoco.MjModel.from_xml_string(builder.to_xml())
        self._data = mujoco.MjData(self._model)
        self._robot_id_to_dof_slice = builder.robot_dof_slices  # {"r0": slice(0,7), "r1": slice(7,14)}
        self._robot_id_to_ee_body = builder.ee_bodies

    def step_multi(self, actions: dict[str, Action]) -> list[Observation]:
        for rid, action in actions.items():
            dof_slice = self._robot_id_to_dof_slice[rid]
            self._data.ctrl[dof_slice] = action.joint_positions  # assuming JOINT_POS
        mujoco.mj_step(self._model, self._data)
        return [self._build_obs_for_robot(rid) for rid in self._robot_id_to_dof_slice]
```

Add `MultiRobotMjcfBuilder` extending `MjcfSceneBuilder`. Handles per-robot include + base_pose offset + actuator namespacing.

### D2. IsaacSim Multi-Robot — `backends/sim/isaacsim/backend.py`

(Already specified in Goal 6 D7.) Implement separate ArticulationView per robot.

### D3. Cross-Backend Multi-Robot API Consistency — `core/interfaces/backend.py`

Solidify `IBackend.initialize_multi` signature + lifecycle. Define `RobotInit` dataclass:

```python
@dataclass
class RobotInit:
    robot_id: str
    description: RobotDescription
    base_pose: Pose3D = field(default_factory=Pose3D)
    namespace: str | None = None  # for ROS2 topic separation
    sensor_rig: list[ISensor] = field(default_factory=list)
```

### D4. Multi-Robot Examples — `examples/multirobot/`

Concrete runnable examples:

1. `examples/multirobot/two_franka_pick_place_mujoco/run.py` — two arms hand-off cube.
2. `examples/multirobot/franka_kuka_collaborative_mujoco/run.py` — franka places into kuka tray.
3. `examples/multirobot/three_arm_assembly_mujoco/run.py` — three arms cooperative assembly.
4. `examples/multirobot/two_so101_real/run.py` — two real SO-101 arms via ROS2.

Each example: <100 lines, scaffold-style. README explains coordination pattern (independent / sequential / concurrent / shared-policy).

### D5. Task Action Resolvers — `robodeploy/multirobot/resolvers.py` (NEW, ~200 lines)

```python
def average_joint_actions(actions: list[Action]) -> Action: ...
def priority_select(actions: list[Action], *, priority_order: list[str]) -> Action: ...
def weighted_blend(actions: list[Action], weights: list[float]) -> Action: ...
def shared_workspace_safe(actions: list[Action], *, safety_zones: list[Box]) -> Action: ...
```

Register via `@register_action_resolver(name)`. Reference from `examples/multirobot/*` configs.

### D6. Multi-Robot Tests — `tests/test_multirobot_mujoco.py`, `tests/test_multirobot_isaacsim.py`

- Two robots in MuJoCo: independent reach to different targets.
- Action dispatch: per-robot actions land at correct DOF slices.
- Obs aggregation: each robot's obs has correct joint positions.
- Shared sensors: overhead camera visible to all robots.
- Resolver: averaged joint command produces expected midpoint motion.

---

### B. Distribution & Packaging (~15h)

### D7. Versioning + CHANGELOG — `CHANGELOG.md` + `robodeploy/__init__.py`

- Adopt SemVer. Bump to `0.2.0` after Goal 1+2+3 land (representation + training + sensors).
- `robodeploy.__version__` exposed.
- `CHANGELOG.md` follows [Keep a Changelog](https://keepachangelog.com/) format.
- Pre-release: `0.2.0a1`, `0.2.0b1`, `0.2.0rc1`.

### D8. PyPI Publish Workflow — `.github/workflows/publish.yml`

```yaml
name: Publish to PyPI
on:
  push:
    tags: ["v*"]
jobs:
  build-and-publish:
    runs-on: ubuntu-latest
    permissions: {id-token: write}   # trusted publishing
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.11"}
      - run: pip install build twine
      - run: python -m build
      - run: twine check dist/*
      - uses: pypa/gh-action-pypi-publish@release/v1
```

Configure trusted publishing in PyPI project settings (no API token needed).

### D9. Test on Multiple Python Versions — `.github/workflows/test.yml`

Extend matrix:
```yaml
strategy:
  matrix:
    python-version: ["3.10", "3.11", "3.12"]
    os: [ubuntu-latest, windows-latest, macos-latest]
```

`pyproject.toml`:
```toml
[project]
requires-python = ">=3.10"
classifiers = [
  "Programming Language :: Python :: 3.10",
  "Programming Language :: Python :: 3.11",
  "Programming Language :: Python :: 3.12",
  ...
]
```

### D10. Dockerfiles — `docker/`

```
docker/
├── Dockerfile.cpu               # base image, all sim backends except Isaac
├── Dockerfile.gpu               # adds CUDA + torch GPU
├── Dockerfile.isaacsim          # NVIDIA Isaac Sim base
├── Dockerfile.ros2              # ROS 2 Jazzy base + ros_gz_bridge
├── docker-compose.yml           # one-command dev env (cpu)
└── README.md
```

Each Dockerfile multi-stage: builder (compile deps) + runtime (slim).

CPU Dockerfile sketch:
```dockerfile
FROM python:3.11-slim AS builder
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender1 \
    libegl1 libosmesa6 ffmpeg git build-essential && \
    rm -rf /var/lib/apt/lists/*
COPY pyproject.toml /app/
WORKDIR /app
RUN pip install --no-cache-dir build && pip install --no-cache-dir ".[sim,dev]"

FROM python:3.11-slim
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY . /app
WORKDIR /app
CMD ["robodeploy", "--help"]
```

### D11. Conda Recipe — `conda-recipe/meta.yaml`

```yaml
package:
  name: robodeploy
  version: "{{ environ.get('GIT_DESCRIBE_TAG', '0.2.0') }}"
source:
  path: ..
requirements:
  host: [python>=3.10, pip, setuptools>=61]
  run: [python>=3.10, numpy, jax, mujoco, h5py, pyyaml]
test:
  imports: [robodeploy]
about:
  home: https://github.com/anthropic-ai/robodeploy
  license: Apache-2.0
  summary: Deploy robot policies across simulators and real hardware.
```

Build + publish via `conda-build` action.

### D12. Entry-Point Plugin Discovery — `core/registry.py`

```python
def auto_discover_plugins():
    """Scan entry points for third-party RoboDeploy extensions."""
    import importlib.metadata as md
    for group in ("robodeploy.backends", "robodeploy.robots", "robodeploy.tasks",
                  "robodeploy.policies", "robodeploy.sensors"):
        for ep in md.entry_points(group=group):
            try:
                ep.load()  # importing triggers @register_* decorators
            except Exception as exc:
                _logger.warning(f"Plugin {ep.name} failed to load: {exc}")
```

Call from `robodeploy/__init__.py` or lazy on first registry lookup.

### D13. Example Plugin Package — `examples/plugin_robot_demo/`

External-package template demonstrating third-party robot/task/policy contribution:

```
plugin_robot_demo/
├── pyproject.toml         # has [project.entry-points."robodeploy.robots"]
├── plugin_robot_demo/
│   ├── __init__.py
│   ├── robot.py           # @register_robot("demo_arm")
│   ├── task.py            # @register_task("demo_task")
│   └── description/...
└── README.md              # explains install + usage
```

`pyproject.toml`:
```toml
[project.entry-points."robodeploy.robots"]
demo_arm = "plugin_robot_demo.robot"

[project.entry-points."robodeploy.tasks"]
demo_task = "plugin_robot_demo.task"
```

User installs: `pip install plugin-robot-demo`. RoboDeploy auto-registers on next import.

### D14. RoboDeploy Hub Registry — `docs/PLUGINS.md`

Curated list of community plugins. Format: name, description, install command, GitHub link.

### D15. Versioned Asset Bundle — `robodeploy/_assets/manifest.json`

Track which URDFs/MJCFs ship in each release. Versioned so users can pin asset version separately from code:

```json
{
  "version": "0.2.0",
  "assets": [
    {"name": "franka_panda", "format": "mjcf", "path": "robodeploy/description/franka/panda.xml", "sha256": "..."},
    {"name": "franka_panda", "format": "urdf", "path": "robodeploy/description/franka/panda.urdf", "sha256": "..."},
    ...
  ]
}
```

Helps reproducibility: `robodeploy assets verify` checks SHA256.

---

## Phased Rollout

### Phase 8.1 — Multi-Robot MuJoCo (~10h)
- D1 MuJoCo multi-robot implementation.
- D3 RobotInit dataclass + interface tightening.
- D5 ActionResolvers.
- D6 multi-robot tests for MuJoCo.

### Phase 8.2 — Multi-Robot examples (~5h)
- D4 two_franka_pick_place_mujoco + three_arm_assembly_mujoco examples.
- README per example.

### Phase 8.3 — Packaging foundation (~10h)
- D7 CHANGELOG + version bump.
- D8 PyPI publish workflow + trusted publishing setup.
- D9 multi-Python + multi-OS CI matrix.
- D15 versioned asset manifest.

### Phase 8.4 — Docker + Conda (~8h)
- D10 four Dockerfiles + docker-compose.
- D11 conda recipe.
- CI builds + pushes images to ghcr.io on release tags.

### Phase 8.5 — Plugin ecosystem (~7h)
- D12 entry-point auto-discovery.
- D13 example plugin package.
- D14 PLUGINS.md.
- Test: `pip install -e examples/plugin_robot_demo/` then `robodeploy list-registry` shows `demo_arm`.

---

## Acceptance Criteria

- [ ] `two_franka_pick_place_mujoco` runs to completion: both arms reach independent targets.
- [ ] `MuJoCoBackend.initialize_multi([r1, r2], scene, sensors)` produces obs with separate joint state per robot.
- [ ] `Robot.task_action_resolver = average_joint_actions` produces midpoint command when both tasks active.
- [ ] `pip install robodeploy` from PyPI works (after first release).
- [ ] CI tests pass on Python 3.10, 3.11, 3.12 across Linux, macOS, Windows.
- [ ] `docker run robodeploy/robodeploy:cpu robodeploy run-episode --preset dummy` succeeds.
- [ ] `conda install -c robodeploy robodeploy` installs successfully (after conda-forge submission).
- [ ] Third-party plugin (`pip install plugin-robot-demo`) registers `demo_arm` automatically.
- [ ] `robodeploy assets verify` checks SHA256 of all shipped assets.
- [ ] CHANGELOG documents every public API change.
- [ ] Release tags trigger automated PyPI upload.

## Dependencies

- `build`, `twine` (PyPI tooling).
- Docker / BuildKit.
- `conda-build` (optional, for conda recipe).
- GitHub Actions secrets for ghcr.io push.

## Risks

- **MuJoCo multi-robot performance**: large N → expensive step. Mitigation: doc scalability profile, recommend Isaac for N>4.
- **Multi-robot test flakiness**: contact between arms timing-sensitive. Mitigation: fixed seed + deterministic step.
- **PyPI name squatting**: `robodeploy` package may exist. Mitigation: claim name early + use namespace `anthropic-robodeploy` if taken.
- **Plugin namespace pollution**: third-party plugin overrides built-in name. Mitigation: registry warns on duplicate registration + last-write-wins documented.
- **Docker image bloat**: Isaac image ≥10 GB. Mitigation: separate tag, pull-on-demand, document size.

## Out of Scope

- Distributed multi-machine training (Ray, Horovod). Defer to Goal 2 future.
- Cloud deployment (AWS, GCP). External tool problem.
- Snap / Flatpak packaging. Marginal value.
- Pinned dependency hashes (pip-tools). Add in Phase 8.3 if release-critical.
