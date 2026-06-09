# RoboDeploy Cookbook

Short recipes for common workflows. Each section is self-contained with commands you can copy-paste.

## Add a new robot description

1. Scaffold the package:

```bash
robodeploy scaffold robot --name myarm --dof 6 \
  --description-dir robodeploy/description
```

2. Replace the MJCF stub at `robodeploy/description/myarm/assets/mjcf/myarm.xml` (or add URDF under `assets/urdf/`).

3. Tune `home_qpos`, joint limits, and `ee_link_name` in `description.py`.

4. Register in a preset:

```bash
robodeploy scaffold preset --name myarm_pick --robot myarm --backend mujoco \
  --output examples/config/myarm_pick.yaml
```

Merge the snippet into `examples/config/presets.yaml` (include `examples/presets/base_sim.yaml`).

5. Verify:

```bash
robodeploy assets info myarm
robodeploy lint preset examples/config/presets.yaml --check myarm_pick
```

## Swap MuJoCo for Gazebo without code change

Presets separate robot, backend, task, and policy. Duplicate a sim preset and change `backend`:

```yaml
myarm_pick_gazebo:
  <<: *base_sim
  robot: myarm
  backend: gazebo
  task: pick_place
  policy: example_sensor_reach_pick
  backend_kwargs:
    world: default
```

Run:

```bash
python -m examples.cli run-episode --preset myarm_pick_gazebo
```

Sensor rigs resolve to Gazebo/ROS2 implementations automatically. Install Gazebo extras per `docs/BACKEND_SETUP.md`.

## Record demos with SpaceMouse

1. Install teleop extras: `pip install -e ".[teleop]"`.

2. Use a preset with teleop enabled or run the teleop CLI:

```bash
robodeploy teleop record --preset kuka_pick_mujoco --device spacemouse --out demos/
```

3. Replay and export:

```bash
robodeploy replay --dataset demos/latest.jsonl --preset kuka_pick_mujoco
robodeploy export-episode --dataset demos/latest.jsonl --format lerobot --out lerobot_ds/
```

See `docs/tutorials/02_teleop.md` for keyboard and gamepad variants.

## Train BC on collected demos

1. Record ≥20 episodes (keyboard or SpaceMouse).

2. Train:

```bash
robodeploy train bc \
  --dataset demos/pick_place.jsonl \
  --obs joint_pos \
  --epochs 50 \
  --log-dir runs/bc_pick
```

3. Evaluate the checkpoint:

```bash
robodeploy eval --policy runs/bc_pick/bc_final.pt --preset kuka_pick_mujoco --episodes 10
```

Or wire the checkpoint into a preset `policy_kwargs.checkpoint` field.

## Wire a custom camera-based pose estimator

1. Implement a perception module (see `examples/perception/color_blob.py`).

2. Register as an obs pipeline transform or custom sensor publishing `obs.objects`.

3. Declare vision requirement in task:

```python
def obs_spec(self) -> ObsSpec:
    return ObsSpec(rgb=True, objects=True)
```

4. Use `vision_target_in_view` success predicate instead of oracle `prop_pose`:

```python
from robodeploy.tasks.success_predicates import vision_target_in_view

def success_fn(self, obs) -> bool:
    return vision_target_in_view(obs, target="source", min_pixels=120)
```

5. Remove `prop_pose` from preset sensor rig for sim2real fidelity.

## Add a new sensor type

1. Scaffold MuJoCo implementation:

```bash
robodeploy scaffold sensor --name pressure --backend mujoco \
  --output robodeploy/sensors/pressure/sim/mujoco_pressure.py
```

2. Implement `_read_impl()` returning populated `SensorData`.

3. Add rig shorthand or reference by registered name in preset YAML.

4. Extend `SensorSampleBuffer` mapping if you introduce new observation fields.

5. Test:

```bash
robodeploy lint all
python -m pytest tests/test_resolve_sensor_class.py -q
```

## Distribute as a third-party plugin

1. Create a package with entry points in `pyproject.toml`:

```toml
[project.entry-points."robodeploy.robots"]
myarm = "my_pkg.description:MyArmDescription"

[project.entry-points."robodeploy.sensors"]
pressure_sim = "my_pkg.sensors.pressure:MuJoCoPressureSensor"
```

2. Document install: `pip install my-robodeploy-plugin`.

3. Users discover components:

```bash
robodeploy list-registry --discover
robodeploy doctor
```

See `docs/PLUGINS.md` and `examples/plugin_robot_demo/` for a minimal reference plugin.

## Define a custom task in <30 lines

```bash
robodeploy scaffold task --name kitchen_pick --template pick_place \
  --output examples/tasks/kitchen_pick.py
robodeploy lint task examples/tasks/kitchen_pick.py
```

Add `examples.tasks.kitchen_pick` to preset `custom_modules`. Customize `scene_spec()`, reward weights, and `language_instruction()`.

## Author a reach policy in YAML

```bash
robodeploy scaffold policy --name kitchen_reach --template reach_dsl \
  --output examples/policies/kitchen_reach.yaml
```

Point preset `policy_kwargs.reach_dsl_path` at the YAML file. Phases reference scene prop names (`source`, `target`).

## Validate scene and preset before runtime

```bash
robodeploy scene validate my_scene.yaml --backend mujoco
robodeploy config validate examples/config/presets.yaml
robodeploy config diff kuka_pick_mujoco kuka_ft_imu_pick_gazebo
```

## Scaffold a runnable example

```bash
robodeploy scaffold example --name my_demo --preset kuka_pick_mujoco \
  --output examples/my_demo/run.py
python examples/my_demo/run.py
```

## Multi-robot MuJoCo

Use preset `two_franka_pick_mujoco` or define `robots:` list in YAML. See `examples/multirobot/` and `robodeploy.multirobot.resolvers` for action blending.

## Sim2real transfer checklist

1. Calibrate: `robodeploy calibrate extrinsic ...` (see `docs/SIM2REAL.md`).
2. Run transfer eval: `robodeploy sim2real eval --pair kuka_pick_mujoco:kuka_pick_real`.
3. Tune domain randomization if transfer gap is large.
4. Replace `prop_pose` with real perception before deploying.

## CI-safe smoke test

No simulator required:

```bash
robodeploy run-episode --dummy --steps 5
robodeploy doctor
robodeploy lint all
```
