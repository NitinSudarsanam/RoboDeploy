# RoboDeploy CLI Reference

Auto-generated from `robodeploy.cli._build_parser()`. Regenerate with:

```bash
python -c "from robodeploy.cli_docs import write_cli_reference; write_cli_reference()"
```

Install the package first: `pip install -e .`

## `robodeploy`

### `robodeploy assets`

### `robodeploy assets info`

- **name** — required. Asset or robot name.
- **--json** — optional (default: `False`). Print as JSON.
- **--pretty** — optional (default: `False`). Pretty-print JSON.

### `robodeploy assets list`

- **--robot** — optional (default: `False`). Robots only.
- **--mesh** — optional (default: `False`). Meshes only.
- **--mjcf** — optional (default: `False`). MJCF files only.
- **--json** — optional (default: `False`). Print as JSON.
- **--pretty** — optional (default: `False`). Pretty-print JSON.

### `robodeploy assets resolve`

- **name** — required. Asset or robot name.
- **--backend** — optional (default: `mujoco`). Target backend.
- **--json** — optional (default: `False`). Print as JSON.
- **--pretty** — optional (default: `False`). Pretty-print JSON.

### `robodeploy assets verify`

- **--json** — optional (default: `False`). Print results as JSON.
- **--pretty** — optional (default: `False`). Pretty-print JSON output.


### `robodeploy config`

### `robodeploy config diff`

- **preset_a** — required. First preset name.
- **preset_b** — required. Second preset name.
- **--presets-file** — optional. Path to presets.yaml.
- **--json** — optional (default: `False`). Print as JSON.
- **--pretty** — optional (default: `False`). Pretty-print JSON.

### `robodeploy config resolve`

- **--preset** — required. Preset name.
- **--presets-file** — optional. Path to presets.yaml.
- **--json** — optional (default: `False`). Print as JSON.
- **--pretty** — optional (default: `False`). Pretty-print JSON.

### `robodeploy config show`

- **--preset** — required. Preset name.
- **--presets-file** — optional. Path to presets.yaml.
- **--json** — optional (default: `False`). Print as JSON.
- **--pretty** — optional (default: `False`). Pretty-print JSON.

### `robodeploy config validate`

- **path** — required. Path to presets.yaml.
- **--json** — optional (default: `False`). Print as JSON.
- **--pretty** — optional (default: `False`). Pretty-print JSON.


### `robodeploy convert-dataset`

- **--from** — required. Source: path or lerobot://repo_id
- **--to** — required. Destination path (.jsonl or .hdf5).
- **--json** — optional (default: `False`). Print structured JSON result.

### `robodeploy doctor`

- **--json** — optional (default: `False`). Print structured JSON report.
- **--pretty** — optional (default: `False`). Pretty-print JSON output.

### `robodeploy dr-sweep`

- **--dummy** — optional (default: `False`). Use built-in dummy backend (required).
- **--output** — required. Output directory for sweep JSON report.
- **--seeds** — optional (default: `2`). Seeds per sweep cell.
- **--episodes** — optional (default: `2`). Episodes per seed.
- **--steps** — optional (default: `20`). Max steps per episode.
- **--json** — optional (default: `False`). Print report JSON to stdout.
- **--pretty** — optional (default: `False`). Pretty-print JSON output.

### `robodeploy eval`

- **--benchmark** — required. Benchmark id, e.g. manipulation_v1/reach_target or manipulation_v1 (full suite).
- **--policy** — optional (default: `scripted`). Registered policy name, or action mode: zero|hold|sinusoid|scripted.
- **--backend** — optional (default: `dummy`). Backend preset suffix (preset_<backend>.yaml), default dummy.
- **--episodes** — optional (default: `100`). Number of evaluation episodes.
- **--seed** — optional (default: `0`). Base seed for episode sequence.
- **--max-steps** — optional (default: `0`). Override per-episode step budget (0 = task default).
- **--output** — optional. Write JSON report to this path.
- **--benchmarks-root** — optional. Override benchmarks/ discovery path (default: repo benchmarks/ or ROBODEPLOY_BENCHMARKS_ROOT).
- **--sweep-backends** — optional (default: `False`). Run all preset_<backend>.yaml files for each task.
- **--parallel** — optional (default: `False`). Evaluate episodes in parallel threads.
- **--workers** — optional (default: `4`). Parallel worker count.
- **--json** — optional (default: `False`). Print report JSON to stdout.
- **--pretty** — optional (default: `False`). Pretty-print JSON output.

### `robodeploy export-episode`

- **--steps** — optional (default: `50`). Number of env steps to run.
- **--out** — required. Output file path.
- **--format** — optional (default: `jsonl`). Export format.
- **--dummy** — optional (default: `False`). Use built-in dummy backend/robot/task (required; preset export moved to examples.cli).
- **--action** — optional (default: `none`). Inject explicit actions instead of using policy actions.
- **--json** — optional (default: `False`). Print a structured JSON result.
- **--pretty** — optional (default: `False`). Pretty-print JSON output.

### `robodeploy lint`

### `robodeploy lint all`

- **--json** — optional (default: `False`). Print issues as JSON.

### `robodeploy lint policy`

- **path** — required. Path to policy module or YAML.
- **--json** — optional (default: `False`). Print issues as JSON.

### `robodeploy lint preset`

- **path** — required. Path to presets.yaml.
- **--check** — optional. Verify a named preset exists.
- **--json** — optional (default: `False`). Print issues as JSON.

### `robodeploy lint task`

- **path** — required. Path to task module.
- **--json** — optional (default: `False`). Print issues as JSON.


### `robodeploy list-benchmarks`

- **--benchmarks-root** — optional. Override benchmarks/ discovery path.
- **--json** — optional (default: `False`). Print as JSON.
- **--pretty** — optional (default: `False`). Pretty-print JSON output.

### `robodeploy list-registry`

- **--discover** — optional (default: `False`). Load Python entry points before listing (pip-installed extensions).
- **--custom-module** — optional (default: `[]`). Import dotted module path(s) before listing (register project components).
- **--builtins** — optional (default: `False`). Import builtin modules before listing (populates robots/tasks/policies).
- **--json** — optional (default: `False`). Print as JSON object.
- **--pretty** — optional (default: `False`). Pretty-print JSON output.

### `robodeploy logs`

### `robodeploy logs summary`

- **path** — required. Path to JSONL log or run directory.
- **--json** — optional (default: `False`). Print summary as JSON.

### `robodeploy logs tail`

- **path** — required. Path to JSONL log or run directory.
- **--interval** — optional (default: `0.5`). Poll interval seconds.
- **--max-lines** — optional (default: `0`). Stop after N new lines (0 = until interrupted).


### `robodeploy manifest`

### `robodeploy manifest show`

- **path** — required. manifest.json or run directory.
- **--json** — optional (default: `False`). Print raw manifest JSON.


### `robodeploy models`

### `robodeploy models download`

- **name** — required. Model alias (e.g. openvla-7b).
- **--json** — optional (default: `False`). Print resolved cache path as JSON.

### `robodeploy models list`

- **--json** — optional (default: `False`). Print as JSON.
- **--pretty** — optional (default: `False`). Pretty-print JSON.


### `robodeploy replay`

- **recording** — required. Path to demo JSON/JSONL.
- **--dummy** — optional (default: `False`). Use built-in dummy env.
- **--seed** — optional. Override replay seed.
- **--diff** — optional (default: `False`). Compute observation divergence report.
- **--output** — optional. Write replay/diff report JSON here.
- **--on-divergence** — optional (default: `warn`). 
- **--json** — optional (default: `False`). Print report JSON to stdout.

### `robodeploy run-episode`

- **--steps** — optional (default: `50`). Number of env steps to run.
- **--dummy** — optional (default: `False`). Use built-in dummy backend/robot/task (required; preset runs moved to examples.cli).
- **--action** — optional (default: `none`). Inject explicit actions instead of using policy actions.
- **--json** — optional (default: `False`). Print a structured JSON result.
- **--pretty** — optional (default: `False`). Pretty-print JSON output.

### `robodeploy safety`

### `robodeploy safety check`

- **--preset** — optional. Preset name from examples/config/presets.yaml.
- **--robot** — optional. Registered robot name (when --preset omitted).
- **--joint-limits** — optional. Optional YAML file with joint_position_limits / joint_velocity_limits overrides.
- **--presets-file** — optional. Path to presets.yaml.
- **--json** — optional (default: `False`). Print as JSON.
- **--pretty** — optional (default: `False`). Pretty-print JSON.

### `robodeploy safety status`

- **--json** — optional (default: `False`). Print as JSON.
- **--pretty** — optional (default: `False`). Pretty-print JSON.

### `robodeploy safety test`

- **--preset** — optional. Preset name (default: built-in dummy env).
- **--inject** — optional (default: `[]`). Injection spec, e.g. force_spike=80N, collision=arm,table, human_proximity=0.1m.
- **--steps** — optional (default: `3`). Observation checks to run.
- **--presets-file** — optional. Path to presets.yaml.
- **--json** — optional (default: `False`). Print as JSON.
- **--pretty** — optional (default: `False`). Pretty-print JSON.


### `robodeploy scaffold`

### `robodeploy scaffold policy`

- **--name** — required. Policy name.
- **--template** — optional (default: `reach_dsl`). Policy template.
- **--output** — required. Output .py or .yaml path.
- **--force** — optional (default: `False`). Overwrite existing file.

### `robodeploy scaffold preset`

- **--name** — required. Preset name.
- **--robot** — required. Registered robot name.
- **--backend** — optional (default: `mujoco`). Backend name.
- **--task** — optional (default: `pick_place`). Task name.
- **--policy** — optional (default: `example_sensor_reach_pick`). Policy name.
- **--template** — optional (default: `sim`). Base preset template (examples/presets/).
- **--output** — required. Output .yaml path.
- **--force** — optional (default: `False`). Overwrite existing file.

### `robodeploy scaffold task`

- **--name** — required. Task name.
- **--template** — optional (default: `pick_place`). Task template.
- **--output** — required. Output .py path.
- **--force** — optional (default: `False`). Overwrite existing file.

### `robodeploy scaffold robot`

- **--name** — required. Robot name (snake_case id derived automatically).
- **--dof** — optional (default: `6`). Controlled degrees of freedom.
- **--description-dir** — optional (default: `robodeploy/description`). Root description directory.
- **--force** — optional (default: `False`). Overwrite existing files.

### `robodeploy scaffold sensor`

- **--name** — required. Sensor name.
- **--backend** — optional (default: `mujoco`). Target backend (`mujoco`, `gazebo`, `isaacsim`, `real`).
- **--output** — required. Output .py path.
- **--force** — optional (default: `False`). Overwrite existing file.

### `robodeploy scaffold example`

- **--name** — required. Example name.
- **--preset** — required. Preset name from `examples/config/presets.yaml`.
- **--output** — required. Output `run.py` path.
- **--force** — optional (default: `False`). Overwrite existing file.


### `robodeploy scene`

### `robodeploy scene inspect`

- **scene** — required. Path to scene.yaml.
- **--backend** — optional. Target backend.
- **--json** — optional (default: `False`). Print as JSON.
- **--pretty** — optional (default: `False`). Pretty-print JSON.

### `robodeploy scene validate`

- **scene** — required. Path to scene.yaml.
- **--backend** — optional. Target backend.
- **--json** — optional (default: `False`). Print report as JSON.
- **--pretty** — optional (default: `False`). Pretty-print JSON.


### `robodeploy serve-policy`

- **--custom-module** — optional (default: `[]`). Import dotted module path(s) before looking up policy.
- **--policy** — required. Registered policy name, hf:<model>, or framework:checkpoint (e.g. vla_stub, hf:openvla-7b).
- **--checkpoint** — optional. Optional checkpoint path for learned policies.
- **--model-spec** — optional. Optional JSON ModelSpec file path.
- **--host** — optional (default: `0.0.0.0`). Bind host/interface.
- **--port** — optional (default: `5555`). Bind port.
- **--transport** — optional (default: `zmq`). Transport.
- **--quiet** — optional (default: `False`). Disable verbose request logging.

### `robodeploy snapshot`

### `robodeploy snapshot restore`

- **path** — required. Input .pkl path.
- **--json** — optional (default: `False`). Print snapshot metadata as JSON.

### `robodeploy snapshot save`

- **path** — required. Output .pkl path.
- **--dummy** — optional (default: `False`). Use built-in dummy env.
- **--steps** — optional (default: `5`). Steps to capture after reset.


### `robodeploy train`

### `robodeploy train bc`

- **--dataset** — required. JSONL or HDF5 demo dataset path.
- **--obs** — optional (default: `proprio`). Comma-separated observation keys.
- **--action-dim** — optional. Action dimension override.
- **--epochs** — optional (default: `50`). Training epochs.
- **--batch-size** — optional (default: `32`). Batch size.
- **--lr** — optional (default: `0.0001`). Learning rate.
- **--log-dir** — optional (default: `./runs/bc`). Checkpoint and log directory.
- **--out** — optional. Final checkpoint path (default: log-dir/bc_final.pt).
- **--dummy** — optional (default: `False`). Synthesize dummy demos if dataset is missing.
- **--json** — optional (default: `False`). Print structured JSON result.

### `robodeploy train eval`

- **--checkpoint** — required. Checkpoint .pt path.
- **--episodes** — optional (default: `10`). Evaluation episodes.
- **--dummy** — optional (default: `False`). Use built-in dummy env.
- **--json** — optional (default: `False`). Print structured JSON result.

### `robodeploy train ppo`

- **--preset** — optional. Example preset name (requires examples on path).
- **--n-envs** — optional (default: `4`). Parallel env count.
- **--total-steps** — optional (default: `10000`). Total environment steps.
- **--rollout-steps** — optional (default: `256`). Steps per rollout.
- **--lr** — optional (default: `0.0003`). Learning rate.
- **--log-dir** — optional (default: `./runs/ppo`). Checkpoint directory.
- **--log** — optional. Logging backend: wandb, tensorboard, or empty.
- **--dummy** — optional (default: `False`). Use built-in dummy env (default when no preset).
- **--json** — optional (default: `False`). Print structured JSON result.


### `robodeploy transfer-eval`

- **--dummy** — optional (default: `False`). Use built-in dummy backend (required).
- **--output** — required. Output directory for transfer report.
- **--episodes** — optional (default: `3`). Matched episodes to run.
- **--steps** — optional (default: `20`). Max steps per episode.
- **--json** — optional (default: `False`). Print metrics JSON to stdout.
- **--pretty** — optional (default: `False`). Pretty-print JSON output.
