# RoboDeploy plugin registry

Third-party packages extend RoboDeploy via Python entry points. After `pip install`,
call `robodeploy.discover()` (or `robodeploy list-registry --discover`) to load them.

## Entry point groups

| Group | Purpose |
|-------|---------|
| `robodeploy.backends` | Simulation / hardware backends |
| `robodeploy.robots` | `RobotDescription` subclasses |
| `robodeploy.tasks` | Task implementations |
| `robodeploy.policies` | Policy implementations |
| `robodeploy.sensors` | Sensor drivers |

## Example: `plugin-robot-demo`

Shipped in this repo at `examples/plugin_robot_demo/`:

```bash
pip install -e examples/plugin_robot_demo/
robodeploy list-registry --discover
```

Registers `demo_arm` (robot) and `demo_task` (task).

## Community plugins

| Name | Description | Install | Source |
|------|-------------|---------|--------|
| *(none yet)* | Submit a PR to add your package | | |

Duplicate registry names log a warning; last import wins (document your namespace).
