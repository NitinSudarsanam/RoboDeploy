"""Scaffold new tasks, policies, and presets from templates."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

TEMPLATE_VERSION = "0.2.0"

_TASK_PICK_PLACE = '''\
"""{class_name} — pick-and-place task scaffolded by robodeploy."""
# template_version: {template_version}

from __future__ import annotations

from robodeploy.core.registry import register_task
from robodeploy.core.types import ObsSpec
from robodeploy.scene_builder import SceneBuilder
from robodeploy.tasks.base import TaskBase


@register_task("{task_id}")
class {class_name}(TaskBase):
    """TODO: describe your pick-and-place task."""

    source_name = "source"
    target_name = "target"

    def obs_spec(self) -> ObsSpec:
        if self.config.get("require_objects"):
            return ObsSpec(rgb=False, depth=False, objects=True)
        return ObsSpec(rgb=False, depth=False)

    def scene_spec(self):
        return (
            SceneBuilder()
            .add_table(height=0.4)
            .add_box("source", size=(0.03, 0.03, 0.03), pos=(0.55, 0.0, 0.41), mass=0.08)
            .add_target("target", pos=(0.65, 0.2, 0.41))
            .build_spec()
        )

    def language_instruction(self) -> str:
        return "TODO: describe the goal in natural language."

    def reset_fn(self, backend) -> None:
        self._bind_backend(backend)
        self._apply_domain_randomization(backend)

    def reward_fn(self, obs, action):
        # TODO: customize reward — see docs/TASK_CREATION.md
        del action
        source_pose = self.object_pose(self.source_name, obs)
        if source_pose is None:
            return -1.0
        source_pos, _ = source_pose
        target = self.scene_prop(self.target_name)
        goal = target.position if target else (0.65, 0.2, 0.41)
        dist = sum((a - b) ** 2 for a, b in zip(source_pos, goal)) ** 0.5
        return -dist

    def success_fn(self, obs) -> bool:
        source_pose = self.object_pose(self.source_name, obs)
        if source_pose is None:
            return False
        source_pos, _ = source_pose
        target = self.scene_prop(self.target_name)
        goal = target.position if target else (0.65, 0.2, 0.41)
        dist = sum((a - b) ** 2 for a, b in zip(source_pos, goal)) ** 0.5
        return dist < 0.04
'''

_TASK_CUSTOM = '''\
"""{class_name} — custom task scaffolded by robodeploy."""
# template_version: {template_version}

from __future__ import annotations

from robodeploy.core.registry import register_task
from robodeploy.core.types import Action, ObsSpec, Observation, SceneSpec
from robodeploy.scene_builder import SceneBuilder
from robodeploy.tasks.base import TaskBase


@register_task("{task_id}")
class {class_name}(TaskBase):
    """TODO: describe your task."""

    def obs_spec(self) -> ObsSpec:
        return ObsSpec(rgb=False, depth=False)

    def scene_spec(self) -> SceneSpec:
        return SceneBuilder().add_table(height=0.4).build_spec()

    def language_instruction(self) -> str:
        return "TODO: task goal."

    def reset_fn(self, backend) -> None:
        self._bind_backend(backend)

    def reward_fn(self, obs: Observation, action: Action) -> float:
        del obs, action
        return 0.0

    def success_fn(self, obs: Observation) -> bool:
        del obs
        return False
'''

_POLICY_REACH_DSL = '''\
# Reach trajectory DSL for {policy_id}
# template_version: {template_version}
# Load with ReachTrajectoryPolicy or examples.policies.reach_pick_place

{policy_id}:
  home: [0.0, -0.6, 0.0, -1.8, 0.0, 1.2, 0.0]
  action_hz: 50.0
  carry_mode: kinematic
  phases:
    - name: pregrasp
      target_frame: source
      offset: [0.0, 0.0, 0.10]
      tracking_blend: 0.22
      settle_threshold: 0.025
    - name: grasp
      offset: [0.0, 0.0, 0.015]
    - name: lift
      offset: [0.0, 0.0, 0.14]
    - name: transit
      target_frame: target
      offset: [0.0, 0.0, 0.10]
    - name: place
      offset: [0.0, 0.0, 0.02]
'''

_POLICY_CUSTOM = '''\
"""{class_name} — custom policy scaffolded by robodeploy."""
# template_version: {template_version}

from __future__ import annotations

from robodeploy.core.registry import register_policy
from robodeploy.core.spaces import ActionSpace
from robodeploy.core.types import Action, Observation
from robodeploy.policies.base import PolicyBase


@register_policy("{policy_id}")
class {class_name}(PolicyBase):
    """TODO: describe your policy."""

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(action_space=ActionSpace.JOINT_POS, config=config)

    def get_action(self, obs: Observation) -> Action:
        del obs
        # TODO: implement policy logic — see docs/POLICY_CREATION.md
        import numpy as np

        return Action(joint_positions=np.zeros(7, dtype=np.float32))
'''


def _to_class_name(name: str) -> str:
    parts = re.split(r"[_\-\s]+", name.strip())
    return "".join(p[:1].upper() + p[1:] for p in parts if p)


def _to_snake(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_]+", "_", name.strip())
    s = re.sub(r"_+", "_", s).strip("_").lower()
    return s or "unnamed"


def scaffold_task(
    *,
    name: str,
    template: Literal["pick_place", "custom"] = "pick_place",
    output: Path | str,
    force: bool = False,
) -> Path:
    task_id = _to_snake(name)
    class_name = _to_class_name(name)
    tmpl = _TASK_PICK_PLACE if template == "pick_place" else _TASK_CUSTOM
    content = tmpl.format(
        class_name=class_name,
        task_id=task_id,
        template_version=TEMPLATE_VERSION,
    )
    return _write_file(output, content, force=force)


_PRESET_SNIPPET_SIM = '''\
# Preset scaffolded by robodeploy — add to examples/config/presets.yaml
# template_version: {template_version}
# Ensure presets.yaml includes examples/presets/base_sim.yaml

{name}:
  <<: *base_sim
  robot: {robot}
  backend: {backend}
  task: {task}
  policy: {policy}
'''

_PRESET_SNIPPET_REAL = '''\
# Preset scaffolded by robodeploy — add to examples/config/presets.yaml
# template_version: {template_version}
# Ensure presets.yaml includes examples/presets/base_real.yaml

{name}:
  <<: *base_real
  robot: {robot}
  backend: {backend}
  task: {task}
  policy: {policy}
'''

_PRESET_SNIPPET_MANIPULATE = '''\
# Preset scaffolded by robodeploy — add to examples/config/presets.yaml
# template_version: {template_version}
# Ensure presets.yaml includes base_sim.yaml and manipulate.yaml

_{name}_bundle: &{name}_bundle
  <<: *base_kuka
  <<: *manipulate_pick
  <<: *arm_sensors
  <<: *manipulate_modules
  robot: {robot}
  backend: {backend}
  task: {task}
  policy: {policy}

{name}:
  <<: *{name}_bundle
'''


def scaffold_preset(
    *,
    name: str,
    robot: str,
    backend: str,
    task: str,
    policy: str,
    template: Literal["sim", "real", "manipulate"] = "sim",
    output: Path | str,
    force: bool = False,
) -> Path:
    preset_name = _to_snake(name)
    if template == "real":
        content = _PRESET_SNIPPET_REAL.format(
            name=preset_name,
            robot=robot,
            backend=backend,
            task=task,
            policy=policy,
            template_version=TEMPLATE_VERSION,
        )
    elif template == "manipulate":
        content = _PRESET_SNIPPET_MANIPULATE.format(
            name=preset_name,
            robot=robot,
            backend=backend,
            task=task,
            policy=policy,
            template_version=TEMPLATE_VERSION,
        )
    else:
        content = _PRESET_SNIPPET_SIM.format(
            name=preset_name,
            robot=robot,
            backend=backend,
            task=task,
            policy=policy,
            template_version=TEMPLATE_VERSION,
        )
    return _write_file(output, content, force=force)


def scaffold_policy(
    *,
    name: str,
    template: Literal["reach_dsl", "custom"] = "reach_dsl",
    output: Path | str,
    force: bool = False,
) -> Path:
    policy_id = _to_snake(name)
    class_name = _to_class_name(name)
    if template == "reach_dsl":
        content = _POLICY_REACH_DSL.format(
            policy_id=policy_id,
            template_version=TEMPLATE_VERSION,
        )
    else:
        content = _POLICY_CUSTOM.format(
            class_name=class_name,
            policy_id=policy_id,
            template_version=TEMPLATE_VERSION,
        )
    return _write_file(output, content, force=force)


_ROBOT_DESCRIPTION = '''\
"""
{class_name} — robot description scaffolded by robodeploy.
# template_version: {template_version}
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from robodeploy.core.registry import register_robot
from robodeploy.core.spaces import AssetFormat
from robodeploy.description.base import RobotDescription


@register_robot("{robot_id}")
class {class_name}(RobotDescription):
    dof = {dof}
    display_name = "{display_name}"
    ee_link_name = "robot0/ee_link"
    ros2_preset_name = "{robot_id}_jtc"

    joint_names = [f"robot0/joint{{i}}" for i in range(1, {dof_plus_one})]

    joint_position_limits = np.array([[-3.14, 3.14]] * dof, dtype=np.float64)
    joint_velocity_limits = np.array([2.0] * dof, dtype=np.float64)
    joint_torque_limits = np.array([50.0] * dof, dtype=np.float64)

    home_qpos = np.zeros(dof, dtype=np.float64)

    def asset_path(self, fmt: AssetFormat, variant: str = "default") -> Path:
        del variant
        assets = Path(__file__).parent / "assets"
        if fmt == AssetFormat.MJCF:
            path = assets / "mjcf" / "{robot_id}.xml"
            if not path.exists():
                raise FileNotFoundError(
                    f"{{self.display_name}} MJCF not found at {{path}}. "
                    "Add a model under assets/mjcf/."
                )
            return path
        if fmt == AssetFormat.URDF:
            path = assets / "urdf" / "{robot_id}.urdf"
            if not path.exists():
                raise FileNotFoundError(
                    f"{{self.display_name}} URDF not found at {{path}}. "
                    "Add a model under assets/urdf/."
                )
            return path
        raise FileNotFoundError(f"{{self.display_name}} does not provide {{fmt.value}} yet.")
'''

_ROBOT_INIT = '''\
"""{class_name} robot description."""
# template_version: {template_version}

from robodeploy.description.{robot_id}.description import {class_name}

__all__ = ["{class_name}"]
'''

_ROBOT_MJCF_STUB = '''\
<!-- {robot_id} MJCF stub — replace with your robot model -->
<mujoco model="{robot_id}">
  <worldbody>
    <body name="robot0/base" pos="0 0 0">
      <joint name="robot0/joint1" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.3" size="0.04"/>
      <body name="robot0/ee_link" pos="0 0 0.3"/>
    </body>
  </worldbody>
</mujoco>
'''

_SENSOR_MUJOCO = '''\
"""MuJoCo {sensor_id} sensor — scaffolded by robodeploy."""
# template_version: {template_version}

from __future__ import annotations

import time

from robodeploy.core.registry import register_sensor, register_sensor_pair
from robodeploy.core.types import SensorData, SensorMount
from robodeploy.sensors.base import SensorBase


@register_sensor("{sensor_id}_sim")
class MuJoCo{class_name}(SensorBase):
    """TODO: implement read logic for {sensor_id}."""

    def __init__(
        self,
        name: str | dict | None = None,
        *,
        config: dict | None = None,
        mount: SensorMount | None = None,
    ) -> None:
        if isinstance(name, dict) and config is None:
            cfg = dict(name)
            sensor_name = str(cfg.get("name", "{sensor_id}"))
        else:
            cfg = dict(config or {{}})
            sensor_name = str(name or cfg.get("name", "{sensor_id}"))
        if mount is None and isinstance(cfg.get("mount"), dict):
            mount = SensorMount(**cfg["mount"])
        super().__init__(name=sensor_name, is_real=False, config=cfg, mount=mount)
        self._backend = None

    def _init_impl(self, backend) -> None:
        self._backend = backend

    def _read_impl(self) -> SensorData:
        assert self._backend is not None
        ts = time.monotonic()
        # TODO: read from backend and populate SensorData fields
        return SensorData(timestamp=ts, timestamp_hw=ts, timestamp_recv=ts, timestamp_source="sim")

    def _close_impl(self) -> None:
        self._backend = None


register_sensor_pair("{sensor_id}_sim", backend="mujoco", cls=MuJoCo{class_name})
'''

_SENSOR_INIT = '''\
"""{sensor_id} sensor package."""
# template_version: {template_version}
'''

_EXAMPLE_RUN = '''\
"""Run {example_id} via preset {preset}.

Scaffolded by robodeploy. Requires repo on PYTHONPATH (or pip install -e .).
"""

from __future__ import annotations

import sys
import time

from examples._bootstrap import ensure_repo_on_path

ensure_repo_on_path()

from examples.env_from_preset import env_from_preset  # noqa: E402


def main() -> None:
    try:
        env = env_from_preset("{preset}", max_episode_steps=500)
    except ImportError as exc:
        print(exc)
        print('\\nInstall simulator extras, e.g.:\\n  pip install -e ".[sim]"')
        sys.exit(1)

    try:
        obs, info = env.reset()
        print("reset episode", info.episode_id)
        for i in range(500):
            obs, reward, done, info = env.step()
            if i % 50 == 0:
                print(f"step {{i:4d}} reward={{reward:7.3f}} success={{info.success}}")
            if done:
                print("done at step", i, "success=", info.success)
                break
            time.sleep(0.003)
    finally:
        env.close()


if __name__ == "__main__":
    main()
'''


def scaffold_robot(
    *,
    name: str,
    dof: int = 6,
    description_dir: Path | str,
    force: bool = False,
) -> Path:
    """Scaffold a robot description package under description_dir."""
    robot_id = _to_snake(name)
    class_name = _to_class_name(name)
    root = Path(description_dir)
    if root.exists() and any(root.iterdir()) and not force:
        raise FileExistsError(f"Description dir not empty: {root} (pass --force to overwrite)")

    pkg = root / robot_id
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "assets" / "mjcf").mkdir(parents=True, exist_ok=True)
    (pkg / "assets" / "urdf").mkdir(parents=True, exist_ok=True)

    desc_path = pkg / "description.py"
    if desc_path.exists() and not force:
        raise FileExistsError(f"Output exists: {desc_path}")
    desc_path.write_text(
        _ROBOT_DESCRIPTION.format(
            class_name=class_name,
            robot_id=robot_id,
            dof=dof,
            dof_plus_one=dof + 1,
            display_name=name,
            template_version=TEMPLATE_VERSION,
        ),
        encoding="utf-8",
    )

    init_path = pkg / "__init__.py"
    if not init_path.exists() or force:
        init_path.write_text(
            _ROBOT_INIT.format(
                class_name=class_name,
                robot_id=robot_id,
                template_version=TEMPLATE_VERSION,
            ),
            encoding="utf-8",
        )

    mjcf_path = pkg / "assets" / "mjcf" / f"{robot_id}.xml"
    if not mjcf_path.exists() or force:
        mjcf_path.write_text(
            _ROBOT_MJCF_STUB.format(robot_id=robot_id),
            encoding="utf-8",
        )

    return pkg


def scaffold_sensor(
    *,
    name: str,
    backend: Literal["mujoco", "gazebo", "isaacsim", "real"] = "mujoco",
    output: Path | str,
    force: bool = False,
) -> Path:
    """Scaffold a sensor implementation module."""
    sensor_id = _to_snake(name)
    class_name = _to_class_name(name)
    out = Path(output)

    if backend != "mujoco":
        content = (
            f'"""{sensor_id} sensor for {backend} — scaffolded by robodeploy."""\n'
            f"# template_version: {TEMPLATE_VERSION}\n\n"
            f"# TODO: implement {backend} driver for {sensor_id}\n"
            f"raise NotImplementedError('{backend} backend not scaffolded yet')\n"
        )
    else:
        content = _SENSOR_MUJOCO.format(
            sensor_id=sensor_id,
            class_name=class_name,
            template_version=TEMPLATE_VERSION,
        )

    return _write_file(out, content, force=force)


def scaffold_example(
    *,
    name: str,
    preset: str,
    output: Path | str,
    force: bool = False,
) -> Path:
    """Scaffold a preset-based example runner script."""
    example_id = _to_snake(name)
    content = _EXAMPLE_RUN.format(
        example_id=example_id,
        preset=preset,
    )
    return _write_file(output, content, force=force)


def _write_file(path: Path | str, content: str, *, force: bool) -> Path:
    out = Path(path)
    if out.exists() and not force:
        raise FileExistsError(f"Output exists: {out} (pass --force to overwrite)")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content, encoding="utf-8")
    return out
