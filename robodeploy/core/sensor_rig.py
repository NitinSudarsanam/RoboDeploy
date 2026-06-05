"""SensorRig — user-facing sensor composition (like Robot for perception)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from robodeploy.core.interfaces.sensor import ISensor
from robodeploy.core.registry import normalize_sensor_backend_name, resolve_sensor_class
from robodeploy.core.types import SensorMount

SensorKind = Literal["wrist_camera", "overhead_camera", "wrist_ft", "sim_prop_pose"]


@dataclass
class SensorSpec:
    """Declarative sensor entry resolved to a concrete ISensor at env build time."""

    kind: SensorKind
    name: str | None = None
    mount: SensorMount | None = None
    config: dict[str, Any] = field(default_factory=dict)

    def logical_name(self) -> str:
        if self.name:
            return str(self.name)
        return str(self.kind)


@dataclass
class SensorRig:
    """A named bundle of sensors attached to one robot or shared across the scene."""

    rig_id: str
    specs: list[SensorSpec] = field(default_factory=list)

    @classmethod
    def robot_mounted(
        cls,
        rig_id: str = "arm_sensors",
        *,
        wrist_rgbd: dict[str, Any] | None = None,
        overhead_rgbd: dict[str, Any] | None = None,
        wrist_ft: dict[str, Any] | None = None,
        prop_pose: dict[str, Any] | None = None,
        ee_link: str = "robot0/ee_link",
    ) -> SensorRig:
        """Build a common manipulation sensor rig from shorthand kwargs."""
        specs: list[SensorSpec] = []
        if wrist_rgbd is not None:
            cfg = dict(wrist_rgbd)
            mount = cfg.pop("mount", None)
            if mount is None and ee_link:
                mount = SensorMount(
                    parent_link=str(cfg.pop("parent_link", ee_link)),
                    position=tuple(cfg.pop("position", (0.0, 0.0, 0.05))),
                )
            specs.append(SensorSpec(kind="wrist_camera", name="wrist_camera", mount=mount, config=cfg))
        if overhead_rgbd is not None:
            cfg = dict(overhead_rgbd)
            mount = cfg.pop("mount", None)
            specs.append(
                SensorSpec(
                    kind="overhead_camera",
                    name=str(cfg.pop("name", "overhead_camera")),
                    mount=mount,
                    config=cfg,
                )
            )
        if wrist_ft is not None:
            cfg = dict(wrist_ft)
            mount = cfg.pop("mount", None)
            if mount is None and ee_link:
                mount = SensorMount(parent_link=str(cfg.pop("parent_link", ee_link)))
            specs.append(SensorSpec(kind="wrist_ft", name="wrist_ft", mount=mount, config=cfg))
        if prop_pose is not None:
            cfg = dict(prop_pose)
            specs.append(SensorSpec(kind="sim_prop_pose", name="prop_pose", config=cfg))
        return cls(rig_id=rig_id, specs=specs)

    def materialize(
        self,
        *,
        is_real: bool,
        backend_name: str | None = None,
    ) -> list[ISensor]:
        """Resolve specs to concrete sensor instances for the active backend."""
        out: list[ISensor] = []
        seen: set[str] = set()
        for spec in self.specs:
            logical = spec.logical_name()
            if logical in seen:
                raise ValueError(f"Duplicate sensor name '{logical}' in rig '{self.rig_id}'.")
            seen.add(logical)
            registry_name = spec.kind
            if spec.kind == "sim_prop_pose":
                registry_name = "sim_prop_pose"
            SensorClass = resolve_sensor_class(registry_name, is_real=is_real, backend_name=backend_name)
            cfg = _apply_backend_sensor_defaults(
                spec.kind,
                {**spec.config, "name": logical},
                backend_name=backend_name,
            )
            if spec.mount is not None:
                cfg.setdefault("mount", {
                    "parent_link": spec.mount.parent_link,
                    "position": spec.mount.position,
                    "orientation": spec.mount.orientation,
                })
            try:
                out.append(SensorClass(config=cfg))
            except TypeError:
                out.append(SensorClass(logical, config=cfg))
        return out


def _apply_backend_sensor_defaults(
    kind: SensorKind,
    cfg: dict[str, Any],
    *,
    backend_name: str | None,
) -> dict[str, Any]:
    """Fill ROS2/Gazebo topic defaults for logical rig sensors."""
    backend = normalize_sensor_backend_name(backend_name)
    if backend not in ("gazebo", "ros2", "ros2_rviz"):
        return cfg
    logical = str(cfg.get("name", kind))
    if kind in ("wrist_camera", "overhead_camera"):
        cfg.setdefault("namespace", f"/{logical}")
        cfg.setdefault("rgb", "image_raw")
        cfg.setdefault("depth", "depth/image_raw")
        cfg.setdefault("info", "camera_info")
    if kind == "wrist_ft":
        cfg.setdefault("namespace", f"/{logical}")
        cfg.setdefault("wrench_topic", "wrench")
    return cfg


def materialize_sensor_rigs(
    rigs: list[SensorRig],
    *,
    is_real: bool,
    backend_name: str | None = None,
) -> list[ISensor]:
    sensors: list[ISensor] = []
    for rig in rigs:
        sensors.extend(rig.materialize(is_real=is_real, backend_name=backend_name))
    return sensors
