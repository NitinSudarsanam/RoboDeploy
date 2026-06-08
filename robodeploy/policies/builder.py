"""Fluent builder for reach trajectory policies."""

from __future__ import annotations

from typing import Any, Literal

from robodeploy.core.spaces import ActionSpace
from robodeploy.core.types import SceneSpec
from robodeploy.policies.reach_dsl import ReachTrajectoryPolicy

CarryMode = Literal["kinematic", "follow", "contact", "weld", "none"]
_VALID_CARRY = {"kinematic", "follow", "contact", "weld", "none"}


class PolicyBuilder:
    def __init__(self) -> None:
        self._spec: dict[str, Any] = {
            "action_space": "JOINT_POS",
            "action_hz": 50.0,
            "home": [0.0, -0.6, 0.0, -1.8, 0.0, 1.2, 0.0],
            "phases": [],
            "carry": {"mode": "kinematic"},
        }
        self._scene: SceneSpec | None = None
        self._description = None
        self._extra_config: dict[str, Any] = {}

    def with_action_space(self, space: ActionSpace) -> PolicyBuilder:
        self._spec["action_space"] = space.name
        return self

    def with_config(self, **kwargs: Any) -> PolicyBuilder:
        self._extra_config.update(kwargs)
        return self

    def with_home(self, home: list[float]) -> PolicyBuilder:
        self._spec["home"] = list(home)
        return self

    def with_scene(self, scene: SceneSpec) -> PolicyBuilder:
        self._scene = scene
        return self

    def add_phase(self, name: str, **kwargs: Any) -> PolicyBuilder:
        phase = {"name": name, **kwargs}
        self._spec["phases"].append(phase)
        return self

    def add_reach_phase(
        self,
        name: str,
        *,
        target: str,
        offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
        blend: float = 0.22,
        settle_threshold: float | None = None,
    ) -> PolicyBuilder:
        return self.add_phase(
            name,
            kind="reach",
            target=target,
            offset=list(offset),
            tracking_blend=blend,
            settle_threshold=settle_threshold,
        )

    def add_grasp_phase(self, *, settle_steps: int = 10, target: str = "source") -> PolicyBuilder:
        return self.add_phase(
            "grasp",
            kind="reach",
            target=target,
            offset=[0.0, 0.0, 0.015],
            settle_threshold=0.015,
            max_steps=settle_steps * 18,
        )

    def add_close_gripper(self, *, hold_steps: int = 10) -> PolicyBuilder:
        return self.add_phase("close_gripper", kind="gripper", command="close", hold_steps=hold_steps)

    def add_release_phase(self, *, target: str = "target") -> PolicyBuilder:
        del target
        return self.add_phase("open_gripper", kind="gripper", command="open", hold_steps=10)

    def add_carry(self, *, mode: CarryMode, follow_blend: float = 0.6) -> PolicyBuilder:
        if mode not in _VALID_CARRY:
            raise ValueError(f"Invalid carry mode '{mode}'. Choose from {_VALID_CARRY}.")
        self._spec["carry"] = {"mode": mode, "follow_blend": follow_blend}
        return self

    def add_hold(self, *, steps: int) -> PolicyBuilder:
        return self.add_phase("hold", kind="hold", steps=steps)

    def add_settle_home(self, *, hold_steps: int = 40) -> PolicyBuilder:
        return self.add_phase("settle_home", kind="settle", hold_steps=hold_steps)

    def build(self) -> ReachTrajectoryPolicy:
        return ReachTrajectoryPolicy(
            self._spec,
            scene=self._scene,
            description=self._description,
            config=self._extra_config or None,
        )
