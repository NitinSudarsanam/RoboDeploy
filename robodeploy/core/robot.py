"""Robot — robot-centric aggregate replacing the old RobotConfig + TaskConfig pair.

A `Robot` owns:
  - Its description (URDF / kinematics / joint limits).
  - Its sensors, observation pipeline, and action adapter.
  - A `tasks` mapping (task_id -> RobotTask) that pairs each task with one or
    more policies and an optional policy-level selector.
  - Either `task_weights` or a custom `task_selector` to decide which
    sequential task is active each step.

`RobotEnv` no longer threads the global `tasks=` list through itself — it just
asks each robot for an action. Concurrent tasks (mode="concurrent") always run
alongside the active sequential task; multiple resulting candidate actions are
combined by an optional `task_action_resolver` (parallels the legacy
`action_resolver` pattern but lives on the robot).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional

from robodeploy.action_adapter import ActionAdapter
from robodeploy.core.interfaces.policy import IPolicy
from robodeploy.core.interfaces.sensor import ISensor
from robodeploy.core.interfaces.task import ITask
from robodeploy.core.local_arbitrator import LocalArbitrator
from robodeploy.core.selectors import (
    IPolicySelector,
    ITaskSelector,
    WeightedPolicySelector,
    WeightTaskSelector,
)
from robodeploy.core.spaces import ActionSpace, infer_action_space
from robodeploy.core.types import Action, ArbitrationEvent, MultiTaskMode, Observation
from robodeploy.description.base import RobotDescription
from robodeploy.obs_pipeline import ObsPipeline


@dataclass
class RobotTask:
    """Pairs a task with one or more policies, plus optional arbitration spec.

    If `policies` has exactly one entry, both `policy_weights` and
    `policy_selector` may be omitted. With multiple policies, supply weights or
    a selector (mutually exclusive). Internally weights wrap into a
    WeightedPolicySelector so the runtime always uses a selector.
    """

    task: ITask
    policies: dict[str, IPolicy]
    policy_weights: Optional[Mapping[str, float]] = None
    policy_selector: Optional[IPolicySelector] = None
    mode: MultiTaskMode = "sequential"
    preserve_policy_state_on_deactivate: bool = False
    task_id: str = ""

    def __post_init__(self) -> None:
        if not self.policies:
            raise ValueError(
                f"RobotTask '{self.task_id or '?'}' must have at least one policy."
            )
        if self.policy_weights is not None and self.policy_selector is not None:
            raise ValueError(
                f"RobotTask '{self.task_id or '?'}' cannot set both policy_weights and policy_selector."
            )
        if self.policy_selector is None and self.policy_weights is not None:
            self.policy_selector = WeightedPolicySelector(self.policy_weights)
        if self.policy_selector is None and len(self.policies) > 1:
            raise ValueError(
                f"RobotTask '{self.task_id or '?'}' has multiple policies but no "
                "policy_weights or policy_selector."
            )

    def language_instruction(self) -> str:
        return self.task.language_instruction()

    def action_space(self) -> ActionSpace:
        first_policy = next(iter(self.policies.values()))
        return first_policy.action_space

    def reset_policies(self) -> None:
        for policy in self.policies.values():
            policy.reset()

    def warmup_policies(self, obs: Observation) -> None:
        for policy in self.policies.values():
            try:
                policy.warmup(obs)
            except Exception as exc:
                print(f"[RobotTask:{self.task_id}] policy warmup failed: {exc}")

    def set_instruction(self) -> None:
        instr = self.task.language_instruction()
        for policy in self.policies.values():
            policy.set_instruction(instr)

    def on_activate(self) -> None:
        self.task.on_activate()

    def on_deactivate(self) -> None:
        self.task.on_deactivate()

    def compute_action(self, *, robot_id: str, obs: Observation) -> Action:
        """Run policies for this task and return a single resolved action."""
        if len(self.policies) == 1:
            policy_id, policy = next(iter(self.policies.items()))
            return policy.get_action(obs)

        candidate_actions: dict[str, Action] = {
            pid: policy.get_action(obs) for pid, policy in self.policies.items()
        }
        assert self.policy_selector is not None  # validated in __post_init__
        return self.policy_selector.select(
            robot_id=robot_id,
            task_id=self.task_id,
            obs=obs,
            candidate_actions=candidate_actions,
        )


def _default_obs_pipeline() -> ObsPipeline:
    return ObsPipeline()


def _default_action_adapter() -> ActionAdapter:
    return ActionAdapter()


@dataclass
class Robot:
    """Robot-centric aggregate. The unit `RoboEnv` orchestrates."""

    robot_id: str
    description: RobotDescription
    tasks: dict[str, RobotTask]
    task_weights: Optional[Mapping[str, float]] = None
    task_selector: Optional[ITaskSelector] = None
    sensors: list[ISensor] = field(default_factory=list)
    obs_pipeline: ObsPipeline = field(default_factory=_default_obs_pipeline)
    action_adapter: ActionAdapter = field(default_factory=_default_action_adapter)
    task_action_resolver: Optional[Callable[[str, list[Action]], Action]] = None

    _arbitrator: LocalArbitrator = field(init=False)
    _safety: object = field(init=False)

    def __post_init__(self) -> None:
        if not self.robot_id:
            raise ValueError("Robot.robot_id is required.")
        if not self.tasks:
            raise ValueError(f"Robot '{self.robot_id}' has no tasks.")
        if self.task_weights is not None and self.task_selector is not None:
            raise ValueError(
                f"Robot '{self.robot_id}' cannot set both task_weights and task_selector."
            )

        for task_id, robot_task in self.tasks.items():
            if not robot_task.task_id:
                robot_task.task_id = task_id

        sequential = [tid for tid, t in self.tasks.items() if t.mode == "sequential"]
        if not sequential:
            raise ValueError(
                f"Robot '{self.robot_id}' must have at least one sequential task."
            )

        if self.task_selector is None and self.task_weights is not None:
            self.task_selector = WeightTaskSelector(self.task_weights)
        if self.task_selector is None and len(sequential) > 1:
            raise ValueError(
                f"Robot '{self.robot_id}' has multiple sequential tasks but no "
                "task_weights or task_selector."
            )

        self._arbitrator = LocalArbitrator(
            robot_id=self.robot_id,
            tasks=self.tasks,
            task_selector=self.task_selector,
        )
        self._safety = self.description.get_safety_filter()

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self.action_adapter.reset()
        for robot_task in self.tasks.values():
            robot_task.task._on_reset()
            robot_task.reset_policies()
            robot_task.set_instruction()

    def warmup(self, obs: Observation) -> None:
        for robot_task in self.tasks.values():
            robot_task.warmup_policies(obs)

    # ------------------------------------------------------------------
    # Step path
    # ------------------------------------------------------------------

    def step(self, obs: Observation) -> Action:
        """Pick active task, query policies, adapt, safety-filter, return."""
        self._arbitrator.evaluate(obs)
        active_ids = list(self._arbitrator.active_and_concurrent())
        if not active_ids:
            raise RuntimeError(f"Robot '{self.robot_id}' has no active task.")

        candidates: list[tuple[str, Action]] = [
            (tid, self.tasks[tid].compute_action(robot_id=self.robot_id, obs=obs))
            for tid in active_ids
        ]

        if len(candidates) == 1:
            chosen_task_id, action = candidates[0]
        else:
            if self.task_action_resolver is None:
                raise RuntimeError(
                    f"Robot '{self.robot_id}' produced {len(candidates)} actions "
                    f"({[c[0] for c in candidates]}) but has no task_action_resolver. "
                    "Provide one to combine concurrent + sequential task outputs."
                )
            chosen_task_id = candidates[-1][0]
            action = self.task_action_resolver(
                self.robot_id, [a for _, a in candidates]
            )

        adapted = self.action_adapter.process(action)
        action_space = self.tasks[chosen_task_id].action_space()
        return self._safety.filter(adapted, action_space)

    # ------------------------------------------------------------------
    # Arbitration controls
    # ------------------------------------------------------------------

    @property
    def active_task_id(self) -> Optional[str]:
        return self._arbitrator.active_task_id

    def switch_task(self, to_task_id: str, reason: str = "") -> ArbitrationEvent:
        return self._arbitrator.switch(to_task_id, reason=reason)

    def set_task_weights(self, weights: Mapping[str, float]) -> None:
        if not isinstance(self.task_selector, WeightTaskSelector):
            raise RuntimeError(
                f"Robot '{self.robot_id}' uses a custom task_selector; "
                "set_task_weights only applies to the default weight selector."
            )
        self.task_selector.update(weights)

    def consume_arbitration_events(self) -> list[ArbitrationEvent]:
        return self._arbitrator.consume_events()

    def active_task_ids(self) -> list[str]:
        return list(self._arbitrator.active_and_concurrent())

    # ------------------------------------------------------------------
    # Helpers used elsewhere (action_space inference fallback)
    # ------------------------------------------------------------------

    def infer_action_space(self, action: Action) -> ActionSpace:
        return infer_action_space(action)
