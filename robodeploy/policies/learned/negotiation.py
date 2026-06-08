"""Action-space compatibility negotiation between policies and backends."""

from __future__ import annotations

from robodeploy.action_adapter import ActionAdapter, DeltaEEToJointPosTransform, IActionTransform, IdentityActionTransform
from robodeploy.core.interfaces.backend import IBackend
from robodeploy.core.interfaces.policy import IPolicy
from robodeploy.core.spaces import ActionSpace, infer_action_space
from robodeploy.description.base import RobotDescription


class ActionSpaceIncompatibility(ValueError):
    """Raised when a policy output space cannot be adapted to a backend."""


_ADAPT_PATHS: dict[tuple[ActionSpace, ActionSpace], str] = {
    (ActionSpace.DELTA_EE, ActionSpace.JOINT_POS): "delta_ee_to_joint_pos",
    (ActionSpace.CARTESIAN_POSE, ActionSpace.JOINT_POS): "cartesian_to_joint_pos",
    (ActionSpace.JOINT_VEL, ActionSpace.JOINT_POS): "unsupported",
}


def can_adapt(source: ActionSpace, target: ActionSpace) -> bool:
    if source == target:
        return True
    path = _ADAPT_PATHS.get((source, target))
    return path == "delta_ee_to_joint_pos" or path == "cartesian_to_joint_pos"


def build_transforms(
    source: ActionSpace,
    target: ActionSpace,
    description: RobotDescription,
    *,
    dt: float = 0.01,
) -> list[IActionTransform]:
    if source == target:
        return [IdentityActionTransform()]
    if source == ActionSpace.DELTA_EE and target == ActionSpace.JOINT_POS:
        return [DeltaEEToJointPosTransform(description.get_kinematics_solver(), dt=dt)]
    if source == ActionSpace.CARTESIAN_POSE and target == ActionSpace.JOINT_POS:
        return [DeltaEEToJointPosTransform(description.get_kinematics_solver(), dt=dt)]
    raise ActionSpaceIncompatibility(
        f"No adapter from {source.name} to {target.name}."
    )


def negotiate_action_space(
    policy: IPolicy,
    backend: IBackend,
    description: RobotDescription,
    *,
    existing_adapter: ActionAdapter | None = None,
    dt: float = 0.01,
) -> tuple[IPolicy, ActionSpace, ActionAdapter]:
    """Return policy, effective action space, and robot-level action adapter."""
    raw = getattr(backend, "supported_action_spaces", None)
    if raw is None or not isinstance(raw, (list, tuple)):
        supported: list[ActionSpace] = []
    else:
        supported = list(raw)
    if not supported:
        adapter = existing_adapter or ActionAdapter()
        return policy, policy.action_space, adapter

    if policy.action_space in supported:
        adapter = existing_adapter or ActionAdapter()
        return policy, policy.action_space, adapter

    target = supported[0]
    if not can_adapt(policy.action_space, target):
        raise ActionSpaceIncompatibility(
            f"Policy outputs {policy.action_space.name} but backend supports "
            f"{[s.name for s in supported]}. No adapter available."
        )

    transforms = build_transforms(policy.action_space, target, description, dt=dt)
    if existing_adapter is not None:
        merged = list(existing_adapter.transforms) + transforms
        adapter = ActionAdapter(merged)
    else:
        adapter = ActionAdapter(transforms)
    return policy, target, adapter


def effective_action_space_for_step(
    *,
    declared_space: ActionSpace,
    adapted_action,
    robot_effective_space: ActionSpace | None,
) -> ActionSpace:
    if robot_effective_space is not None:
        return robot_effective_space
    inferred = infer_action_space(adapted_action)
    if inferred != declared_space:
        return inferred
    return declared_space
