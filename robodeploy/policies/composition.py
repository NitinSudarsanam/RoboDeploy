"""Sequential policy composition on the normalized IPolicy contract."""

from __future__ import annotations

from robodeploy.core.interfaces.policy import IPolicy
from robodeploy.core.registry import register_policy
from robodeploy.core.spaces import ActionSpace
from robodeploy.core.types import Action, Observation
from robodeploy.policies.base import PolicyBase


@register_policy("policy_chain")
class PolicyChain(PolicyBase):
    """Run child policies in order; later policies may refine the action."""

    def __init__(self, config: dict | None = None, *, policies: list[IPolicy] | None = None) -> None:
        cfg = dict(config or {})
        children = list(policies or cfg.get("policies") or [])
        if not children:
            raise ValueError("PolicyChain requires at least one child policy.")
        spaces = {child.action_space for child in children}
        if len(spaces) != 1:
            raise ValueError(f"PolicyChain children mix action spaces: {spaces}")
        super().__init__(action_space=next(iter(spaces)), config=cfg)
        self._children = children
        self._mode = str(cfg.get("mode", "refine"))

    def _reset_impl(self) -> None:
        for child in self._children:
            child.reset()

    def get_action(self, obs: Observation) -> Action:
        action = self._children[0].get_action(obs)
        for child in self._children[1:]:
            candidate = child.get_action(obs)
            action = self._merge(action, candidate)
        return action

    def _merge(self, base: Action, update: Action) -> Action:
        if self._mode == "last":
            return update
        merged = Action(
            joint_positions=update.joint_positions if update.joint_positions is not None else base.joint_positions,
            joint_velocities=update.joint_velocities if update.joint_velocities is not None else base.joint_velocities,
            joint_torques=update.joint_torques if update.joint_torques is not None else base.joint_torques,
            ee_position=update.ee_position if update.ee_position is not None else base.ee_position,
            ee_orientation=update.ee_orientation if update.ee_orientation is not None else base.ee_orientation,
            ee_velocity=update.ee_velocity if update.ee_velocity is not None else base.ee_velocity,
            gripper=update.gripper if update.gripper is not None else base.gripper,
            timestamp=update.timestamp or base.timestamp,
            action_space=update.action_space or base.action_space,
            is_delta_ee=update.is_delta_ee or base.is_delta_ee,
        )
        return merged
