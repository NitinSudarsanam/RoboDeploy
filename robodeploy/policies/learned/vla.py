"""VLAPolicy placeholder matching architecture layout."""

from __future__ import annotations

from robodeploy.core.registry import register_policy
from robodeploy.core.spaces import ActionSpace
from robodeploy.core.types import Action, Observation
from robodeploy.policies.base import PolicyBase


@register_policy("vla")
class VLAPolicy(PolicyBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(action_space=ActionSpace.JOINT_POS)

    def get_action(self, obs: Observation) -> Action:
        raise NotImplementedError("VLAPolicy placeholder only.")

