"""Deploy a learned policy with unified ModelLoader + action-space negotiation.

Dry-run (no checkpoint, injectable predict_fn):

    python examples/learned_policy_deploy/run.py --dry-run --steps 5

Serve over ZMQ (separate terminal):

    robodeploy serve-policy --policy vla_stub --host 127.0.0.1 --port 5555

Load from HF alias (requires huggingface_hub + model packages):

    python examples/learned_policy_deploy/run.py --policy-ref hf:openvla-7b
"""

from __future__ import annotations

import argparse

import numpy as np

from robodeploy.core.robot import Robot, RobotTask
from robodeploy.core.spaces import ActionSpace
from robodeploy.env import RoboEnv
from robodeploy.policies.learned.diffusion import DiffusionPolicy
from robodeploy.policies.learned.factory import load_policy_from_ref
from robodeploy.testing import DummyBackend, DummyRobot, DummyTask


def _make_policy(*, dry_run: bool, policy_ref: str | None):
    if policy_ref:
        return load_policy_from_ref(policy_ref, action_space=ActionSpace.DELTA_EE)
    if dry_run:

        def predict_plan(packet):
            del packet
            return [{"ee_position": [0.02, 0.0, 0.0]}]

        return DiffusionPolicy(
            config={
                "predict_plan_fn": predict_plan,
                "action_space": "delta_ee",
                "replan_interval": 2,
            }
        )
    raise ValueError("Provide --policy-ref or --dry-run.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Use injectable predict_plan_fn.")
    parser.add_argument("--policy-ref", default=None, help="Policy ref (hf:name, diffusion:ckpt.pt, vla_stub).")
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    policy = _make_policy(dry_run=bool(args.dry_run), policy_ref=args.policy_ref)
    robot = Robot(
        robot_id="robot0",
        description=DummyRobot(),
        tasks={"task0": RobotTask(task=DummyTask(), policies={"p": policy})},
    )
    env = RoboEnv(backend=DummyBackend(), robots=[robot])
    obs, info = env.reset()
    print("reset effective_action_space:", robot.effective_action_space)
    for step in range(int(args.steps)):
        obs, reward, done, info = env.step()
        diag = info.extra.get("policy_diagnostics", {})
        print(f"step={step} reward={reward:.3f} diag_count={diag.get('count', 0)}")
        if done:
            break
    env.close()


if __name__ == "__main__":
    main()
