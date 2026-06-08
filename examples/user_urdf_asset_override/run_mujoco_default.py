"""Demonstrate URDF canonical description failing without MJCF."""

from __future__ import annotations

from examples._bootstrap import ensure_repo_on_path

ensure_repo_on_path()

from robodeploy import RoboEnv  # noqa: E402

from examples.user_urdf_asset_override import components  # noqa: E402,F401


def main() -> None:
    env = RoboEnv.make(
        robot="user_urdf_robot",
        backend="mujoco",
        task="user_dummy_task",
        policy="user_hold_policy",
        backend_kwargs={"enable_viewer": False},
    )
    try:
        _, info = env.reset()
    except Exception as exc:
        print("Expected failure:", exc)
        return
    print("Unexpected success. assets:", info.extra.get("assets"))


if __name__ == "__main__":
    main()

