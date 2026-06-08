"""Demonstrate URDF canonical description with explicit MJCF override."""

from __future__ import annotations

from examples._bootstrap import ensure_repo_on_path

ensure_repo_on_path()

from robodeploy import RoboEnv  # noqa: E402

from examples.user_urdf_asset_override import components  # noqa: E402,F401


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    mjcf_path = repo_root / "robodeploy" / "description" / "kuka" / "assets" / "mjcf" / "kuka.xml"

    env = RoboEnv.make(
        robot="user_urdf_robot",
        backend="mujoco",
        task="user_dummy_task",
        policy="user_hold_policy",
        backend_kwargs={
            "enable_viewer": False,
            "allow_actuator_name_fallback": True,
            "asset_overrides": {
                "robot0": {"mjcf": str(mjcf_path)},
            },
        },
    )

    try:
        _, info = env.reset()
    except ImportError as exc:
        print(exc)
        print("\nInstall mujoco to run:\n  pip install mujoco")
        return

    print("assets selection:", info.extra.get("assets"))
    env.close()


if __name__ == "__main__":
    main()

