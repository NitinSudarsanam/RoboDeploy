"""Franka Panda simulation pointer.

The old oscillation demo depended on a removed API. Use the SO-101 or user Kuka
switch-simulator examples as the currently maintained end-to-end demos.
"""


def main() -> None:
    print("This legacy demo has been retired. Try:")
    print("  python -m examples.so101.run_switch_simulator --backend mujoco --home")
    print("  python -m examples.user_kuka_sinusoid.run_mujoco")


if __name__ == "__main__":
    main()

