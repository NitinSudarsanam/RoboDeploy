"""RoboDeploy CLI.

Thin wrapper over the public Python APIs for registry listing, dummy smoke runs,
dataset export, and policy serving. Demo YAML presets live under ``examples/`` —
use ``python -m examples.cli`` for preset-based commands.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from robodeploy.cli_helpers import (
    action_fn_for_mode,
    close_quietly,
    episode_info_summary,
    print_json,
)


def _import_custom_modules(custom_modules: list[str]) -> None:
    if not custom_modules:
        return
    from robodeploy.core.registry import use

    for mod in custom_modules:
        use(str(mod))


def _make_dummy_env():
    from robodeploy.core.robot import Robot, RobotTask
    from robodeploy.env import RoboEnv
    from robodeploy.testing import DummyBackend, DummyPolicy, DummyRobot, DummyTask

    robot = Robot(
        robot_id="robot0",
        description=DummyRobot(),
        tasks={"task0": RobotTask(task=DummyTask(), policies={"p": DummyPolicy(0.0)})},
    )
    return RoboEnv(backend=DummyBackend(), robots=[robot])


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="robodeploy", add_help=True)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_reg = sub.add_parser("list-registry", help="List registered component names.")
    p_reg.add_argument(
        "--discover",
        action="store_true",
        help="Load Python entry points before listing (pip-installed extensions).",
    )
    p_reg.add_argument(
        "--custom-module",
        action="append",
        default=[],
        help="Import dotted module path(s) before listing (register project components).",
    )
    p_reg.add_argument(
        "--builtins",
        action="store_true",
        help="Import builtin modules before listing (populates robots/tasks/policies).",
    )
    p_reg.add_argument("--json", action="store_true", help="Print as JSON object.")
    p_reg.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")

    p_export = sub.add_parser("export-episode", help="Run a dummy episode and export a recording.")
    p_export.add_argument("--steps", type=int, default=50, help="Number of env steps to run.")
    p_export.add_argument("--out", required=True, help="Output file path.")
    p_export.add_argument(
        "--format",
        choices=("jsonl", "hdf5"),
        default="jsonl",
        help="Export format.",
    )
    p_export.add_argument(
        "--dummy",
        action="store_true",
        help="Use built-in dummy backend/robot/task (required; preset export moved to examples.cli).",
    )
    p_export.add_argument(
        "--action",
        choices=("none", "zero", "hold", "sinusoid"),
        default="none",
        help="Inject explicit actions instead of using policy actions.",
    )
    p_export.add_argument("--json", action="store_true", help="Print a structured JSON result.")
    p_export.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")

    p_run = sub.add_parser("run-episode", help="Run a dummy episode and print EpisodeInfo summary JSON.")
    p_run.add_argument("--steps", type=int, default=50, help="Number of env steps to run.")
    p_run.add_argument(
        "--dummy",
        action="store_true",
        help="Use built-in dummy backend/robot/task (required; preset runs moved to examples.cli).",
    )
    p_run.add_argument(
        "--action",
        choices=("none", "zero", "hold", "sinusoid"),
        default="none",
        help="Inject explicit actions instead of using policy actions.",
    )
    p_run.add_argument("--json", action="store_true", help="Print a structured JSON result.")
    p_run.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")

    p_serve = sub.add_parser("serve-policy", help="Serve a registered policy via ZMQ or gRPC.")
    p_serve.add_argument(
        "--custom-module",
        action="append",
        default=[],
        help="Import dotted module path(s) before looking up policy.",
    )
    p_serve.add_argument(
        "--policy",
        required=True,
        help="Registered policy name, hf:<model>, or framework:checkpoint (e.g. vla_stub, hf:openvla-7b).",
    )
    p_serve.add_argument("--checkpoint", default=None, help="Optional checkpoint path for learned policies.")
    p_serve.add_argument("--model-spec", default=None, help="Optional JSON ModelSpec file path.")
    p_serve.add_argument("--host", default="0.0.0.0", help="Bind host/interface.")
    p_serve.add_argument("--port", type=int, default=5555, help="Bind port.")
    p_serve.add_argument("--transport", choices=("zmq", "grpc"), default="zmq", help="Transport.")
    p_serve.add_argument("--quiet", action="store_true", help="Disable verbose request logging.")

    p_models = sub.add_parser("models", help="List or download known Hugging Face learned policies.")
    models_sub = p_models.add_subparsers(dest="models_cmd", required=True)
    p_models_list = models_sub.add_parser("list", help="List known HF model aliases.")
    p_models_list.add_argument("--json", action="store_true", help="Print as JSON.")
    p_models_list.add_argument("--pretty", action="store_true", help="Pretty-print JSON.")
    p_models_dl = models_sub.add_parser("download", help="Cache a known HF model locally.")
    p_models_dl.add_argument("name", help="Model alias (e.g. openvla-7b).")
    p_models_dl.add_argument("--json", action="store_true", help="Print resolved cache path as JSON.")

    p_dr = sub.add_parser("dr-sweep", help="Run a domain-randomization sensitivity sweep (dummy env).")
    p_dr.add_argument("--dummy", action="store_true", help="Use built-in dummy backend (required).")
    p_dr.add_argument("--output", required=True, help="Output directory for sweep JSON report.")
    p_dr.add_argument("--seeds", type=int, default=2, help="Seeds per sweep cell.")
    p_dr.add_argument("--episodes", type=int, default=2, help="Episodes per seed.")
    p_dr.add_argument("--steps", type=int, default=20, help="Max steps per episode.")
    p_dr.add_argument("--json", action="store_true", help="Print report JSON to stdout.")
    p_dr.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")

    p_xfer = sub.add_parser(
        "transfer-eval",
        help="Compare sim vs noisy-sim rollouts as a transfer-gap proxy (dummy env).",
    )
    p_xfer.add_argument("--dummy", action="store_true", help="Use built-in dummy backend (required).")
    p_xfer.add_argument("--output", required=True, help="Output directory for transfer report.")
    p_xfer.add_argument("--episodes", type=int, default=3, help="Matched episodes to run.")
    p_xfer.add_argument("--steps", type=int, default=20, help="Max steps per episode.")
    p_xfer.add_argument("--json", action="store_true", help="Print metrics JSON to stdout.")
    p_xfer.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")

    p_cal = sub.add_parser("calibrate", help="Kinematic, extrinsic, hand-eye, and system-ID calibration.")
    cal_sub = p_cal.add_subparsers(dest="calibrate_cmd", required=True)

    p_cal_kin = cal_sub.add_parser("kinematic", help="Kinematic calibration (SO-101 supported).")
    p_cal_kin.add_argument("--robot", required=True, help="Robot id (e.g. so101).")
    p_cal_kin.add_argument("--port", default="/dev/ttyACM0", help="Serial port for bus calibration.")
    p_cal_kin.add_argument("--out", default=None, help="Output JSON path.")
    p_cal_kin.add_argument("--json", action="store_true", help="Print result JSON.")

    p_cal_ext = cal_sub.add_parser("extrinsic", help="Camera extrinsic calibration (checkerboard).")
    p_cal_ext.add_argument("--camera", required=True, help="Camera name (e.g. wrist).")
    p_cal_ext.add_argument("--pattern", default="checkerboard", choices=("checkerboard",))
    p_cal_ext.add_argument("--board", required=True, help="Board spec COLSxROWSxSIZE_M (e.g. 7x5x0.025).")
    p_cal_ext.add_argument("--robot-id", default="default", help="CalibrationStore robot id.")
    p_cal_ext.add_argument("--json", action="store_true", help="Print result JSON.")

    p_cal_he = cal_sub.add_parser("handeye", help="Hand-eye calibration (ArUco / checkerboard poses).")
    p_cal_he.add_argument("--robot", required=True, help="Robot id (e.g. franka).")
    p_cal_he.add_argument("--pattern", default="aruco", choices=("aruco", "checkerboard"))
    p_cal_he.add_argument("--method", default="park", choices=("tsai", "park", "daniilidis"))
    p_cal_he.add_argument("--json", action="store_true", help="Print result JSON.")

    p_cal_sid = cal_sub.add_parser("system-id", help="Friction + payload system identification.")
    p_cal_sid.add_argument("--robot", required=True, help="Robot id.")
    p_cal_sid.add_argument("--joint", type=int, default=0, help="Joint index to characterize.")
    p_cal_sid.add_argument("--dummy", action="store_true", help="Run on dummy env (CI-safe).")
    p_cal_sid.add_argument("--json", action="store_true", help="Print result JSON.")

    p_scaffold = sub.add_parser("scaffold", help="Generate task/policy skeleton files.")
    sc_sub = p_scaffold.add_subparsers(dest="scaffold_kind", required=True)

    p_sc_task = sc_sub.add_parser("task", help="Scaffold a new task module.")
    p_sc_task.add_argument("--name", required=True, help="Task name.")
    p_sc_task.add_argument(
        "--template",
        choices=("pick_place", "custom"),
        default="pick_place",
        help="Task template.",
    )
    p_sc_task.add_argument("--output", required=True, help="Output .py path.")
    p_sc_task.add_argument("--force", action="store_true", help="Overwrite existing file.")

    p_sc_policy = sc_sub.add_parser("policy", help="Scaffold a policy module or reach DSL YAML.")
    p_sc_policy.add_argument("--name", required=True, help="Policy name.")
    p_sc_policy.add_argument(
        "--template",
        choices=("reach_dsl", "custom"),
        default="reach_dsl",
        help="Policy template.",
    )
    p_sc_policy.add_argument("--output", required=True, help="Output .py or .yaml path.")
    p_sc_policy.add_argument("--force", action="store_true", help="Overwrite existing file.")

    p_sc_preset = sc_sub.add_parser("preset", help="Scaffold a preset YAML snippet.")
    p_sc_preset.add_argument("--name", required=True, help="Preset name.")
    p_sc_preset.add_argument("--robot", required=True, help="Registered robot name.")
    p_sc_preset.add_argument("--backend", default="mujoco", help="Backend name.")
    p_sc_preset.add_argument("--task", default="pick_place", help="Task name.")
    p_sc_preset.add_argument("--policy", default="example_sensor_reach_pick", help="Policy name.")
    p_sc_preset.add_argument(
        "--template",
        choices=("sim", "real", "manipulate"),
        default="sim",
        help="Base preset template (examples/presets/).",
    )
    p_sc_preset.add_argument("--output", required=True, help="Output .yaml path.")
    p_sc_preset.add_argument("--force", action="store_true", help="Overwrite existing file.")

    p_sc_robot = sc_sub.add_parser("robot", help="Scaffold a robot description package.")
    p_sc_robot.add_argument("--name", required=True, help="Robot name.")
    p_sc_robot.add_argument("--dof", type=int, default=6, help="Controlled DoF count.")
    p_sc_robot.add_argument(
        "--description-dir",
        default="robodeploy/description",
        help="Root description directory.",
    )
    p_sc_robot.add_argument("--force", action="store_true", help="Overwrite existing files.")

    p_sc_sensor = sc_sub.add_parser("sensor", help="Scaffold a sensor implementation module.")
    p_sc_sensor.add_argument("--name", required=True, help="Sensor name.")
    p_sc_sensor.add_argument(
        "--backend",
        choices=("mujoco", "gazebo", "isaacsim", "real"),
        default="mujoco",
        help="Target backend.",
    )
    p_sc_sensor.add_argument("--output", required=True, help="Output .py path.")
    p_sc_sensor.add_argument("--force", action="store_true", help="Overwrite existing file.")

    p_sc_example = sc_sub.add_parser("example", help="Scaffold a preset-based example runner.")
    p_sc_example.add_argument("--name", required=True, help="Example name.")
    p_sc_example.add_argument("--preset", required=True, help="Preset name from presets.yaml.")
    p_sc_example.add_argument("--output", required=True, help="Output run.py path.")
    p_sc_example.add_argument("--force", action="store_true", help="Overwrite existing file.")

    p_doctor = sub.add_parser("doctor", help="Check environment health and dependencies.")
    p_doctor.add_argument("--json", action="store_true", help="Print structured JSON report.")
    p_doctor.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")

    p_lint = sub.add_parser("lint", help="Lint tasks, policies, or presets.")
    lint_sub = p_lint.add_subparsers(dest="lint_kind", required=True)

    p_lint_task = lint_sub.add_parser("task", help="Lint a task .py file.")
    p_lint_task.add_argument("path", help="Path to task module.")
    p_lint_task.add_argument("--json", action="store_true", help="Print issues as JSON.")

    p_lint_policy = lint_sub.add_parser("policy", help="Lint a policy .py or reach DSL .yaml.")
    p_lint_policy.add_argument("path", help="Path to policy module or YAML.")
    p_lint_policy.add_argument("--json", action="store_true", help="Print issues as JSON.")

    p_lint_preset = lint_sub.add_parser("preset", help="Lint a presets YAML file.")
    p_lint_preset.add_argument("path", help="Path to presets.yaml.")
    p_lint_preset.add_argument("--check", default=None, help="Verify a named preset exists.")
    p_lint_preset.add_argument("--json", action="store_true", help="Print issues as JSON.")

    p_lint_all = lint_sub.add_parser("all", help="Lint example tasks, policies, and presets.")
    p_lint_all.add_argument("--json", action="store_true", help="Print issues as JSON.")

    p_scene = sub.add_parser("scene", help="Validate or inspect scene YAML files.")
    scene_sub = p_scene.add_subparsers(dest="scene_cmd", required=True)

    p_scene_val = scene_sub.add_parser("validate", help="Validate a scene YAML file.")
    p_scene_val.add_argument("scene", help="Path to scene.yaml.")
    p_scene_val.add_argument("--backend", default=None, help="Target backend.")
    p_scene_val.add_argument("--json", action="store_true", help="Print report as JSON.")
    p_scene_val.add_argument("--pretty", action="store_true", help="Pretty-print JSON.")

    p_scene_ins = scene_sub.add_parser("inspect", help="Inspect scene contents.")
    p_scene_ins.add_argument("scene", help="Path to scene.yaml.")
    p_scene_ins.add_argument("--backend", default=None, help="Target backend.")
    p_scene_ins.add_argument("--json", action="store_true", help="Print as JSON.")
    p_scene_ins.add_argument("--pretty", action="store_true", help="Pretty-print JSON.")

    p_config = sub.add_parser("config", help="Show, resolve, validate, or diff presets.")
    cfg_sub = p_config.add_subparsers(dest="config_cmd", required=True)

    p_cfg_show = cfg_sub.add_parser("show", help="Show a preset configuration.")
    p_cfg_show.add_argument("--preset", required=True, help="Preset name.")
    p_cfg_show.add_argument("--presets-file", default=None, help="Path to presets.yaml.")
    p_cfg_show.add_argument("--json", action="store_true", help="Print as JSON.")
    p_cfg_show.add_argument("--pretty", action="store_true", help="Pretty-print JSON.")

    p_cfg_resolve = cfg_sub.add_parser("resolve", help="Resolve a preset to full config.")
    p_cfg_resolve.add_argument("--preset", required=True, help="Preset name.")
    p_cfg_resolve.add_argument("--presets-file", default=None, help="Path to presets.yaml.")
    p_cfg_resolve.add_argument("--json", action="store_true", help="Print as JSON.")
    p_cfg_resolve.add_argument("--pretty", action="store_true", help="Pretty-print JSON.")

    p_cfg_validate = cfg_sub.add_parser("validate", help="Validate a presets YAML file.")
    p_cfg_validate.add_argument("path", help="Path to presets.yaml.")
    p_cfg_validate.add_argument("--json", action="store_true", help="Print as JSON.")
    p_cfg_validate.add_argument("--pretty", action="store_true", help="Pretty-print JSON.")

    p_cfg_diff = cfg_sub.add_parser("diff", help="Diff two presets.")
    p_cfg_diff.add_argument("preset_a", help="First preset name.")
    p_cfg_diff.add_argument("preset_b", help="Second preset name.")
    p_cfg_diff.add_argument("--presets-file", default=None, help="Path to presets.yaml.")
    p_cfg_diff.add_argument("--json", action="store_true", help="Print as JSON.")
    p_cfg_diff.add_argument("--pretty", action="store_true", help="Pretty-print JSON.")

    p_assets = sub.add_parser("assets", help="List, resolve, and verify bundled assets.")
    assets_sub = p_assets.add_subparsers(dest="assets_cmd", required=True)

    p_assets_verify = assets_sub.add_parser("verify", help="Verify SHA256 of bundled assets.")
    p_assets_verify.add_argument("--json", action="store_true", help="Print results as JSON.")
    p_assets_verify.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")

    p_assets_list = assets_sub.add_parser("list", help="List known assets.")
    p_assets_list.add_argument("--robot", action="store_true", help="Robots only.")
    p_assets_list.add_argument("--mesh", action="store_true", help="Meshes only.")
    p_assets_list.add_argument("--mjcf", action="store_true", help="MJCF files only.")
    p_assets_list.add_argument("--json", action="store_true", help="Print as JSON.")
    p_assets_list.add_argument("--pretty", action="store_true", help="Pretty-print JSON.")

    p_assets_resolve = assets_sub.add_parser("resolve", help="Resolve asset path for a backend.")
    p_assets_resolve.add_argument("name", help="Asset or robot name.")
    p_assets_resolve.add_argument("--backend", default="mujoco", help="Target backend.")
    p_assets_resolve.add_argument("--json", action="store_true", help="Print as JSON.")
    p_assets_resolve.add_argument("--pretty", action="store_true", help="Pretty-print JSON.")

    p_assets_info = assets_sub.add_parser("info", help="Show asset details.")
    p_assets_info.add_argument("name", help="Asset or robot name.")
    p_assets_info.add_argument("--json", action="store_true", help="Print as JSON.")
    p_assets_info.add_argument("--pretty", action="store_true", help="Pretty-print JSON.")

    p_eval = sub.add_parser("eval", help="Run a benchmark evaluation and write a JSON report.")
    p_eval.add_argument(
        "--benchmark",
        required=True,
        help="Benchmark id, e.g. manipulation_v1/reach_target or manipulation_v1 (full suite).",
    )
    p_eval.add_argument(
        "--policy",
        default="scripted",
        help="Registered policy name, or action mode: zero|hold|sinusoid|scripted.",
    )
    p_eval.add_argument(
        "--backend",
        default="dummy",
        help="Backend preset suffix (preset_<backend>.yaml), default dummy.",
    )
    p_eval.add_argument("--episodes", type=int, default=100, help="Number of evaluation episodes.")
    p_eval.add_argument("--seed", type=int, default=0, help="Base seed for episode sequence.")
    p_eval.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Override per-episode step budget (0 = task default).",
    )
    p_eval.add_argument("--output", default="", help="Write JSON report to this path.")
    p_eval.add_argument(
        "--benchmarks-root",
        default="",
        help="Override benchmarks/ discovery path (default: repo benchmarks/ or ROBODEPLOY_BENCHMARKS_ROOT).",
    )
    p_eval.add_argument(
        "--sweep-backends",
        action="store_true",
        help="Run all preset_<backend>.yaml files for each task.",
    )
    p_eval.add_argument(
        "--parallel",
        action="store_true",
        help="Evaluate episodes in parallel subprocess workers (SubprocVecEnv-style).",
    )
    p_eval.add_argument("--workers", type=int, default=4, help="Parallel worker count.")
    p_eval.add_argument("--record-videos", action="store_true", help="Record per-episode videos when RGB is available.")
    p_eval.add_argument("--video-dir", default="", help="Directory for episode videos.")
    p_eval.add_argument("--html", default="", help="Write HTML eval report to this path.")
    p_eval.add_argument("--baseline", default="", help="Baseline JSON report for HTML comparison table.")
    p_eval.add_argument("--json", action="store_true", help="Print report JSON to stdout.")
    p_eval.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")

    p_eval_cmp = sub.add_parser("eval-compare", help="Compare two eval JSON reports and write HTML.")
    p_eval_cmp.add_argument("report_a", help="First eval report JSON path.")
    p_eval_cmp.add_argument("report_b", help="Second eval report JSON path.")
    p_eval_cmp.add_argument("--output", required=True, help="Output comparison HTML path.")

    p_leaderboard = sub.add_parser("leaderboard", help="Leaderboard submit/show commands.")
    lb_sub = p_leaderboard.add_subparsers(dest="leaderboard_cmd", required=True)
    p_lb_submit = lb_sub.add_parser("submit", help="Validate and write a leaderboard submission JSON.")
    p_lb_submit.add_argument("report", help="Eval report JSON path.")
    p_lb_submit.add_argument("--benchmark", required=True, help="Benchmark id, e.g. manipulation_v1/reach_target.")
    p_lb_submit.add_argument("--author", required=True, help="Submission author name.")
    p_lb_submit.add_argument("--checkpoint", default="", help="Optional policy checkpoint path.")
    p_lb_submit.add_argument(
        "--benchmarks-root",
        default="",
        help="Override benchmarks/ discovery path.",
    )
    p_lb_show = lb_sub.add_parser("show", help="Show leaderboard entries for a suite.")
    p_lb_show.add_argument("suite", help="Suite name, e.g. manipulation_v1.")
    p_lb_show.add_argument(
        "--benchmarks-root",
        default="",
        help="Override benchmarks/ discovery path.",
    )
    p_lb_show.add_argument("--json", action="store_true", help="Print entries as JSON.")

    p_list_bench = sub.add_parser("list-benchmarks", help="List registered benchmark suites and tasks.")
    p_list_bench.add_argument(
        "--benchmarks-root",
        default="",
        help="Override benchmarks/ discovery path.",
    )
    p_list_bench.add_argument("--json", action="store_true", help="Print as JSON.")
    p_list_bench.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")

    p_train = sub.add_parser("train", help="Train policies (BC, checkpoint eval).")
    train_sub = p_train.add_subparsers(dest="train_cmd", required=True)

    p_bc = train_sub.add_parser("bc", help="Behavior cloning from recorded demos.")
    p_bc.add_argument("--dataset", required=True, help="JSONL or HDF5 demo dataset path.")
    p_bc.add_argument("--obs", default="proprio", help="Comma-separated observation keys.")
    p_bc.add_argument("--action-dim", type=int, default=None, help="Action dimension override.")
    p_bc.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    p_bc.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    p_bc.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    p_bc.add_argument("--log-dir", default="./runs/bc", help="Checkpoint and log directory.")
    p_bc.add_argument("--log", default="", help="Logging backend: wandb, tensorboard, or empty.")
    p_bc.add_argument("--out", default=None, help="Final checkpoint path (default: log-dir/bc_final.pt).")
    p_bc.add_argument("--dummy", action="store_true", help="Synthesize dummy demos if dataset is missing.")
    p_bc.add_argument("--json", action="store_true", help="Print structured JSON result.")

    p_train_eval = train_sub.add_parser("eval", help="Evaluate a BC checkpoint on dummy env.")
    p_train_eval.add_argument("--checkpoint", required=True, help="Checkpoint .pt path.")
    p_train_eval.add_argument("--episodes", type=int, default=10, help="Evaluation episodes.")
    p_train_eval.add_argument("--dummy", action="store_true", help="Use built-in dummy env.")
    p_train_eval.add_argument("--json", action="store_true", help="Print structured JSON result.")

    p_ppo = train_sub.add_parser("ppo", help="PPO reinforcement learning (sim / dummy).")
    p_ppo.add_argument("--preset", default="", help="Example preset name (requires examples on path).")
    p_ppo.add_argument("--n-envs", type=int, default=4, help="Parallel env count.")
    p_ppo.add_argument("--total-steps", type=int, default=10_000, help="Total environment steps.")
    p_ppo.add_argument("--rollout-steps", type=int, default=256, help="Steps per rollout.")
    p_ppo.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    p_ppo.add_argument("--log-dir", default="./runs/ppo", help="Checkpoint directory.")
    p_ppo.add_argument("--log", default="", help="Logging backend: wandb, tensorboard, or empty.")
    p_ppo.add_argument("--dummy", action="store_true", help="Use built-in dummy env (default when no preset).")
    p_ppo.add_argument("--json", action="store_true", help="Print structured JSON result.")

    p_convert = sub.add_parser("convert-dataset", help="Convert between demo dataset formats.")
    p_convert.add_argument(
        "--from",
        dest="from_path",
        required=True,
        help="Source: path, lerobot://repo_id, or lerobot://org/dataset (HuggingFace LeRobot).",
    )
    p_convert.add_argument("--to", dest="to_path", required=True, help="Destination path (.jsonl or .hdf5).")
    p_convert.add_argument(
        "--lerobot-root",
        default=None,
        help="Local LeRobot dataset root (for lerobot:// sources not on the Hub).",
    )
    p_convert.add_argument("--json", action="store_true", help="Print structured JSON result.")

    p_safety = sub.add_parser("safety", help="Check, test, or inspect safety monitor state.")
    safety_sub = p_safety.add_subparsers(dest="safety_cmd", required=True)

    p_safety_check = safety_sub.add_parser("check", help="Validate robot limits and guard wiring.")
    p_safety_check.add_argument("--preset", default=None, help="Preset name from examples/config/presets.yaml.")
    p_safety_check.add_argument("--robot", default=None, help="Registered robot name (when --preset omitted).")
    p_safety_check.add_argument(
        "--joint-limits",
        default=None,
        help="Optional YAML file with joint_position_limits / joint_velocity_limits overrides.",
    )
    p_safety_check.add_argument("--presets-file", default=None, help="Path to presets.yaml.")
    p_safety_check.add_argument("--json", action="store_true", help="Print as JSON.")
    p_safety_check.add_argument("--pretty", action="store_true", help="Pretty-print JSON.")

    p_safety_test = safety_sub.add_parser("test", help="Inject synthetic violations and verify monitor response.")
    p_safety_test.add_argument("--preset", default=None, help="Preset name (default: built-in dummy env).")
    p_safety_test.add_argument(
        "--inject",
        action="append",
        default=[],
        help="Injection spec, e.g. force_spike=80N, collision=arm,table, human_proximity=0.1m.",
    )
    p_safety_test.add_argument("--steps", type=int, default=3, help="Observation checks to run.")
    p_safety_test.add_argument("--presets-file", default=None, help="Path to presets.yaml.")
    p_safety_test.add_argument("--json", action="store_true", help="Print as JSON.")
    p_safety_test.add_argument("--pretty", action="store_true", help="Pretty-print JSON.")

    p_safety_status = safety_sub.add_parser("status", help="Show active SafetyMonitor from a running RoboEnv.")
    p_safety_status.add_argument("--json", action="store_true", help="Print as JSON.")
    p_safety_status.add_argument("--pretty", action="store_true", help="Pretty-print JSON.")

    from robodeploy.cli_observability import add_observability_parsers
    from robodeploy.cli_teleop import add_teleop_parsers

    add_observability_parsers(sub)
    add_teleop_parsers(sub)

    return parser


def _cmd_assets_verify(*, as_json: bool, pretty: bool) -> int:
    from robodeploy.assets import verify_assets

    rows = verify_assets()
    bad = [r for r in rows if r.get("status") != "ok"]
    payload = {"ok": not bad, "assets": rows}
    if as_json:
        print_json(payload, pretty=pretty)
        return 0 if not bad else 1
    for row in rows:
        print(f"{row.get('path')}: {row.get('status')}")
    return 0 if not bad else 1


def _cmd_list_registry(*, discover: bool, custom_modules: list[str], builtins: bool, as_json: bool, pretty: bool) -> int:
    if discover:
        from robodeploy import discover as _discover

        _discover()
    if custom_modules:
        from robodeploy.core.registry import use

        for mod in custom_modules:
            use(str(mod))
    if builtins:
        from robodeploy.builtins import import_builtins

        import_builtins()
    from robodeploy.core.registry import list_registered

    payload = list_registered()
    if as_json:
        print_json(payload, pretty=pretty)
        return 0

    for group in ("backends", "robots", "tasks", "policies", "sensors", "sensor_pairs"):
        items = payload.get(group, [])
        print(f"{group}:")
        for name in items:
            print(f"  - {name}")
    return 0


def _require_dummy(dummy: bool, *, cmd: str) -> None:
    if not dummy:
        raise ValueError(
            f"{cmd} requires --dummy in robodeploy CLI. "
            "Preset-based runs moved to: python -m examples.cli"
        )


def _cmd_export_episode(
    *,
    steps: int,
    out: str,
    fmt: str,
    dummy: bool,
    action_mode: str,
    as_json: bool,
    pretty: bool,
) -> int:
    _require_dummy(dummy, cmd="export-episode")
    env = _make_dummy_env()
    out_path = Path(out)
    try:
        action_fn = action_fn_for_mode(action_mode, env)
        recorder = env.run_episode(int(steps), record=True, action_fn=action_fn)
        if fmt == "hdf5":
            from robodeploy.dataset_export import export_demo_hdf5

            export_demo_hdf5(recorder, out_path)
        else:
            from robodeploy.dataset_export import export_demo_jsonl

            export_demo_jsonl(recorder, out_path)
    finally:
        close_quietly(env)
    if as_json:
        payload = {
            "out": str(out_path),
            "format": str(fmt),
            "steps": int(steps),
            "dummy": True,
            "preset": "",
            "action": str(action_mode),
        }
        print_json(payload, pretty=pretty)
    else:
        print(str(out_path))
    return 0


def _cmd_run_episode(
    *,
    steps: int,
    dummy: bool,
    action_mode: str,
    pretty: bool,
    as_json: bool,
) -> int:
    from robodeploy.policies.remote.http_client import to_jsonable

    _require_dummy(dummy, cmd="run-episode")
    env = _make_dummy_env()

    try:
        action_fn = action_fn_for_mode(action_mode, env)
        _, info = env.run_episode(int(steps), record=False, action_fn=action_fn)
        info_payload = to_jsonable(episode_info_summary(info))
        if as_json:
            payload = {
                "preset": "",
                "dummy": True,
                "steps": int(steps),
                "action": str(action_mode),
                "info": info_payload,
            }
            print_json(payload, pretty=pretty)
        else:
            print_json(info_payload, pretty=pretty)
        return 0
    finally:
        close_quietly(env)


def _resolve_serve_policy(
    *,
    policy: str,
    checkpoint: str | None,
    model_spec_path: str | None,
) -> object:
    from robodeploy.builtins import import_builtins
    from robodeploy.core.registry import get_policy
    from robodeploy.policies.learned.factory import load_model_spec_file, load_policy_from_ref, parse_policy_ref

    import_builtins()
    cfg: dict = {}
    if checkpoint:
        cfg["checkpoint"] = checkpoint
        cfg["checkpoint_path"] = checkpoint
    if model_spec_path:
        cfg["model_spec"] = load_model_spec_file(model_spec_path)

    ref = policy
    if checkpoint and ":" not in ref and not ref.startswith("hf:"):
        kind, _ = parse_policy_ref(ref)
        if kind in {"robomimic", "diffusion", "vla", "vla_stub", "diffusion_stub"}:
            ref = f"{kind}:{checkpoint}" if kind not in {"vla_stub", "diffusion_stub"} else ref

    if ref.startswith("hf:") or (":" in ref and not ref.startswith("http")) or ref.endswith((".pt", ".pth", ".ckpt")):
        return load_policy_from_ref(ref, config=cfg)

    PolicyClass = get_policy(ref)
    return PolicyClass(config=cfg or None)


def _cmd_serve_policy(
    *,
    policy: str,
    host: str,
    port: int,
    transport: str,
    quiet: bool,
    checkpoint: str | None = None,
    model_spec: str | None = None,
) -> int:
    from robodeploy.policies.remote.server import serve

    policy_obj = _resolve_serve_policy(policy=policy, checkpoint=checkpoint, model_spec_path=model_spec)
    serve(policy_obj, host=host, port=int(port), transport=str(transport), verbose=not quiet)
    return 0


def _cmd_models_list(*, as_json: bool, pretty: bool) -> int:
    from robodeploy.policies.learned.hf_hub import HFModelRegistry

    names = HFModelRegistry.list_models()
    if as_json:
        print_json({"models": names}, pretty=pretty)
    else:
        for name in names:
            print(name)
    return 0


def _cmd_models_download(*, name: str, as_json: bool) -> int:
    from robodeploy.policies.learned.hf_hub import HFModelRegistry

    path = HFModelRegistry.download(name)
    if as_json:
        print_json({"name": name, "path": path})
    else:
        print(path)
    return 0


def _cmd_dr_sweep(
    *,
    dummy: bool,
    output: str,
    seeds: int,
    episodes: int,
    steps: int,
    as_json: bool,
    pretty: bool,
) -> int:
    _require_dummy(dummy, cmd="dr-sweep")
    from robodeploy.tasks.randomization import DomainRandomizerConfig, RandomLevel
    from robodeploy.training.dr_sweep import DRSweep, DRSweepConfig

    def env_fn(dr_cfg: DomainRandomizerConfig, seed: int):
        from robodeploy.core.robot import Robot, RobotTask
        from robodeploy.env import RoboEnv
        from robodeploy.testing import DummyBackend, DummyPolicy, DummyRobot, DummyTask

        task = DummyTask(
            config={
                "domain_randomization": {
                    "level": dr_cfg.level.name,
                    "seed": seed,
                }
            }
        )
        robot = Robot(
            robot_id="robot0",
            description=DummyRobot(),
            tasks={"task0": RobotTask(task=task, policies={"p": DummyPolicy(0.0)})},
        )
        return RoboEnv(backend=DummyBackend(), robots=[robot])

    sweep = DRSweep(
        env_fn=env_fn,
        policy_fn=lambda _env: lambda obs: None,
        config=DRSweepConfig(
            n_seeds=int(seeds),
            n_episodes_per_seed=int(episodes),
            max_steps_per_episode=int(steps),
            levels=[RandomLevel.NONE, RandomLevel.LIGHT],
            object_position_ranges=[(0.0, 0.0)],
            physics_friction_ranges=[(1.0, 1.0)],
            sensor_noise_scales=[0.0, 1.0],
        ),
    )
    report = sweep.run()
    out = Path(output)
    report.write_json(out / "dr_sweep_report.json")
    if as_json:
        print_json(report.report(), pretty=pretty)
    else:
        print(str(out / "dr_sweep_report.json"))
    return 0


def _cmd_transfer_eval(
    *,
    dummy: bool,
    output: str,
    episodes: int,
    steps: int,
    as_json: bool,
    pretty: bool,
) -> int:
    _require_dummy(dummy, cmd="transfer-eval")
    from robodeploy.core.robot import Robot, RobotTask
    from robodeploy.core.transforms import GaussianNoiseTransform
    from robodeploy.env import RoboEnv
    from robodeploy.evaluation.transfer_metrics import TransferEvaluator
    from robodeploy.obs_pipeline import ObsPipeline
    from robodeploy.testing import DummyBackend, DummyPolicy, DummyRobot, DummyTask

    def sim_env_fn(seed: int):
        del seed
        robot = Robot(
            robot_id="robot0",
            description=DummyRobot(),
            tasks={"task0": RobotTask(task=DummyTask(), policies={"p": DummyPolicy(0.0)})},
        )
        return RoboEnv(backend=DummyBackend(), robots=[robot])

    def real_env_fn(seed: int):
        del seed
        robot = Robot(
            robot_id="robot0",
            description=DummyRobot(),
            tasks={"task0": RobotTask(task=DummyTask(), policies={"p": DummyPolicy(0.0)})},
            obs_pipeline=ObsPipeline([GaussianNoiseTransform(joint_pos_std=0.02, seed=0)]),
        )
        return RoboEnv(backend=DummyBackend(), robots=[robot])

    evaluator = TransferEvaluator(
        sim_env_fn=sim_env_fn,
        real_env_fn=real_env_fn,
        policy_fn=lambda _env: lambda obs: None,
        n_episodes=int(episodes),
        max_steps_per_episode=int(steps),
    )
    metrics = evaluator.run()
    path = evaluator.render_report(output)
    if as_json:
        print_json(metrics.to_dict(), pretty=pretty)
    else:
        print(str(path))
    return 0


def _lint_report_to_json(report) -> dict:  # noqa: ANN001
    return {
        "ok": report.ok,
        "issues": [
            {
                "level": i.level,
                "message": i.message,
                "file": i.file,
                "line": i.line,
                "suggested_fix": i.suggested_fix,
            }
            for i in report.issues
        ],
    }


def _cmd_scaffold(
    *,
    kind: str,
    name: str,
    template: str,
    output: str,
    force: bool,
    robot: str | None = None,
    backend: str | None = None,
    task: str | None = None,
    policy: str | None = None,
    dof: int | None = None,
    description_dir: str | None = None,
    preset: str | None = None,
) -> int:
    from robodeploy.scaffold import (
        scaffold_example,
        scaffold_policy,
        scaffold_preset,
        scaffold_robot,
        scaffold_sensor,
        scaffold_task,
    )

    if kind == "task":
        path = scaffold_task(name=name, template=template, output=output, force=force)  # type: ignore[arg-type]
    elif kind == "policy":
        path = scaffold_policy(name=name, template=template, output=output, force=force)  # type: ignore[arg-type]
    elif kind == "preset":
        path = scaffold_preset(
            name=name,
            robot=str(robot or "kuka"),
            backend=str(backend or "mujoco"),
            task=str(task or "pick_place"),
            policy=str(policy or "example_sensor_reach_pick"),
            template=template,  # type: ignore[arg-type]
            output=output,
            force=force,
        )
    elif kind == "robot":
        path = scaffold_robot(
            name=name,
            dof=int(dof or 6),
            description_dir=str(description_dir or "robodeploy/description"),
            force=force,
        )
    elif kind == "sensor":
        path = scaffold_sensor(
            name=name,
            backend=backend or "mujoco",  # type: ignore[arg-type]
            output=output,
            force=force,
        )
    elif kind == "example":
        if not preset:
            raise ValueError("--preset is required for scaffold example")
        path = scaffold_example(name=name, preset=preset, output=output, force=force)
    else:
        raise RuntimeError(f"Unknown scaffold kind: {kind}")
    print(path)
    return 0


def _cmd_lint(*, kind: str, path: str | None, check: str | None, as_json: bool) -> int:
    from robodeploy.linter import format_report, lint_all, lint_policy, lint_preset, lint_task

    if kind == "all":
        report = lint_all()
    elif kind == "task":
        report = lint_task(path or "")
    elif kind == "policy":
        report = lint_policy(path or "")
    elif kind == "preset":
        report = lint_preset(path or "", check=check)
    else:
        raise RuntimeError(f"Unknown lint kind: {kind}")

    if as_json:
        print_json(_lint_report_to_json(report), pretty=False)
    else:
        print(format_report(report))
    return 0 if report.ok else 1


def _cmd_train_bc(
    *,
    dataset: str,
    obs: str,
    action_dim: int | None,
    epochs: int,
    batch_size: int,
    lr: float,
    log_dir: str,
    log: str,
    out: str | None,
    dummy: bool,
    as_json: bool,
) -> int:
    from pathlib import Path as _Path

    from robodeploy.training.bc import train_bc
    from robodeploy.training.callbacks import TensorBoardCallback, WandbCallback
    from robodeploy.training.dataset import DemoDataset
    from robodeploy.training.gym_adapter import GymRoboEnv
    from robodeploy.training.trainer import TrainerConfig

    dataset_path = _Path(dataset)
    if not dataset_path.exists():
        if not dummy:
            raise FileNotFoundError(f"Dataset not found: {dataset_path}. Pass --dummy to synthesize demos.")
        env = _make_dummy_env()
        try:
            from robodeploy.dataset_export import export_recorded_episode

            export_recorded_episode(
                env,
                steps=100,
                path=dataset_path,
                action_fn=action_fn_for_mode("hold", env),
            )
        finally:
            close_quietly(env)

    if dataset_path.suffix.lower() in {".h5", ".hdf5"}:
        demo = DemoDataset.from_hdf5(dataset_path)
    else:
        demo = DemoDataset.from_jsonl(dataset_path)

    obs_keys = [k.strip() for k in obs.split(",") if k.strip()]
    cfg = TrainerConfig(
        epochs=int(epochs),
        batch_size=int(batch_size),
        lr=float(lr),
        log_dir=str(log_dir),
        checkpoint_interval=max(int(epochs), 1) * 1000,
    )
    eval_env = GymRoboEnv(_make_dummy_env(), max_episode_steps=50)
    callbacks = []
    if log == "wandb":
        callbacks.append(
            WandbCallback(
                project="robodeploy-bc",
                config={"dataset": str(dataset_path), "obs_keys": obs_keys},
            )
        )
    elif log == "tensorboard":
        callbacks.append(TensorBoardCallback(str(_Path(log_dir) / "tb")))
    train_bc(
        dataset=demo,
        obs_keys=obs_keys,
        action_dim=action_dim,
        config=cfg,
        eval_env=eval_env,
        callbacks=callbacks,
    )
    out_path = _Path(out or _Path(log_dir) / "bc_final.pt")
    close_quietly(eval_env.robo_env)
    payload = {
        "checkpoint": str(out_path),
        "epochs": int(epochs),
        "dataset": str(dataset_path),
        "obs_keys": obs_keys,
        "action_dim": demo.action_dim,
    }
    if as_json:
        print_json(payload)
    else:
        print(str(out_path))
    return 0


def _cmd_eval_checkpoint(
    *,
    checkpoint: str,
    episodes: int,
    dummy: bool,
    as_json: bool,
) -> int:
    _require_dummy(dummy, cmd="eval")
    import torch

    from robodeploy.training.bc import BCPolicyModule, bc_mse_loss
    from robodeploy.training.gym_adapter import GymRoboEnv
    from robodeploy.training.trainer import Trainer, TrainerConfig

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    env = _make_dummy_env()
    gym_env = GymRoboEnv(env, max_episode_steps=50)
    try:
        module = BCPolicyModule(obs_keys=["proprio"], action_dim=2, proprio_dim=6)
        module.load_state_dict(payload["policy"])
        trainer = Trainer(
            policy_module=module,
            dataset=_make_dummy_dataset_for_eval(),
            loss_fn=bc_mse_loss,
            config=TrainerConfig(),
            eval_env=gym_env,
        )
        metrics = trainer.evaluate(n_episodes=int(episodes))
    finally:
        close_quietly(env)
    if as_json:
        print_json({"checkpoint": checkpoint, "episodes": int(episodes), **metrics})
    else:
        print_json(metrics)
    return 0


def _make_gym_env_factory(*, dummy: bool, preset: str, max_episode_steps: int = 100):
    if dummy or not preset:
        from functools import partial

        from robodeploy.training.gym_register import robodeploy_dummy_gym_env_factory

        return partial(robodeploy_dummy_gym_env_factory, max_episode_steps=max_episode_steps)
    from functools import partial

    from robodeploy.training.gym_register import make_kuka_pick_mujoco

    return partial(make_kuka_pick_mujoco, max_episode_steps=max_episode_steps)


def _cmd_train_ppo(
    *,
    preset: str,
    n_envs: int,
    total_steps: int,
    rollout_steps: int,
    lr: float,
    log_dir: str,
    log: str,
    dummy: bool,
    as_json: bool,
) -> int:
    from functools import partial
    from pathlib import Path as _Path

    from robodeploy.training.callbacks import TensorBoardCallback, WandbCallback
    from robodeploy.training.parallel_vec_env import SubprocVecEnv
    from robodeploy.training.ppo import ActorCritic, PPOConfig, PPOTrainer

    use_dummy = dummy or not preset
    if not use_dummy and not preset:
        _require_dummy(True, cmd="train ppo")

    env_fn = _make_gym_env_factory(dummy=use_dummy, preset=preset, max_episode_steps=50)
    probe = env_fn()
    try:
        obs_dim = int(probe.observation_space["proprio"].shape[0])
        action_dim = int(probe.action_space.shape[0])
    finally:
        close_quietly(probe)

    cfg = PPOConfig(
        n_envs=int(n_envs),
        total_steps=int(total_steps),
        rollout_steps=int(rollout_steps),
        lr=float(lr),
        log_dir=str(log_dir),
    )
    callbacks = []
    if log == "wandb":
        callbacks.append(WandbCallback(project="robodeploy-ppo", config={"preset": preset, "dummy": use_dummy}))
    elif log == "tensorboard":
        callbacks.append(TensorBoardCallback(str(_Path(log_dir) / "tb")))

    vec = SubprocVecEnv([env_fn for _ in range(int(n_envs))])
    try:
        model = ActorCritic(obs_dim, action_dim)
        trainer = PPOTrainer(env=vec, model=model, config=cfg, callbacks=callbacks)
        metrics = trainer.fit()
        ckpt = _Path(log_dir) / "ppo_final.pt"
        import torch

        torch.save({"policy": model.state_dict(), "config": cfg, "metrics": metrics}, ckpt)
    finally:
        vec.close()

    payload = {
        "checkpoint": str(ckpt),
        "total_steps": int(total_steps),
        "n_envs": int(n_envs),
        "dummy": use_dummy,
        "preset": preset or None,
        "metrics": metrics,
    }
    if as_json:
        print_json(payload, pretty=False)
    else:
        print_json(metrics, pretty=False)
    return 0


def _cmd_convert_dataset(
    *,
    from_path: str,
    to_path: str,
    as_json: bool,
    lerobot_root: str | None = None,
) -> int:
    from pathlib import Path as _Path

    from robodeploy.training.dataset import DemoDataset

    src = str(from_path)
    dst = _Path(to_path)
    if src.startswith("lerobot://"):
        repo_id = src.split("://", 1)[1].strip("/")
        dataset = DemoDataset.from_lerobot(repo_id, root=lerobot_root)
    elif dst.suffix.lower() in {".h5", ".hdf5"} and _Path(src).suffix.lower() in {".h5", ".hdf5"}:
        dataset = DemoDataset.from_hdf5(src)
    elif _Path(src).suffix.lower() in {".h5", ".hdf5"}:
        dataset = DemoDataset.from_hdf5(src)
    elif "robomimic" in src.lower() or (_Path(src).suffix.lower() in {".h5", ".hdf5"} and _Path(src).exists()):
        try:
            dataset = DemoDataset.from_robomimic(src)
        except (ValueError, KeyError):
            dataset = DemoDataset.from_jsonl(src)
    else:
        dataset = DemoDataset.from_teleop_jsonl(src)

    if dst.suffix.lower() in {".h5", ".hdf5"}:
        dataset.to_hdf5(dst)
    else:
        dataset.to_jsonl(dst)

    payload = {"from": src, "to": str(dst), "frames": len(dataset)}
    if as_json:
        print_json(payload, pretty=False)
    else:
        print(f"Wrote {len(dataset)} frames to {dst}")
    return 0


def _make_dummy_dataset_for_eval():
    from robodeploy.demo_recording import DemoFrame
    from robodeploy.training.dataset import DemoDataset

    frame = DemoFrame(
        observation={
            "joint_positions": [0.0, 0.0],
            "joint_velocities": [0.0, 0.0],
            "joint_torques": [0.0, 0.0],
        },
        action={"joint_positions": [0.0, 0.0]},
        reward=0.0,
        done=False,
    )
    return DemoDataset([frame])


def _cmd_eval(
    *,
    benchmark: str,
    policy: str,
    backend: str,
    episodes: int,
    seed: int,
    max_steps: int,
    output: str,
    benchmarks_root: str,
    sweep_backends: bool,
    parallel: bool,
    workers: int,
    record_videos: bool,
    video_dir: str,
    html_output: str,
    baseline_report: str,
    as_json: bool,
    pretty: bool,
) -> int:
    from robodeploy.evaluation.runner import run_eval

    report = run_eval(
        benchmark=benchmark,
        policy=policy,
        backend=backend,
        episodes=int(episodes),
        base_seed=int(seed),
        max_steps=int(max_steps) if int(max_steps) > 0 else None,
        parallel=bool(parallel),
        n_workers=int(workers),
        sweep_backends=bool(sweep_backends),
        benchmarks_root=benchmarks_root or None,
        record_videos=bool(record_videos),
        video_dir=video_dir or None,
        html_output=html_output or None,
        baseline_report=baseline_report or None,
    )
    if output:
        report.save(Path(output))
    if html_output and not output:
        report.render_html(html_output)
    if as_json or (not output and not html_output):
        print_json(report.to_json(), pretty=pretty)
    return 0


def _cmd_eval_compare(*, report_a: str, report_b: str, output: str) -> int:
    import json

    from robodeploy.evaluation.render import render_comparison

    a = json.loads(Path(report_a).read_text(encoding="utf-8"))
    b = json.loads(Path(report_b).read_text(encoding="utf-8"))
    render_comparison(a, b, output)
    print(f"Wrote comparison report to {output}")
    return 0


def _cmd_leaderboard_submit(
    *,
    report: str,
    benchmark: str,
    author: str,
    checkpoint: str,
    benchmarks_root: str,
) -> int:
    from robodeploy.evaluation.leaderboard import submit_score

    out = submit_score(
        report,
        benchmark=benchmark,
        author=author,
        benchmarks_root_path=benchmarks_root or None,
        policy_checkpoint=checkpoint or None,
    )
    print(f"Wrote leaderboard submission to {out}")
    return 0


def _cmd_leaderboard_show(*, suite: str, benchmarks_root: str, as_json: bool) -> int:
    from robodeploy.evaluation.leaderboard import show_leaderboard

    payload = show_leaderboard(suite, benchmarks_root_path=benchmarks_root or None, as_json=as_json)
    if as_json:
        print_json(payload, pretty=False)
    else:
        print(payload)
    return 0


def _cmd_list_benchmarks(*, benchmarks_root: str, as_json: bool, pretty: bool) -> int:
    from robodeploy.evaluation.runner import list_benchmarks

    payload = list_benchmarks(benchmarks_root=benchmarks_root or None)
    if as_json:
        print_json(payload, pretty=pretty)
        return 0
    for suite_name, suite in payload.items():
        print(f"{suite_name} (v{suite['version']}):")
        for task in suite["tasks"]:
            backends = ", ".join(task["backends"]) or "(none)"
            print(f"  - {task['name']} tier={task['tier']} backends=[{backends}]")
    return 0


def _cmd_serve_policy_with_modules(
    *,
    policy: str,
    host: str,
    port: int,
    transport: str,
    quiet: bool,
    custom_modules: list[str],
    checkpoint: str | None = None,
    model_spec: str | None = None,
) -> int:
    _import_custom_modules(custom_modules)
    return _cmd_serve_policy(
        policy=policy,
        host=host,
        port=port,
        transport=transport,
        quiet=quiet,
        checkpoint=checkpoint,
        model_spec=model_spec,
    )


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    cmd = str(args.cmd)

    if cmd == "list-registry":
        if bool(args.pretty) and not bool(args.json):
            raise ValueError("--pretty requires --json.")
        return _cmd_list_registry(
            discover=bool(args.discover),
            custom_modules=list(args.custom_module or []),
            builtins=bool(args.builtins),
            as_json=bool(args.json),
            pretty=bool(args.pretty),
        )
    if cmd == "export-episode":
        if bool(args.pretty) and not bool(args.json):
            raise ValueError("--pretty requires --json.")
        return _cmd_export_episode(
            steps=int(args.steps),
            out=str(args.out),
            fmt=str(args.format),
            dummy=bool(args.dummy),
            action_mode=str(args.action),
            as_json=bool(args.json),
            pretty=bool(args.pretty),
        )
    if cmd == "run-episode":
        return _cmd_run_episode(
            steps=int(args.steps),
            dummy=bool(args.dummy),
            action_mode=str(args.action),
            pretty=bool(args.pretty),
            as_json=bool(args.json),
        )
    if cmd == "serve-policy":
        return _cmd_serve_policy_with_modules(
            policy=str(args.policy),
            host=str(args.host),
            port=int(args.port),
            transport=str(args.transport),
            quiet=bool(args.quiet),
            custom_modules=list(args.custom_module or []),
            checkpoint=getattr(args, "checkpoint", None),
            model_spec=getattr(args, "model_spec", None),
        )
    if cmd == "models":
        if str(args.models_cmd) == "list":
            if bool(args.pretty) and not bool(args.json):
                raise ValueError("--pretty requires --json.")
            return _cmd_models_list(as_json=bool(args.json), pretty=bool(args.pretty))
        if str(args.models_cmd) == "download":
            return _cmd_models_download(name=str(args.name), as_json=bool(args.json))
        raise RuntimeError(f"Unknown models subcommand: {args.models_cmd}")
    if cmd == "dr-sweep":
        if bool(args.pretty) and not bool(args.json):
            raise ValueError("--pretty requires --json.")
        return _cmd_dr_sweep(
            dummy=bool(args.dummy),
            output=str(args.output),
            seeds=int(args.seeds),
            episodes=int(args.episodes),
            steps=int(args.steps),
            as_json=bool(args.json),
            pretty=bool(args.pretty),
        )
    if cmd == "transfer-eval":
        if bool(args.pretty) and not bool(args.json):
            raise ValueError("--pretty requires --json.")
        return _cmd_transfer_eval(
            dummy=bool(args.dummy),
            output=str(args.output),
            episodes=int(args.episodes),
            steps=int(args.steps),
            as_json=bool(args.json),
            pretty=bool(args.pretty),
        )
    if cmd == "calibrate":
        from robodeploy.calibration import cli as cal_cli

        subcmd = str(args.calibrate_cmd)
        if subcmd == "kinematic":
            result = cal_cli.cmd_calibrate_kinematic(
                robot=str(args.robot),
                port=str(args.port),
                out=args.out,
                as_json=True,
            )
        elif subcmd == "extrinsic":
            result = cal_cli.cmd_calibrate_extrinsic(
                camera=str(args.camera),
                pattern=str(args.pattern),
                board=str(args.board),
                robot_id=str(args.robot_id),
                as_json=True,
            )
        elif subcmd == "handeye":
            result = cal_cli.cmd_calibrate_handeye(
                robot=str(args.robot),
                pattern=str(args.pattern),
                method=str(args.method),
                as_json=True,
            )
        elif subcmd == "system-id":
            result = cal_cli.cmd_calibrate_system_id(
                robot=str(args.robot),
                joint=int(args.joint),
                dummy=bool(args.dummy),
                as_json=True,
            )
        else:
            raise RuntimeError(f"Unknown calibrate subcommand: {subcmd}")
        print_json(result, pretty=False)
        return 0
    if cmd == "scaffold":
        return _cmd_scaffold(
            kind=str(args.scaffold_kind),
            name=str(args.name),
            template=str(getattr(args, "template", "")),
            output=str(getattr(args, "output", "")),
            force=bool(args.force),
            robot=getattr(args, "robot", None),
            backend=getattr(args, "backend", None),
            task=getattr(args, "task", None),
            policy=getattr(args, "policy", None),
            dof=getattr(args, "dof", None),
            description_dir=getattr(args, "description_dir", None),
            preset=getattr(args, "preset", None),
        )
    if cmd == "doctor":
        if bool(args.pretty) and not bool(args.json):
            raise ValueError("--pretty requires --json.")
        from robodeploy.cli_doctor import cmd_doctor

        return cmd_doctor(as_json=bool(args.json), pretty=bool(args.pretty))
    if cmd == "lint":
        return _cmd_lint(
            kind=str(args.lint_kind),
            path=getattr(args, "path", None),
            check=getattr(args, "check", None),
            as_json=bool(getattr(args, "json", False)),
        )
    if cmd == "scene":
        from robodeploy.cli_scene import cmd_scene_inspect, cmd_scene_validate

        scene_cmd = str(args.scene_cmd)
        if scene_cmd == "validate":
            return cmd_scene_validate(
                scene=str(args.scene),
                backend=getattr(args, "backend", None),
                as_json=bool(args.json),
                pretty=bool(args.pretty),
            )
        if scene_cmd == "inspect":
            return cmd_scene_inspect(
                scene=str(args.scene),
                backend=getattr(args, "backend", None),
                as_json=bool(args.json),
                pretty=bool(args.pretty),
            )
        raise RuntimeError(f"Unknown scene command: {scene_cmd}")
    if cmd == "config":
        from robodeploy.cli_config import (
            cmd_config_diff,
            cmd_config_resolve,
            cmd_config_show,
            cmd_config_validate,
        )

        config_cmd = str(args.config_cmd)
        presets_file = Path(args.presets_file) if getattr(args, "presets_file", None) else None
        if config_cmd == "show":
            return cmd_config_show(
                preset=str(args.preset),
                presets_file=presets_file,
                as_json=bool(args.json),
                pretty=bool(args.pretty),
            )
        if config_cmd == "resolve":
            return cmd_config_resolve(
                preset=str(args.preset),
                presets_file=presets_file,
                as_json=bool(args.json),
                pretty=bool(args.pretty),
            )
        if config_cmd == "validate":
            return cmd_config_validate(
                presets_file=str(args.path),
                as_json=bool(args.json),
                pretty=bool(args.pretty),
            )
        if config_cmd == "diff":
            return cmd_config_diff(
                preset_a=str(args.preset_a),
                preset_b=str(args.preset_b),
                presets_file=presets_file,
                as_json=bool(args.json),
                pretty=bool(args.pretty),
            )
        raise RuntimeError(f"Unknown config command: {config_cmd}")
    if cmd == "assets":
        from robodeploy.cli_assets import cmd_assets_info, cmd_assets_list, cmd_assets_resolve

        assets_cmd = str(args.assets_cmd)
        if assets_cmd == "verify":
            if bool(args.pretty) and not bool(args.json):
                raise ValueError("--pretty requires --json.")
            return _cmd_assets_verify(as_json=bool(args.json), pretty=bool(args.pretty))
        if assets_cmd == "list":
            return cmd_assets_list(
                robot=bool(args.robot),
                mesh=bool(args.mesh),
                mjcf=bool(args.mjcf),
                as_json=bool(args.json),
                pretty=bool(args.pretty),
            )
        if assets_cmd == "resolve":
            return cmd_assets_resolve(
                name=str(args.name),
                backend=str(args.backend),
                as_json=bool(args.json),
                pretty=bool(args.pretty),
            )
        if assets_cmd == "info":
            return cmd_assets_info(
                name=str(args.name),
                as_json=bool(args.json),
                pretty=bool(args.pretty),
            )
        raise RuntimeError(f"Unknown assets command: {assets_cmd}")
    if cmd == "eval":
        if bool(args.pretty) and not bool(args.json) and not str(args.output):
            raise ValueError("--pretty requires --json when --output is not set.")
        return _cmd_eval(
            benchmark=str(args.benchmark),
            policy=str(args.policy),
            backend=str(args.backend),
            episodes=int(args.episodes),
            seed=int(args.seed),
            max_steps=int(args.max_steps),
            output=str(args.output),
            benchmarks_root=str(args.benchmarks_root),
            sweep_backends=bool(args.sweep_backends),
            parallel=bool(args.parallel),
            workers=int(args.workers),
            record_videos=bool(args.record_videos),
            video_dir=str(args.video_dir),
            html_output=str(args.html),
            baseline_report=str(args.baseline),
            as_json=bool(args.json) or (not bool(args.output) and not bool(args.html)),
            pretty=bool(args.pretty),
        )
    if cmd == "eval-compare":
        return _cmd_eval_compare(
            report_a=str(args.report_a),
            report_b=str(args.report_b),
            output=str(args.output),
        )
    if cmd == "leaderboard":
        if str(args.leaderboard_cmd) == "submit":
            return _cmd_leaderboard_submit(
                report=str(args.report),
                benchmark=str(args.benchmark),
                author=str(args.author),
                checkpoint=str(args.checkpoint),
                benchmarks_root=str(args.benchmarks_root),
            )
        if str(args.leaderboard_cmd) == "show":
            return _cmd_leaderboard_show(
                suite=str(args.suite),
                benchmarks_root=str(args.benchmarks_root),
                as_json=bool(args.json),
            )
        raise RuntimeError(f"Unknown leaderboard subcommand: {args.leaderboard_cmd}")
    if cmd == "list-benchmarks":
        if bool(args.pretty) and not bool(args.json):
            raise ValueError("--pretty requires --json.")
        return _cmd_list_benchmarks(
            benchmarks_root=str(args.benchmarks_root),
            as_json=bool(args.json),
            pretty=bool(args.pretty),
        )
    if cmd == "train":
        if str(args.train_cmd) == "bc":
            return _cmd_train_bc(
                dataset=str(args.dataset),
                obs=str(args.obs),
                action_dim=args.action_dim,
                epochs=int(args.epochs),
                batch_size=int(args.batch_size),
                lr=float(args.lr),
                log_dir=str(args.log_dir),
                log=str(args.log),
                out=args.out,
                dummy=bool(args.dummy),
                as_json=bool(args.json),
            )
        if str(args.train_cmd) == "eval":
            return _cmd_eval_checkpoint(
                checkpoint=str(args.checkpoint),
                episodes=int(args.episodes),
                dummy=bool(args.dummy),
                as_json=bool(args.json),
            )
        if str(args.train_cmd) == "ppo":
            return _cmd_train_ppo(
                preset=str(args.preset),
                n_envs=int(args.n_envs),
                total_steps=int(args.total_steps),
                rollout_steps=int(args.rollout_steps),
                lr=float(args.lr),
                log_dir=str(args.log_dir),
                log=str(args.log),
                dummy=bool(args.dummy) or not str(args.preset),
                as_json=bool(args.json),
            )
        raise RuntimeError(f"Unknown train subcommand: {args.train_cmd}")
    if cmd == "convert-dataset":
        return _cmd_convert_dataset(
            from_path=str(args.from_path),
            to_path=str(args.to_path),
            as_json=bool(args.json),
            lerobot_root=getattr(args, "lerobot_root", None),
        )
    if cmd == "safety":
        from robodeploy.cli_safety import cmd_safety_check, cmd_safety_status, cmd_safety_test

        safety_cmd = str(args.safety_cmd)
        presets_file = Path(args.presets_file) if getattr(args, "presets_file", None) else None
        if safety_cmd == "check":
            joint_limits = Path(args.joint_limits) if getattr(args, "joint_limits", None) else None
            return cmd_safety_check(
                preset=getattr(args, "preset", None),
                robot=getattr(args, "robot", None),
                joint_limits=joint_limits,
                presets_file=presets_file,
                as_json=bool(args.json),
                pretty=bool(args.pretty),
            )
        if safety_cmd == "test":
            return cmd_safety_test(
                preset=getattr(args, "preset", None),
                inject=list(getattr(args, "inject", []) or []),
                steps=int(args.steps),
                presets_file=presets_file,
                as_json=bool(args.json),
                pretty=bool(args.pretty),
            )
        if safety_cmd == "status":
            return cmd_safety_status(as_json=bool(args.json), pretty=bool(args.pretty))
        raise RuntimeError(f"Unknown safety command: {safety_cmd}")
    if cmd == "teleop":
        from robodeploy.cli_teleop import cmd_teleop

        return cmd_teleop(
            preset=str(args.preset),
            presets_file=getattr(args, "presets_file", None),
            device=str(args.device),
            record=str(args.record),
            fmt=str(args.format),
            max_steps=int(args.max_steps),
            start_recording=bool(args.start_recording),
            as_json=bool(args.json),
        )
    if cmd in {"logs", "replay", "manifest", "snapshot"}:
        from robodeploy.cli_observability import dispatch_observability

        return dispatch_observability(cmd, args)

    raise RuntimeError(f"Unknown command: {cmd}")


if __name__ == "__main__":
    sys.exit(main())
