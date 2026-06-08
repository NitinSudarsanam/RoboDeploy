from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class CliTests(unittest.TestCase):
    def test_list_registry_without_builtins_is_sparse(self):
        from robodeploy.cli import main

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(["list-registry"])
        self.assertEqual(code, 0)
        out = buf.getvalue()
        self.assertIn("backends:", out)
        self.assertIn("robots:", out)

    def test_list_registry_with_builtins_includes_backends(self):
        from robodeploy.cli import main

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(["list-registry", "--builtins"])
        self.assertEqual(code, 0)
        out = buf.getvalue()
        self.assertIn("backends:", out)
        self.assertIn("mujoco", out)

    def test_list_registry_includes_example_tasks_after_use(self):
        from robodeploy.cli import main
        from robodeploy.core.registry import use

        use("examples.tasks")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(["list-registry"])
        self.assertEqual(code, 0)
        out = buf.getvalue()
        self.assertIn("pick_place", out)

    def test_list_registry_discover_flag_is_safe(self):
        from robodeploy.cli import main

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(["list-registry", "--discover"])
        self.assertEqual(code, 0)

    def test_list_registry_custom_module_registers_user_components(self):
        from robodeploy.cli import main

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(["list-registry", "--custom-module", "examples.user_kuka_sinusoid.components"])
        self.assertEqual(code, 0)
        out = buf.getvalue()
        self.assertIn("user_kuka", out)
        self.assertIn("user_kuka_sinusoid", out)
        self.assertIn("user_sinusoid", out)

    def test_list_registry_json_pretty_is_parseable(self):
        import json as _json

        from robodeploy.cli import main

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(["list-registry", "--json", "--pretty"])
        self.assertEqual(code, 0)
        payload = _json.loads(buf.getvalue())
        self.assertIsInstance(payload, dict)

    def test_export_episode_dummy_writes_path(self):
        from pathlib import Path
        import tempfile

        from robodeploy.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "demo.jsonl"
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                code = main(
                    [
                        "export-episode",
                        "--dummy",
                        "--steps",
                        "2",
                        "--out",
                        str(out_path),
                        "--format",
                        "jsonl",
                    ]
                )
            self.assertEqual(code, 0)
            self.assertTrue(out_path.exists())

    def test_export_episode_json_pretty_is_parseable(self):
        import json as _json
        from pathlib import Path
        import tempfile

        from robodeploy.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "demo.jsonl"
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                code = main(
                    [
                        "export-episode",
                        "--dummy",
                        "--steps",
                        "2",
                        "--out",
                        str(out_path),
                        "--format",
                        "jsonl",
                        "--json",
                        "--pretty",
                    ]
                )
            self.assertEqual(code, 0)
            payload = _json.loads(buf.getvalue())
            self.assertEqual(payload["out"], str(out_path))

    def test_run_episode_dummy_prints_json_summary(self):
        import json as _json

        from robodeploy.cli import main

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(["run-episode", "--dummy", "--steps", "2"])
        self.assertEqual(code, 0)
        payload = _json.loads(buf.getvalue().strip())
        self.assertIn("episode_id", payload)
        self.assertIn("reward", payload)

    def test_run_episode_pretty_is_parseable(self):
        import json as _json

        from robodeploy.cli import main

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(["run-episode", "--dummy", "--steps", "1", "--pretty"])
        self.assertEqual(code, 0)
        payload = _json.loads(buf.getvalue())
        self.assertIn("episode_id", payload)

    def test_run_episode_json_pretty_is_parseable(self):
        import json as _json

        from robodeploy.cli import main

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(["run-episode", "--dummy", "--steps", "1", "--json", "--pretty"])
        self.assertEqual(code, 0)
        payload = _json.loads(buf.getvalue())
        self.assertIn("info", payload)
        self.assertIn("episode_id", payload["info"])

    def test_run_episode_dummy_action_mode_sinusoid_is_ok(self):
        import json as _json

        from robodeploy.cli import main

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(
                [
                    "run-episode",
                    "--dummy",
                    "--steps",
                    "3",
                    "--action",
                    "sinusoid",
                ]
            )
        self.assertEqual(code, 0)
        payload = _json.loads(buf.getvalue().strip())
        self.assertIn("episode_id", payload)

    def test_run_episode_without_dummy_raises(self):
        from robodeploy.cli import main

        with self.assertRaises(ValueError) as ctx:
            main(["run-episode", "--steps", "1"])
        self.assertIn("examples.cli", str(ctx.exception))


class CliScaffoldLintTests(unittest.TestCase):
    def test_scaffold_task_writes_compilable_file(self):
        import tempfile
        from pathlib import Path

        from robodeploy.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "kitchen_pick.py"
            code = main(
                [
                    "scaffold",
                    "task",
                    "--name",
                    "kitchen_pick",
                    "--template",
                    "pick_place",
                    "--output",
                    str(out),
                ]
            )
            self.assertEqual(code, 0)
            self.assertTrue(out.is_file())
            src = out.read_text(encoding="utf-8")
            self.assertIn("@register_task", src)
            self.assertIn("SceneBuilder", src)
            compile(src, str(out), "exec")

    def test_scaffold_policy_reach_dsl_yaml(self):
        import tempfile
        from pathlib import Path

        from robodeploy.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "custom_reach.yaml"
            code = main(
                [
                    "scaffold",
                    "policy",
                    "--name",
                    "custom_reach",
                    "--template",
                    "reach_dsl",
                    "--output",
                    str(out),
                ]
            )
            self.assertEqual(code, 0)
            self.assertIn("phases:", out.read_text(encoding="utf-8"))

    def test_lint_task_pick_place_no_errors(self):
        from pathlib import Path

        from robodeploy.cli import main

        repo = Path(__file__).resolve().parents[1]
        task_path = repo / "examples" / "tasks" / "pick_place.py"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(["lint", "task", str(task_path)])
        self.assertEqual(code, 0)
        out = buf.getvalue()
        self.assertNotIn("[ERROR]", out)

    def test_lint_task_missing_method_fails(self):
        import tempfile
        from pathlib import Path

        from robodeploy.cli import main

        bad_src = (
            "from robodeploy.core.registry import register_task\n"
            "from robodeploy.tasks.base import TaskBase\n"
            "@register_task('bad')\n"
            "class Bad(TaskBase):\n"
            "    pass\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad_task.py"
            path.write_text(bad_src, encoding="utf-8")
            code = main(["lint", "task", str(path)])
        self.assertEqual(code, 1)

    def test_lint_preset_kuka_pick_exists(self):
        from pathlib import Path

        from robodeploy.cli import main

        repo = Path(__file__).resolve().parents[1]
        presets = repo / "examples" / "config" / "presets.yaml"
        code = main(["lint", "preset", str(presets), "--check", "kuka_pick_mujoco"])
        self.assertEqual(code, 0)


class CliSceneConfigAssetsTests(unittest.TestCase):
    def test_scene_validate_good_scene(self):
        from pathlib import Path

        from robodeploy.cli import main

        scene = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "pick_place_scene.yaml"
        code = main(["scene", "validate", str(scene), "--backend", "mujoco"])
        self.assertEqual(code, 0)

    def test_scene_validate_bad_scene_exits_nonzero(self):
        from pathlib import Path

        from robodeploy.cli import main

        scene = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "bad_scene.yaml"
        code = main(["scene", "validate", str(scene), "--backend", "mujoco"])
        self.assertEqual(code, 1)

    def test_config_resolve_kuka_pick_mujoco(self):
        import json as _json
        from pathlib import Path

        from robodeploy.cli import main

        repo = Path(__file__).resolve().parents[1]
        presets = repo / "examples" / "config" / "presets.yaml"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(
                [
                    "config",
                    "resolve",
                    "--preset",
                    "kuka_pick_mujoco",
                    "--presets-file",
                    str(presets),
                    "--json",
                ]
            )
        self.assertEqual(code, 0)
        payload = _json.loads(buf.getvalue())
        self.assertEqual(payload["robot"], "kuka")
        self.assertEqual(payload["task"], "pick_place")

    def test_config_validate_presets(self):
        from pathlib import Path

        from robodeploy.cli import main

        presets = Path(__file__).resolve().parents[1] / "examples" / "config" / "presets.yaml"
        code = main(["config", "validate", str(presets)])
        self.assertEqual(code, 0)

    def test_assets_list_robots(self):
        import json as _json

        from robodeploy.cli import main

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(["assets", "list", "--robot", "--json"])
        self.assertEqual(code, 0)
        payload = _json.loads(buf.getvalue())
        names = {item["name"] for item in payload}
        self.assertIn("kuka", names)

    def test_scene_builder_validate_integration(self):
        from robodeploy.scene_builder import SceneBuilder

        spec = (
            SceneBuilder()
            .add_table(height=0.4)
            .add_box("source", size=(0.03, 0.03, 0.03), pos=(0.55, 0.0, 0.41))
            .add_target("target", pos=(0.65, 0.2, 0.41))
            .validate(backend="mujoco")
            .build_spec()
        )
        self.assertGreaterEqual(len(spec.props), 2)

    def test_doctor_runs_and_reports_python(self):
        from robodeploy.cli import main

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(["doctor"])
        out = buf.getvalue()
        self.assertIn("RoboDeploy Doctor", out)
        self.assertIn("Python", out)
        self.assertIn(code, (0, 1))

    def test_doctor_json_is_parseable(self):
        import json as _json

        from robodeploy.cli import main

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(["doctor", "--json"])
        self.assertIn(code, (0, 1))
        payload = _json.loads(buf.getvalue())
        self.assertIn("checks", payload)
        self.assertIn("version", payload)

    def test_scaffold_preset_writes_yaml(self):
        import tempfile
        from pathlib import Path

        from robodeploy.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "my_pick.yaml"
            code = main(
                [
                    "scaffold",
                    "preset",
                    "--name",
                    "my_pick",
                    "--robot",
                    "kuka",
                    "--template",
                    "manipulate",
                    "--output",
                    str(out),
                ]
            )
            self.assertEqual(code, 0)
            text = out.read_text(encoding="utf-8")
            self.assertIn("my_pick:", text)
            self.assertIn("<<: *manipulate_pick", text)
            self.assertIn("my_pick:", text)

    def test_scaffold_robot_writes_description_package(self):
        import tempfile
        from pathlib import Path

        from robodeploy.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            desc_root = Path(tmp) / "description"
            code = main(
                [
                    "scaffold",
                    "robot",
                    "--name",
                    "myarm",
                    "--dof",
                    "6",
                    "--description-dir",
                    str(desc_root),
                ]
            )
            self.assertEqual(code, 0)
            pkg = desc_root / "myarm"
            self.assertTrue((pkg / "description.py").is_file())
            self.assertTrue((pkg / "assets" / "mjcf" / "myarm.xml").is_file())
            src = (pkg / "description.py").read_text(encoding="utf-8")
            self.assertIn("@register_robot", src)
            compile(src, str(pkg / "description.py"), "exec")

    def test_scaffold_sensor_mujoco_writes_module(self):
        import tempfile
        from pathlib import Path

        from robodeploy.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "mujoco_pressure.py"
            code = main(
                [
                    "scaffold",
                    "sensor",
                    "--name",
                    "pressure",
                    "--backend",
                    "mujoco",
                    "--output",
                    str(out),
                ]
            )
            self.assertEqual(code, 0)
            src = out.read_text(encoding="utf-8")
            self.assertIn("@register_sensor", src)
            compile(src, str(out), "exec")

    def test_scaffold_example_writes_run_script(self):
        import tempfile
        from pathlib import Path

        from robodeploy.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run.py"
            code = main(
                [
                    "scaffold",
                    "example",
                    "--name",
                    "my_demo",
                    "--preset",
                    "kuka_pick_mujoco",
                    "--output",
                    str(out),
                ]
            )
            self.assertEqual(code, 0)
            src = out.read_text(encoding="utf-8")
            self.assertIn("kuka_pick_mujoco", src)
            self.assertIn("env_from_preset", src)
            compile(src, str(out), "exec")

    def test_config_resolve_without_presets_file(self):
        import json as _json
        import io

        from robodeploy.cli import main

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(["config", "resolve", "--preset", "kuka_pick_mujoco", "--json"])
        self.assertEqual(code, 0)
        payload = _json.loads(buf.getvalue())
        self.assertEqual(payload["robot"], "kuka")
        self.assertEqual(payload["task"], "pick_place")


if __name__ == "__main__":
    unittest.main()
