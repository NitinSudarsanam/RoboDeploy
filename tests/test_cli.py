from __future__ import annotations

import contextlib
import io
import unittest


class CliTests(unittest.TestCase):
    def test_list_presets_prints_names(self):
        from robodeploy.cli import main

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(["list-presets"])
        self.assertEqual(code, 0)
        out = buf.getvalue()
        self.assertIn("kuka_pick_mujoco", out)

    def test_list_registry_without_builtins_is_sparse(self):
        from robodeploy.cli import main

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(["list-registry"])
        self.assertEqual(code, 0)
        out = buf.getvalue()
        self.assertIn("backends:", out)
        self.assertIn("robots:", out)

    def test_list_registry_with_builtins_includes_tasks(self):
        from robodeploy.cli import main

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(["list-registry", "--builtins"])
        self.assertEqual(code, 0)
        out = buf.getvalue()
        self.assertIn("tasks:", out)
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
                        "--preset",
                        "kuka_pick_mujoco",
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


if __name__ == "__main__":
    unittest.main()

