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


if __name__ == "__main__":
    unittest.main()

