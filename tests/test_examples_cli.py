from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class ExamplesCliTests(unittest.TestCase):
    def test_list_presets_prints_names(self):
        from examples.cli import main

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(["list-presets"])
        self.assertEqual(code, 0)
        out = buf.getvalue()
        self.assertIn("kuka_pick_mujoco", out)
        self.assertIn("mujoco_showcase_kuka", out)

    def test_list_presets_json_pretty_is_parseable(self):
        import json as _json

        from examples.cli import main

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(["list-presets", "--json", "--pretty"])
        self.assertEqual(code, 0)
        payload = _json.loads(buf.getvalue())
        self.assertIsInstance(payload, list)

    def test_run_episode_dummy_prints_json_summary(self):
        import json as _json

        from examples.cli import main

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(["run-episode", "--preset", "kuka_pick_mujoco", "--dummy", "--steps", "2"])
        self.assertEqual(code, 0)
        payload = _json.loads(buf.getvalue().strip())
        self.assertIn("episode_id", payload)
        self.assertIn("reward", payload)

    def test_run_episode_dummy_action_mode_sinusoid_is_ok(self):
        import json as _json

        from examples.cli import main

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(
                [
                    "run-episode",
                    "--preset",
                    "kuka_pick_mujoco",
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

    def test_export_episode_dummy_writes_path(self):
        import tempfile

        from examples.cli import main

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
