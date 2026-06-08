from __future__ import annotations

import contextlib
import io
import json
import unittest
from unittest import mock


class LearnedCliTests(unittest.TestCase):
    def test_models_list_prints_known_aliases(self):
        from robodeploy.cli import main

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(["models", "list"])
        self.assertEqual(code, 0)
        out = buf.getvalue()
        self.assertIn("openvla-7b", out)
        self.assertIn("pi0-base", out)
        self.assertIn("octo-base", out)

    def test_models_list_json(self):
        from robodeploy.cli import main

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(["models", "list", "--json"])
        self.assertEqual(code, 0)
        payload = json.loads(buf.getvalue())
        self.assertIn("openvla-7b", payload["models"])

    def test_models_download_mocked(self):
        from robodeploy.cli import main

        with mock.patch(
            "robodeploy.policies.learned.hf_hub.HFModelRegistry.download",
            return_value="C:/cache/openvla/model.pt",
        ):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                code = main(["models", "download", "openvla-7b"])
        self.assertEqual(code, 0)
        self.assertIn("openvla", buf.getvalue())

    def test_serve_policy_resolves_stub_without_binding(self):
        from robodeploy.cli import _resolve_serve_policy

        policy = _resolve_serve_policy(policy="vla_stub", checkpoint=None, model_spec_path=None)
        self.assertEqual(policy.__class__.__name__, "VLAPolicy")


if __name__ == "__main__":
    unittest.main()
