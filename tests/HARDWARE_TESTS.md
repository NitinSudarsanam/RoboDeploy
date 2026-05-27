# Hardware integration tests

Pytest-only tests (e.g. `test_so101_real.py`) require real devices or ROS2 graphs.

- Default CI runs `python -m unittest discover` and does **not** execute these files.
- Local run: set `ROBODEPLOY_SO101_PORT` and run `pytest tests/test_so101_real.py -m hardware`.
