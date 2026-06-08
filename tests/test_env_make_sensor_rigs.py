"""RoboEnv.make() accepts sensor_rigs like from_config."""

from robodeploy.env import RoboEnv


def test_make_accepts_sensor_rigs_parameter():
    import inspect

    sig = inspect.signature(RoboEnv.make)
    assert "sensor_rigs" in sig.parameters
    assert "custom_modules" in sig.parameters
