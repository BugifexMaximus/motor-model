"""Tests for the brushed motor model."""

from motor_model import BrushedMotorModel


def test_zero_voltage_keeps_motor_idle():
    model = BrushedMotorModel()
    result = model.simulate(0.0, duration=0.05, dt=1e-4)

    assert all(abs(i) < 1e-9 for i in result.current)
    assert all(abs(w) < 1e-9 for w in result.speed)
    assert all(abs(theta) < 1e-9 for theta in result.position)


def test_static_friction_holds_under_small_voltage():
    model = BrushedMotorModel()
    # Apply a voltage below the static friction torque threshold.
    result = model.simulate(0.3, duration=0.1, dt=1e-4)

    assert max(abs(w) for w in result.speed) < 1e-6
    assert max(abs(theta) for theta in result.position) < 1e-6


def test_step_voltage_overcomes_friction_and_spins_up():
    model = BrushedMotorModel()
    result = model.simulate(5.0, duration=0.2, dt=1e-4)

    assert result.speed[-1] > 0.0
    # Current should decrease as speed (and back EMF) build up.
    assert result.current[0] > result.current[-1]
    # Position should have advanced significantly.
    assert result.position[-1] > 0.1
