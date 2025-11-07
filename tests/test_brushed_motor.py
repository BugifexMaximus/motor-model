"""Tests for the brushed motor model."""

import random

from motor_model import BrushedMotorModel


def test_zero_voltage_keeps_motor_idle():
    model = BrushedMotorModel()
    result = model.simulate(0.0, duration=0.05, dt=1e-4)

    assert all(abs(i) < 1e-9 for i in result.current)
    assert all(abs(w) < 1e-9 for w in result.speed)
    assert all(abs(theta) < 1e-9 for theta in result.position)
    assert len(result.lvdt) == len(result.time)


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


def test_lvdt_measurement_is_normalized_and_noiseless_when_requested():
    rng = random.Random(0)
    model = BrushedMotorModel(lvdt_full_scale=0.05, lvdt_noise_std=0.0, rng=rng)
    result = model.simulate(0.0, duration=0.01, dt=1e-4)

    assert len(result.lvdt) == len(result.time)
    assert all(abs(value) <= 1.0 for value in result.lvdt)
    assert all(value == 0.0 for value in result.lvdt)


def test_lvdt_measurement_saturates_at_full_scale():
    rng = random.Random(1)
    model = BrushedMotorModel(lvdt_full_scale=1e-5, lvdt_noise_std=0.0, rng=rng)
    result = model.simulate(5.0, duration=5e-3, dt=1e-4)

    assert result.lvdt[-1] == 1.0
    negative_result = model.simulate(-5.0, duration=5e-3, dt=1e-4)
    assert negative_result.lvdt[-1] == -1.0
