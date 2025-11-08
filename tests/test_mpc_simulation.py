"""Behavioural tests for the lightweight MPC simulation loop."""

from __future__ import annotations

import math

import pytest

from motor_model import (
    MPCWeights,
    MotorSimulation,
    build_default_controller_kwargs,
    build_default_motor_kwargs,
)


def _make_simulation(*, motor_overrides=None, controller_overrides=None):
    motor_kwargs = build_default_motor_kwargs(**(motor_overrides or {}))
    controller_kwargs = build_default_controller_kwargs(**(controller_overrides or {}))
    controller_kwargs["weights"] = MPCWeights()  # ensure a fresh copy
    return MotorSimulation(motor_kwargs, controller_kwargs)


def test_simulation_remains_bounded_at_zero_setpoint():
    sim = _make_simulation()
    sim.run_for(1.0)
    history = sim.history()

    assert max(abs(p) for p in history.position) < math.radians(5)
    assert max(abs(v) for v in history.voltage) <= sim.controller.voltage_limit + 1e-6


def test_simulation_tracks_step_setpoint():
    target_deg = 4.0
    sim = _make_simulation()
    sim.set_target_position(math.radians(target_deg))
    sim.run_for(2.0)

    state = sim.state()
    assert math.isclose(math.degrees(state.position), target_deg, abs_tol=0.5)
    assert abs(state.speed) < math.radians(2.0)


def _deg(value: float) -> float:
    return math.degrees(value)


@pytest.mark.parametrize(
    ("motor_overrides", "controller_overrides", "target_deg", "duration"),
    [
        ({}, {}, 25.0, 0.8),
        (
            {"spring_constant": 2.5e-4, "spring_compression_ratio": 0.6},
            {},
            12.0,
            0.8,
        ),
        (
            {"inertia": 7e-5, "viscous_friction": 4e-5, "coulomb_friction": 3e-3},
            {"dt": 0.004, "internal_substeps": 4, "voltage_limit": 12.0},
            -18.0,
            0.8,
        ),
    ],
)
def test_no_divergence_for_varied_parameters(
    motor_overrides, controller_overrides, target_deg, duration
):
    sim = _make_simulation(
        motor_overrides=motor_overrides, controller_overrides=controller_overrides
    )
    sim.set_target_position(math.radians(target_deg))
    sim.run_for(duration)

    history = sim.history()
    state = sim.state()

    max_position = max(abs(_deg(p)) for p in history.position)
    assert max_position <= 30.5

    final_position = _deg(state.position)
    assert math.isfinite(final_position)
    assert math.isclose(final_position, target_deg, abs_tol=2.0)

    final_speed = _deg(state.speed)
    assert math.isfinite(final_speed)
    assert abs(final_speed) < 3.0

    max_voltage = max(abs(v) for v in history.voltage)
    voltage_limit = sim.controller.voltage_limit
    assert max_voltage <= voltage_limit + 1e-6
