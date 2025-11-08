"""Behavioural tests for the lightweight MPC simulation loop."""

from __future__ import annotations

import math

from motor_model import (
    MPCWeights,
    MotorSimulation,
    build_default_controller_kwargs,
    build_default_motor_kwargs,
)


def _make_simulation(**controller_overrides):
    motor_kwargs = build_default_motor_kwargs()
    controller_kwargs = build_default_controller_kwargs(**controller_overrides)
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
