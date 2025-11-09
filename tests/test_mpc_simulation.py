"""Behavioural tests for the lightweight MPC simulation loop."""

from __future__ import annotations

import math

import pytest

from motor_model import (
    MPCWeights,
    MotorSimulation,
    build_default_controller_kwargs,
    build_default_continuous_controller_kwargs,
    build_default_tube_controller_kwargs,
    build_default_motor_kwargs,
)


def _make_simulation(
    *,
    motor_overrides=None,
    controller_overrides=None,
    controller_model_overrides=None,
    controller_type: str = "lvdtnom",
):
    motor_kwargs = build_default_motor_kwargs(**(motor_overrides or {}))
    if controller_type == "tube":
        controller_kwargs = build_default_tube_controller_kwargs(
            **(controller_overrides or {})
        )
    elif controller_type == "continuous":
        controller_kwargs = build_default_continuous_controller_kwargs(
            **(controller_overrides or {})
        )
    else:
        controller_kwargs = build_default_controller_kwargs(
            **(controller_overrides or {})
        )
    controller_kwargs["weights"] = MPCWeights()  # ensure a fresh copy
    controller_model_kwargs = None
    if controller_model_overrides:
        controller_model_kwargs = build_default_motor_kwargs(**controller_model_overrides)
    return MotorSimulation(
        motor_kwargs,
        controller_kwargs,
        controller_motor_kwargs=controller_model_kwargs,
        controller_type=controller_type,
    )


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
    assert math.isclose(math.degrees(state.position), target_deg, abs_tol=1.0)
    assert abs(state.speed) < math.radians(2.0)


def test_tube_controller_simulation_remains_stable():
    target_deg = 6.0
    sim = _make_simulation(controller_type="tube")
    sim.set_target_position(math.radians(target_deg))

    expected_lvdt = math.radians(target_deg) / sim.motor.lvdt_full_scale
    assert math.isclose(sim.controller.target_lvdt, expected_lvdt, rel_tol=1e-6)

    sim.run_for(2.0)

    state = sim.state()
    assert math.isfinite(state.position)
    assert math.isfinite(state.speed)
    assert abs(math.degrees(state.position)) <= 30.0
    assert abs(math.degrees(state.speed)) <= 30.0

    history = sim.history()
    assert max(abs(v) for v in history.voltage) <= sim.controller.voltage_limit + 1e-6


def test_continuous_controller_simulation_behaves():
    target_deg = 5.0
    sim = _make_simulation(
        controller_type="continuous", controller_overrides={"opt_iters": 3}
    )
    sim.set_target_position(math.radians(target_deg))

    expected_lvdt = math.radians(target_deg) / sim.motor.lvdt_full_scale
    assert math.isclose(sim.controller.target_lvdt, expected_lvdt, rel_tol=1e-6)

    sim.run_for(1.5)

    state = sim.state()
    assert math.isfinite(state.position)
    assert math.isfinite(state.speed)
    assert abs(math.degrees(state.position)) <= 30.0
    assert abs(math.degrees(state.speed)) <= 30.0

    history = sim.history()
    assert max(abs(v) for v in history.voltage) <= sim.controller.voltage_limit + 1e-6


def _deg(value: float) -> float:
    return math.degrees(value)


@pytest.mark.parametrize(
    ("motor_overrides", "controller_overrides", "target_deg", "duration"),
    [
        ({}, {}, 25.0, 0.8),
        (
            {"spring_constant": 2.4e-3, "spring_compression_ratio": 0.6},
            {},
            12.0,
            0.8,
        ),
        (
            {"inertia": 6.8e-4, "viscous_friction": 3.8e-4, "coulomb_friction": 2.85e-2},
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
    assert math.isclose(final_position, target_deg, abs_tol=5.0)

    final_speed = _deg(state.speed)
    assert math.isfinite(final_speed)
    assert abs(final_speed) < 3.0

    max_voltage = max(abs(v) for v in history.voltage)
    voltage_limit = sim.controller.voltage_limit
    assert max_voltage <= voltage_limit + 1e-6


def test_controller_model_parameters_can_differ_from_physical():
    sim = _make_simulation(
        motor_overrides={"resistance": 3.5, "inertia": 9e-5},
        controller_model_overrides={"resistance": 7.0, "inertia": 4e-5},
    )
    sim.set_target_position(math.radians(6.0))
    sim.run_for(0.6)

    # The controller should be using a separate motor model instance.
    assert not math.isclose(sim.motor.resistance, sim.controller._motor.resistance)
    assert not math.isclose(sim.motor.inertia, sim.controller._motor.inertia)

    state = sim.state()
    assert math.isfinite(state.position)
