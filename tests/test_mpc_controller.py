"""Tests for the LVDT-based MPC controller."""

import math

from motor_model import BrushedMotorModel, LVDTMPCController


def test_mpc_overcomes_static_friction():
    model = BrushedMotorModel(lvdt_noise_std=0.0)
    controller_dt = 0.01
    controller = LVDTMPCController(
        model,
        dt=controller_dt,
        horizon=4,
        voltage_limit=8.0,
        target_lvdt=0.2,
        candidate_count=5,
        position_tolerance=0.01,
        static_friction_penalty=80.0,
        internal_substeps=10,
    )

    sim_dt = 0.001
    result = model.simulate(
        controller,
        duration=0.4,
        dt=sim_dt,
        measurement_period=controller_dt,
        controller_period=controller_dt,
    )

    final_lvdt = result.lvdt[-1]
    assert abs(final_lvdt - 0.2) < 0.03
    # The controller should have broken static friction and produced motion.
    assert result.position[-1] > 0.015
    assert max(abs(speed) for speed in result.speed) > 0.0

    # LVDT readings should be logged at the controller/measurement rate.
    lvdt_intervals = [
        b - a for a, b in zip(result.lvdt_time, result.lvdt_time[1:])
    ]
    assert all(abs(interval - controller_dt) < 1e-9 for interval in lvdt_intervals)


def test_mpc_handles_small_adjustments():
    model = BrushedMotorModel(lvdt_noise_std=0.0)
    controller_dt = 0.01
    controller = LVDTMPCController(
        model,
        dt=controller_dt,
        horizon=4,
        voltage_limit=6.0,
        target_lvdt=0.05,
        candidate_count=5,
        position_tolerance=0.01,
        static_friction_penalty=60.0,
        internal_substeps=10,
    )

    sim_dt = 0.001
    result = model.simulate(
        controller,
        duration=0.3,
        dt=sim_dt,
        measurement_period=controller_dt,
        controller_period=controller_dt,
    )

    final_lvdt = result.lvdt[-1]
    assert abs(final_lvdt - 0.05) < 0.03
    # Position should settle near the small reference instead of sticking at zero.
    final_position = result.position[-1]
    assert final_position > 0.0
    assert final_position < model.lvdt_full_scale * 0.2


def test_mpc_adapts_to_new_parameter_sets():
    base_motor = BrushedMotorModel(lvdt_noise_std=0.0)
    controller = LVDTMPCController(
        base_motor,
        dt=0.01,
        horizon=3,
        voltage_limit=6.0,
        target_lvdt=0.0,
        candidate_count=5,
        internal_substeps=5,
    )

    assert controller.friction_compensation > 0.0
    original_candidates = controller._voltage_candidates

    new_motor = BrushedMotorModel(
        resistance=24.0,
        inductance=12e-3,
        kv=6.5,
        inertia=7e-5,
        viscous_friction=3e-5,
        coulomb_friction=2.8e-3,
        static_friction=3.2e-3,
        lvdt_noise_std=0.0,
    )

    controller._last_voltage = 10.0
    controller.adapt_to_motor(
        new_motor,
        candidate_count=7,
        voltage_limit=5.0,
    )

    assert controller._motor is new_motor
    expected_friction = min(
        new_motor.static_friction * new_motor.resistance / new_motor._kt * 1.1,
        controller.voltage_limit,
    )
    assert math.isclose(controller.friction_compensation, expected_friction)
    assert controller._last_voltage == controller.voltage_limit
    assert controller._candidate_count == 7
    assert len(controller._voltage_candidates) >= controller._candidate_count
    assert controller._voltage_candidates != original_candidates
    assert any(
        math.isclose(abs(value), expected_friction, rel_tol=1e-6)
        for value in controller._voltage_candidates
    )
