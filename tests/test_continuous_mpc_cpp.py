"""Tests ensuring the native ContMPCController matches the Python behaviour."""

from __future__ import annotations

import math

import pytest

from motor_model.brushed_motor import (
    BrushedMotorModel,
    rpm_per_volt_to_rad_per_sec_per_volt,
)
from motor_model.continuous_mpc_controller import ContMPCController as PyContMPC
from motor_model._mpc_common import MPCWeights
from motor_model._native.continuous_mpc import ContMPCController as NativeContMPC


SUPPLY_BUS_VOLTAGE = 28.0


def make_motor(**overrides: float) -> BrushedMotorModel:
    params = {
        "lvdt_noise_std": 0.0,
        "integration_substeps": 1,
    }
    params.update(overrides)
    return BrushedMotorModel(**params)


def make_weights(**overrides: float) -> MPCWeights:
    params = {
        "position": 280.0,
        "speed": 0.8,
        "voltage": 0.05,
        "delta_voltage": 0.4,
        "terminal_position": 620.0,
    }
    params.update(overrides)
    return MPCWeights(**params)


def instantiate_controllers(motor: BrushedMotorModel) -> tuple[PyContMPC, NativeContMPC]:
    py_weights = make_weights()
    native_weights = make_weights()

    kwargs = dict(
        dt=0.01,
        horizon=4,
        voltage_limit=8.0,
        target_lvdt=0.35,
        position_tolerance=0.01,
        static_friction_penalty=35.0,
        friction_compensation=None,
        auto_fc_gain=1.7,
        auto_fc_floor=0.1,
        auto_fc_cap=0.9,
        friction_blend_error_low=0.05,
        friction_blend_error_high=0.3,
        internal_substeps=12,
        robust_electrical=False,
        electrical_alpha=0.75,
        inductance_rel_uncertainty=0.2,
        pd_blend=0.55,
        pd_kp=5.2,
        pd_kd=0.28,
        pi_ki=0.45,
        pi_limit=3.8,
        pi_gate_saturation=False,
        pi_gate_blocked=False,
        pi_gate_error_band=True,
        pi_leak_near_setpoint=False,
        use_model_integrator=True,
        opt_iters=6,
        opt_step=0.12,
        opt_eps=0.08,
    )

    py_controller = PyContMPC(motor, weights=py_weights, **kwargs)
    native_controller = NativeContMPC(motor, weights=native_weights, **kwargs)

    return py_controller, native_controller


def controller_pair() -> tuple[PyContMPC, NativeContMPC]:
    motor = make_motor()
    return instantiate_controllers(motor)


def compare_sequences(
    py_controller: PyContMPC,
    native_controller: NativeContMPC,
    steps: int,
    *,
    duty_cycle: bool = False,
) -> None:
    dt = py_controller.dt
    last_time = py_controller._last_measurement_time or 0.0
    time = last_time + dt
    measurements = [0.1 * math.sin(i * 0.4) for i in range(steps)]

    outputs_py: list[float] = []
    outputs_native: list[float] = []

    for measurement in measurements:
        outputs_py.append(
            py_controller.update(time=time, measurement=measurement)
        )
        outputs_native.append(
            native_controller.update(time=time, measurement=measurement)
        )
        time += dt

    if duty_cycle:
        scaled_native = [value * SUPPLY_BUS_VOLTAGE for value in outputs_native]
        assert scaled_native == pytest.approx(outputs_py, rel=1e-9, abs=1e-9)
    else:
        assert outputs_native == pytest.approx(outputs_py, rel=1e-9, abs=1e-9)
    assert native_controller._state == pytest.approx(py_controller._state, rel=1e-9, abs=1e-9)
    assert native_controller._u_seq == pytest.approx(py_controller._u_seq, rel=1e-9, abs=1e-9)
    assert native_controller._int_err == pytest.approx(py_controller._int_err, rel=1e-9, abs=1e-9)
    assert native_controller._u_bias == pytest.approx(py_controller._u_bias, rel=1e-9, abs=1e-9)


def test_native_controller_matches_python() -> None:
    py_controller, native_controller = controller_pair()

    initial_measurement = 0.2
    py_controller.reset(
        initial_measurement=initial_measurement,
        initial_current=0.0,
        initial_speed=0.0,
    )
    native_controller.reset(
        initial_measurement=initial_measurement,
        initial_current=0.0,
        initial_speed=0.0,
    )

    assert native_controller.friction_compensation == pytest.approx(
        py_controller.friction_compensation
    )
    assert native_controller._auto_friction_compensation == pytest.approx(
        py_controller._auto_friction_compensation
    )
    assert len(native_controller._prediction_models) == len(py_controller._prediction_models)
    assert native_controller._motor is py_controller._motor

    compare_sequences(py_controller, native_controller, steps=6)

    # Switch to error-based PI mode and tweak weights mid-flight.
    py_controller.use_model_integrator = False
    native_controller.use_model_integrator = False
    py_controller.pi_ki = 0.62
    native_controller.pi_ki = 0.62
    py_controller.pi_gate_blocked = True
    native_controller.pi_gate_blocked = True
    py_controller.pi_gate_saturation = True
    native_controller.pi_gate_saturation = True
    py_controller.pi_gate_error_band = False
    native_controller.pi_gate_error_band = False
    py_controller.pi_leak_near_setpoint = True
    native_controller.pi_leak_near_setpoint = True

    py_weights = py_controller.weights
    native_weights = native_controller.weights
    py_weights.voltage = 0.07
    native_weights.voltage = 0.07
    py_weights.speed = 0.95
    native_weights.speed = 0.95

    compare_sequences(py_controller, native_controller, steps=5)

    # Adapt to a new motor and test explicit friction compensation settings.
    new_motor = make_motor(
        resistance=24.0,
        inductance=14e-3,
        kv=rpm_per_volt_to_rad_per_sec_per_volt(6.5),
    )

    py_controller.adapt_to_motor(
        new_motor,
        voltage_limit=7.2,
        friction_compensation=0.55,
    )
    native_controller.adapt_to_motor(
        new_motor,
        voltage_limit=7.2,
        friction_compensation=0.55,
    )

    assert native_controller.friction_compensation == pytest.approx(
        py_controller.friction_compensation
    )
    assert native_controller._active_user_friction_compensation == pytest.approx(
        py_controller._active_user_friction_compensation
    )

    # Reapply the cached user request to ensure state tracking is identical.
    py_controller.adapt_to_motor(new_motor, friction_compensation=None)
    native_controller.adapt_to_motor(new_motor, friction_compensation=None)

    assert native_controller.friction_compensation == pytest.approx(
        py_controller.friction_compensation
    )

    py_controller.reset(
        initial_measurement=-0.15,
        initial_current=0.0,
        initial_speed=0.0,
    )
    native_controller.reset(
        initial_measurement=-0.15,
        initial_current=0.0,
        initial_speed=0.0,
    )

    compare_sequences(py_controller, native_controller, steps=7)


def test_parameter_mutations_cover_all_attributes() -> None:
    py_controller, native_controller = controller_pair()

    py_controller.reset(
        initial_measurement=-0.12,
        initial_current=0.05,
        initial_speed=-0.08,
    )
    native_controller.reset(
        initial_measurement=-0.12,
        initial_current=0.05,
        initial_speed=-0.08,
    )

    param_updates = [
        ("dt", 0.012),
        ("horizon", 6),
        ("voltage_limit", 6.5),
        ("target_lvdt", -0.25),
        ("position_tolerance", 0.015),
        ("static_friction_penalty", 42.0),
        ("internal_substeps", 9),
        ("robust_electrical", True),
        ("inductance_rel_uncertainty", 0.15),
        ("pd_blend", 0.35),
        ("pd_kp", 4.7),
        ("pd_kd", 0.33),
        ("pi_ki", 0.52),
        ("pi_limit", 3.6),
        ("pi_gate_saturation", True),
        ("pi_gate_blocked", True),
        ("pi_gate_error_band", False),
        ("pi_leak_near_setpoint", True),
        ("use_model_integrator", False),
        ("auto_fc_gain", 1.6),
        ("auto_fc_floor", 0.07),
        ("friction_blend_error_low", 0.08),
        ("friction_blend_error_high", 0.45),
        ("model_bias_limit", 2.4),
        ("_opt_iters", 5),
        ("_opt_step", 0.09),
        ("_opt_eps", 0.04),
        ("_electrical_alpha", 0.6),
    ]

    for attribute, value in param_updates:
        setattr(py_controller, attribute, value)
        setattr(native_controller, attribute, value)

    # Optional parameters can be cleared and reassigned.
    py_controller.auto_fc_cap = None
    native_controller.auto_fc_cap = None
    py_controller.auto_fc_cap = 0.6
    native_controller.auto_fc_cap = 0.6

    py_controller.friction_compensation = 0.35
    native_controller.friction_compensation = 0.35

    weight_updates = dict(
        position=310.0,
        speed=1.25,
        voltage=0.06,
        delta_voltage=0.5,
        terminal_position=680.0,
    )
    py_weights = py_controller.weights
    native_weights = native_controller.weights
    for attribute, value in weight_updates.items():
        setattr(py_weights, attribute, value)
        setattr(native_weights, attribute, value)

    py_controller.reset(
        initial_measurement=0.18,
        initial_current=-0.02,
        initial_speed=0.11,
    )
    native_controller.reset(
        initial_measurement=0.18,
        initial_current=-0.02,
        initial_speed=0.11,
    )

    # Direct integrator state and cached friction parameters remain writable.
    py_controller._int_err = 0.12
    native_controller._int_err = 0.12
    py_controller._u_bias = -0.18
    native_controller._u_bias = -0.18
    py_controller._user_friction_compensation_request = 0.31
    native_controller._user_friction_compensation_request = 0.31
    py_controller._auto_friction_compensation = 0.22
    native_controller._auto_friction_compensation = 0.22
    py_controller._active_user_friction_compensation = 0.27
    native_controller._active_user_friction_compensation = 0.27
    py_controller._last_voltage = 0.9
    native_controller._last_voltage = 0.9
    py_controller._last_measured_position = -0.03
    native_controller._last_measured_position = -0.03
    py_controller._last_measurement_time = 0.05
    native_controller._last_measurement_time = 0.05

    compare_sequences(py_controller, native_controller, steps=4)


def test_duty_cycle_output_matches_scaled_voltage() -> None:
    py_controller, native_controller = controller_pair()

    py_controller.reset(
        initial_measurement=0.05,
        initial_current=0.0,
        initial_speed=0.0,
    )
    native_controller.reset(
        initial_measurement=0.05,
        initial_current=0.0,
        initial_speed=0.0,
    )

    native_controller.output_duty_cycle = True

    compare_sequences(py_controller, native_controller, steps=5, duty_cycle=True)

    assert native_controller._last_voltage == pytest.approx(
        py_controller._last_voltage, rel=1e-9, abs=1e-9
    )
