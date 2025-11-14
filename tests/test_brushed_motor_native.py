import math

import pytest

from motor_model.brushed_motor import (
    BrushedMotorModel as PythonMotor,
    rad_per_sec_per_volt_to_rpm_per_volt as py_rad_to_rpm,
    rpm_per_volt_to_rad_per_sec_per_volt as py_rpm_to_rad,
)
from motor_model._native import load_brushed_motor, load_continuous_mpc


def assert_sequences_close(native_values, python_values, *, rel=1e-12, abs=1e-12):
    assert len(native_values) == len(python_values)
    for native_value, python_value in zip(native_values, python_values):
        assert native_value == pytest.approx(python_value, rel=rel, abs=abs)


def test_native_conversions_match_python():
    native = load_brushed_motor()
    rpm_value = 7.0
    assert native.rpm_per_volt_to_rad_per_sec_per_volt(rpm_value) == pytest.approx(
        py_rpm_to_rad(rpm_value)
    )
    rad_value = 2.0 * math.pi / 60.0
    assert native.rad_per_sec_per_volt_to_rpm_per_volt(rad_value) == pytest.approx(
        py_rad_to_rpm(rad_value)
    )


def test_native_motor_matches_python_simulation():
    native_module = load_brushed_motor()
    native_motor = native_module.BrushedMotorModel(lvdt_noise_std=0.0)
    python_motor = PythonMotor(lvdt_noise_std=0.0)

    duration = 0.12
    dt = 1e-3
    measurement_period = 5e-3

    def voltage_source(t: float) -> float:
        return 6.0 if t < 0.05 else -3.0

    def load_torque(t: float) -> float:
        return 0.01 * math.sin(10.0 * t)

    result_native = native_motor.simulate(
        voltage_source,
        duration=duration,
        dt=dt,
        initial_speed=0.15,
        initial_current=0.05,
        load_torque=load_torque,
        measurement_period=measurement_period,
    )

    result_python = python_motor.simulate(
        voltage_source,
        duration=duration,
        dt=dt,
        initial_speed=0.15,
        initial_current=0.05,
        load_torque=load_torque,
        measurement_period=measurement_period,
    )

    assert_sequences_close(result_native.time, result_python.time)
    assert_sequences_close(result_native.current, result_python.current)
    assert_sequences_close(result_native.speed, result_python.speed)
    assert_sequences_close(result_native.position, result_python.position)
    assert_sequences_close(result_native.torque, result_python.torque)
    assert_sequences_close(result_native.voltage, result_python.voltage)
    assert_sequences_close(result_native.lvdt_time, result_python.lvdt_time)
    assert_sequences_close(result_native.lvdt, result_python.lvdt)

    assert native_motor.speed_constant_rpm_per_volt == pytest.approx(
        python_motor.speed_constant_rpm_per_volt
    )


def test_native_motor_collects_controller_diagnostics() -> None:
    brushed_module = load_brushed_motor()
    controller_module = load_continuous_mpc()

    motor = brushed_module.BrushedMotorModel(lvdt_noise_std=0.0)
    controller = controller_module.ContMPCController(
        motor,
        dt=0.01,
        horizon=3,
        voltage_limit=6.0,
        target_lvdt=0.1,
        weights=None,
        position_tolerance=0.01,
        static_friction_penalty=35.0,
        friction_compensation=None,
        auto_fc_gain=1.5,
        auto_fc_floor=0.0,
        auto_fc_cap=None,
        friction_blend_error_low=0.05,
        friction_blend_error_high=0.25,
        internal_substeps=6,
        robust_electrical=False,
        electrical_alpha=None,
        inductance_rel_uncertainty=0.2,
        pd_blend=0.6,
        pd_kp=5.0,
        pd_kd=0.3,
        pi_ki=0.4,
        pi_limit=3.0,
        pi_gate_saturation=False,
        pi_gate_blocked=False,
        pi_gate_error_band=False,
        pi_leak_near_setpoint=False,
        use_model_integrator=True,
        opt_iters=5,
        opt_step=0.12,
        opt_eps=None,
    )

    result = motor.simulate(
        controller,
        duration=0.05,
        dt=0.005,
        measurement_period=0.01,
        controller_period=0.01,
    )

    assert len(result.pi_integrator) == len(result.time)
    assert len(result.model_integrator) == len(result.time)
    assert len(result.planned_voltage) == len(result.time)

    assert any(math.isfinite(value) for value in result.pi_integrator)
    assert any(math.isfinite(value) for value in result.model_integrator)
    assert any(result.planned_voltage)
    horizon = controller.horizon
    assert any(len(seq) == horizon for seq in result.planned_voltage if seq)
