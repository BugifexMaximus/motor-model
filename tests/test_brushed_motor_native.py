import math

import pytest

from motor_model.brushed_motor import (
    BrushedMotorModel as PythonMotor,
    rad_per_sec_per_volt_to_rpm_per_volt as py_rad_to_rpm,
    rpm_per_volt_to_rad_per_sec_per_volt as py_rpm_to_rad,
)
from motor_model._native import load_brushed_motor


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
