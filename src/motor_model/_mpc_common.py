"""Shared helpers for motor MPC controllers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

from .brushed_motor import BrushedMotorModel


Vector3 = Tuple[float, float, float]
Matrix3 = Tuple[
    Tuple[float, float, float],
    Tuple[float, float, float],
    Tuple[float, float, float],
]


@dataclass
class MPCWeights:
    """Weights used by the MPC cost function."""

    position: float = 300.0
    speed: float = 0.5
    voltage: float = 0.02
    delta_voltage: float = 0.25
    terminal_position: float = 700.0


def _predict_next_state(
    state: Vector3,
    voltage: float,
    *,
    motor: BrushedMotorModel,
    dt: float,
    internal_substeps: int,
    robust_electrical: bool,
    electrical_alpha: float,
) -> Vector3:
    current, speed, position = state
    sub_dt = dt / internal_substeps

    for _ in range(internal_substeps):
        if robust_electrical:
            back_emf = motor._ke * speed
            steady_state_current = (voltage - back_emf) / motor.resistance
            if motor.resistance > 0.0:
                tau = motor.inductance / motor.resistance
            else:
                tau = math.inf
            if tau <= 0.0:
                alpha = 1.0
            else:
                alpha = 1.0 - math.exp(-sub_dt / tau)
            alpha = max(0.0, min(alpha, electrical_alpha, 1.0))
            current += alpha * (steady_state_current - current)
        else:
            di_dt = (
                voltage
                - motor.resistance * current
                - motor._ke * speed
            ) / motor.inductance
            current += di_dt * sub_dt

        electromagnetic_torque = motor._kt * current
        spring_torque = motor._spring_torque(position)
        available_torque = electromagnetic_torque - spring_torque

        if (
            abs(speed) < motor.stop_speed_threshold
            and abs(available_torque) <= motor.static_friction
        ):
            speed = 0.0
        else:
            friction_direction = motor._sign(speed) or motor._sign(available_torque)
            dynamic_friction = (
                motor.coulomb_friction * friction_direction
                + motor.viscous_friction * speed
            )
            net_torque = available_torque - dynamic_friction
            angular_acceleration = net_torque / motor.inertia
            speed += angular_acceleration * sub_dt

        position += speed * sub_dt

    return current, speed, position


def _measurement_to_position_value(measurement: float, motor: BrushedMotorModel) -> float:
    measurement = max(-1.0, min(1.0, measurement))
    return measurement * motor.lvdt_full_scale


def _position_to_measurement_value(position: float, motor: BrushedMotorModel) -> float:
    normalized = position / motor.lvdt_full_scale
    return max(-1.0, min(1.0, normalized))


def _clamp_symmetric(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def _clone_motor_with_inductance(
    motor: BrushedMotorModel, *, inductance: float
) -> BrushedMotorModel:
    inductance = max(inductance, 1e-9)
    return BrushedMotorModel(
        resistance=motor.resistance,
        inductance=inductance,
        kv=motor.kv,
        inertia=motor.inertia,
        viscous_friction=motor.viscous_friction,
        coulomb_friction=motor.coulomb_friction,
        static_friction=motor.static_friction,
        stop_speed_threshold=motor.stop_speed_threshold,
        spring_constant=motor.spring_constant,
        spring_compression_ratio=motor.spring_compression_ratio,
        spring_zero_position=motor.spring_zero_position,
        lvdt_full_scale=motor.lvdt_full_scale,
        lvdt_noise_std=0.0,
        integration_substeps=motor.integration_substeps,
        rng=motor._rng,
    )


def _mat_vec_mul(matrix: Matrix3, vector: Vector3) -> Vector3:
    return (
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1] + matrix[0][2] * vector[2],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1] + matrix[1][2] * vector[2],
        matrix[2][0] * vector[0] + matrix[2][1] * vector[1] + matrix[2][2] * vector[2],
    )


def _mat_mul(a: Matrix3, b: Matrix3) -> Matrix3:
    return (
        (
            a[0][0] * b[0][0] + a[0][1] * b[1][0] + a[0][2] * b[2][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1] + a[0][2] * b[2][1],
            a[0][0] * b[0][2] + a[0][1] * b[1][2] + a[0][2] * b[2][2],
        ),
        (
            a[1][0] * b[0][0] + a[1][1] * b[1][0] + a[1][2] * b[2][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1] + a[1][2] * b[2][1],
            a[1][0] * b[0][2] + a[1][1] * b[1][2] + a[1][2] * b[2][2],
        ),
        (
            a[2][0] * b[0][0] + a[2][1] * b[1][0] + a[2][2] * b[2][0],
            a[2][0] * b[0][1] + a[2][1] * b[1][1] + a[2][2] * b[2][1],
            a[2][0] * b[0][2] + a[2][1] * b[1][2] + a[2][2] * b[2][2],
        ),
    )


def _transpose(matrix: Matrix3) -> Matrix3:
    return (
        (matrix[0][0], matrix[1][0], matrix[2][0]),
        (matrix[0][1], matrix[1][1], matrix[2][1]),
        (matrix[0][2], matrix[1][2], matrix[2][2]),
    )


def _vec_dot(a: Vector3, b: Vector3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _outer_product(a: Vector3, b: Sequence[float]) -> Matrix3:
    return (
        (a[0] * b[0], a[0] * b[1], a[0] * b[2]),
        (a[1] * b[0], a[1] * b[1], a[1] * b[2]),
        (a[2] * b[0], a[2] * b[1], a[2] * b[2]),
    )


__all__ = [
    "Matrix3",
    "MPCWeights",
    "Vector3",
    "_clamp_symmetric",
    "_clone_motor_with_inductance",
    "_mat_mul",
    "_mat_vec_mul",
    "_measurement_to_position_value",
    "_outer_product",
    "_position_to_measurement_value",
    "_predict_next_state",
    "_transpose",
    "_vec_dot",
]
