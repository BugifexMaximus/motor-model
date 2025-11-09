"""Tube-based MPC controller implementation."""

from __future__ import annotations

import math
from itertools import product
from typing import Iterable, List, Sequence, Tuple

from ._mpc_common import (
    Matrix3,
    MPCWeights,
    Vector3,
    _clamp_symmetric,
    _clone_motor_with_inductance,
    _mat_mul,
    _mat_vec_mul,
    _measurement_to_position_value,
    _outer_product,
    _position_to_measurement_value,
    _predict_next_state,
    _transpose,
    _vec_dot,
)
from .brushed_motor import BrushedMotorModel


class TubeMPCController:
    """Tube-based robust MPC controller using a nominal motor model."""

    def __init__(
        self,
        motor: BrushedMotorModel,
        *,
        dt: float,
        horizon: int = 4,
        voltage_limit: float = 10.0,
        target_lvdt: float = 0.0,
        candidate_count: int = 5,
        weights: MPCWeights | None = None,
        position_tolerance: float = 0.02,
        static_friction_penalty: float = 50.0,
        friction_compensation: float | None = None,
        internal_substeps: int = 1,
        robust_electrical: bool = True,
        electrical_alpha: float | None = None,
        inductance_rel_uncertainty: float = 0.5,
        tube_tolerance: float = 1e-6,
        tube_max_iterations: int = 500,
        lqr_state_weight: Sequence[float] = (2.0, 0.2, 5.0),
        lqr_input_weight: float = 0.5,
    ) -> None:
        if dt <= 0:
            raise ValueError("dt must be positive")
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if voltage_limit <= 0:
            raise ValueError("voltage_limit must be positive")
        if candidate_count < 3 or candidate_count % 2 == 0:
            raise ValueError("candidate_count must be an odd integer >= 3")
        if not -1.0 <= target_lvdt <= 1.0:
            raise ValueError("target_lvdt must be within [-1, 1]")
        if position_tolerance < 0:
            raise ValueError("position_tolerance must be non-negative")
        if static_friction_penalty < 0:
            raise ValueError("static_friction_penalty must be non-negative")
        if internal_substeps <= 0:
            raise ValueError("internal_substeps must be positive")
        if electrical_alpha is not None and not 0.0 <= electrical_alpha <= 1.0:
            raise ValueError("electrical_alpha must be within [0, 1]")
        if inductance_rel_uncertainty < 0.0:
            raise ValueError("inductance_rel_uncertainty must be non-negative")
        if tube_tolerance <= 0.0:
            raise ValueError("tube_tolerance must be positive")
        if tube_max_iterations <= 0:
            raise ValueError("tube_max_iterations must be positive")
        if len(lqr_state_weight) != 3:
            raise ValueError("lqr_state_weight must contain three entries")
        if lqr_input_weight <= 0.0:
            raise ValueError("lqr_input_weight must be positive")

        self.dt = dt
        self.horizon = horizon
        self.voltage_limit = voltage_limit
        self.target_lvdt = target_lvdt
        self.weights = weights or MPCWeights()
        self.position_tolerance = position_tolerance
        self.static_friction_penalty = static_friction_penalty
        self.internal_substeps = internal_substeps
        self.robust_electrical = robust_electrical
        self._electrical_alpha = 1.0 if electrical_alpha is None else electrical_alpha
        self.inductance_rel_uncertainty = inductance_rel_uncertainty
        self._tube_tolerance = tube_tolerance
        self._tube_max_iterations = tube_max_iterations
        self._lqr_state_weight = tuple(float(value) for value in lqr_state_weight)
        self._lqr_input_weight = float(lqr_input_weight)

        self._candidate_count = candidate_count
        self._user_friction_compensation = friction_compensation

        self._state: Vector3 = (0.0, 0.0, 0.0)
        self._nominal_state: Vector3 = (0.0, 0.0, 0.0)
        self._last_control: float | None = None
        self._last_nominal_voltage: float | None = None
        self._last_measured_position: float | None = None
        self._last_measurement_time: float | None = None

        self._motor: BrushedMotorModel | None = None
        self._error_bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] | None = None
        self._max_current_estimate: float = 0.0
        self._tightened_voltage_candidates: Tuple[float, ...] = (0.0,)
        self._tightened_voltage_limit: float = voltage_limit
        self._nominal_position_bounds: Tuple[float, float] = (-motor.lvdt_full_scale, motor.lvdt_full_scale)
        self._K: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._A: Matrix3 | None = None
        self._B: Vector3 | None = None
        self._A_K: Matrix3 | None = None
        self._tube_set: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] | None = None
        self._disturbance_bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] | None = None

        self._apply_motor_parameters(motor, friction_compensation)

    def adapt_to_motor(
        self,
        motor: BrushedMotorModel,
        *,
        candidate_count: int | None = None,
        voltage_limit: float | None = None,
        friction_compensation: float | None = None,
    ) -> None:
        if voltage_limit is not None:
            if voltage_limit <= 0:
                raise ValueError("voltage_limit must be positive")
            self.voltage_limit = voltage_limit

        if candidate_count is not None:
            if candidate_count < 3 or candidate_count % 2 == 0:
                raise ValueError("candidate_count must be an odd integer >= 3")
            self._candidate_count = candidate_count

        self._apply_motor_parameters(motor, friction_compensation)

    def reset(
        self,
        *,
        initial_measurement: float,
        initial_current: float,
        initial_speed: float,
    ) -> None:
        position = _measurement_to_position_value(initial_measurement, self._motor)
        self._state = (initial_current, initial_speed, position)
        self._nominal_state = (initial_current, initial_speed, position)
        self._last_control = None
        self._last_nominal_voltage = None
        self._last_measured_position = position
        self._last_measurement_time = None

    def update(self, *, time: float, measurement: float) -> float:
        if self._motor is None:
            return 0.0

        measured_position = _measurement_to_position_value(measurement, self._motor)
        normalized_measurement = _position_to_measurement_value(measured_position, self._motor)

        estimated_speed = 0.0
        if self._last_measured_position is not None and self._last_measurement_time is not None:
            dt = time - self._last_measurement_time
            if dt <= 0.0:
                raise ValueError("Controller update time must be strictly increasing")
            tolerance = max(1e-9, 0.25 * self.dt)
            if abs(dt - self.dt) > tolerance:
                raise ValueError(
                    "Controller update period deviates from configured dt: "
                    f"observed={dt:.6g}s expected={self.dt:.6g}s"
                )
            estimated_speed = (measured_position - self._last_measured_position) / dt

        current_estimate = self._state[0]
        if self._last_control is not None:
            quasi_static_current = (
                self._last_control - self._motor._ke * estimated_speed
            ) / self._motor.resistance
            limit = self._max_current_estimate
            current_estimate = max(-limit, min(limit, quasi_static_current))

        self._state = (current_estimate, estimated_speed, measured_position)
        self._last_measured_position = measured_position
        self._last_measurement_time = time

        best_sequence: Tuple[float, ...] | None = None
        best_cost = float("inf")

        for sequence in self._voltage_sequences():
            cost, terminal_state, feasible = self._evaluate_nominal_sequence(
                self._nominal_state, sequence
            )
            if not feasible:
                continue
            if cost < best_cost:
                best_cost = cost
                best_sequence = sequence

        nominal_voltage = 0.0
        if best_sequence is not None:
            nominal_voltage = best_sequence[0]

        error = (
            self._state[0] - self._nominal_state[0],
            self._state[1] - self._nominal_state[1],
            self._state[2] - self._nominal_state[2],
        )
        feedback_voltage = sum(k * e for k, e in zip(self._K, error))
        control_voltage = _clamp_symmetric(nominal_voltage + feedback_voltage, self.voltage_limit)

        position_error = self.target_lvdt - normalized_measurement
        if (
            abs(self._state[1]) < self._motor.stop_speed_threshold
            and abs(position_error) > self.position_tolerance
            and abs(control_voltage) < self.friction_compensation
        ):
            direction = 1.0 if position_error >= 0.0 else -1.0
            control_voltage = direction * self.friction_compensation
            control_voltage = _clamp_symmetric(control_voltage, self.voltage_limit)

        self._last_control = control_voltage
        self._last_nominal_voltage = nominal_voltage

        self._state = _predict_next_state(
            self._state,
            control_voltage,
            motor=self._motor,
            dt=self.dt,
            internal_substeps=self.internal_substeps,
            robust_electrical=self.robust_electrical,
            electrical_alpha=self._electrical_alpha,
        )

        self._nominal_state = _predict_next_state(
            self._nominal_state,
            nominal_voltage,
            motor=self._motor,
            dt=self.dt,
            internal_substeps=self.internal_substeps,
            robust_electrical=self.robust_electrical,
            electrical_alpha=self._electrical_alpha,
        )

        return control_voltage

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _apply_motor_parameters(
        self, motor: BrushedMotorModel, friction_compensation: float | None
    ) -> None:
        self._motor = motor
        self.friction_compensation = self._determine_friction_compensation(
            motor, friction_compensation
        )
        self._error_bounds = self._compute_error_bounds(motor)
        self._max_current_estimate = max(abs(bound) for bound in self._error_bounds[0])
        self._compute_feedback_and_tube()

    def _determine_friction_compensation(
        self, motor: BrushedMotorModel, friction_compensation: float | None
    ) -> float:
        if friction_compensation is None and self._user_friction_compensation is not None:
            friction_compensation = self._user_friction_compensation
        elif friction_compensation is not None:
            self._user_friction_compensation = friction_compensation

        if friction_compensation is not None:
            if friction_compensation <= 0:
                raise ValueError("friction_compensation must be positive")
            return min(friction_compensation, self.voltage_limit)

        min_motion_voltage = motor.static_friction * motor.resistance / motor._kt
        return min(min_motion_voltage * 1.1, self.voltage_limit)

    def _compute_error_bounds(
        self, motor: BrushedMotorModel
    ) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        max_current = self.voltage_limit / motor.resistance
        current_bounds = (-2.0 * max_current, 2.0 * max_current)
        max_speed = motor.kv * self.voltage_limit
        speed_bounds = (-2.0 * max_speed, 2.0 * max_speed)
        position_bounds = (-motor.lvdt_full_scale, motor.lvdt_full_scale)
        return (current_bounds, speed_bounds, position_bounds)

    def _compute_feedback_and_tube(self) -> None:
        linear_state = (0.0, 0.0, 0.0)
        A, B = self._linearize_nominal_dynamics(linear_state, 0.0)
        self._A = A
        self._B = B
        self._K = self._design_stabilising_feedback(A, B)
        self._A_K = self._compute_closed_loop_matrix(A, B, self._K)
        self._disturbance_bounds = self._estimate_disturbance_bounds(
            self._K, self._A_K
        )
        self._tube_set = self._compute_rpi_set(self._A_K, self._disturbance_bounds)
        self._tightened_voltage_limit = self._compute_tightened_voltage_limit(
            self._K, self._tube_set
        )
        self._tightened_voltage_candidates = self._build_voltage_candidates(
            self._tightened_voltage_limit
        )
        self._nominal_position_bounds = self._compute_nominal_position_bounds(
            self._tube_set
        )

    def _linearize_nominal_dynamics(
        self, state: Vector3, voltage: float
    ) -> Tuple[Matrix3, Vector3]:
        epsilon_state = 1e-5
        epsilon_voltage = 1e-4

        A_rows = []
        for index in range(3):
            perturb_plus = list(state)
            perturb_minus = list(state)
            perturb_plus[index] += epsilon_state
            perturb_minus[index] -= epsilon_state
            next_plus = _predict_next_state(
                tuple(perturb_plus),
                voltage,
                motor=self._motor,
                dt=self.dt,
                internal_substeps=self.internal_substeps,
                robust_electrical=self.robust_electrical,
                electrical_alpha=self._electrical_alpha,
            )
            next_minus = _predict_next_state(
                tuple(perturb_minus),
                voltage,
                motor=self._motor,
                dt=self.dt,
                internal_substeps=self.internal_substeps,
                robust_electrical=self.robust_electrical,
                electrical_alpha=self._electrical_alpha,
            )
            derivative = [
                (next_plus[i] - next_minus[i]) / (2.0 * epsilon_state)
                for i in range(3)
            ]
            A_rows.append(tuple(derivative))

        A = (A_rows[0], A_rows[1], A_rows[2])

        next_plus = _predict_next_state(
            state,
            voltage + epsilon_voltage,
            motor=self._motor,
            dt=self.dt,
            internal_substeps=self.internal_substeps,
            robust_electrical=self.robust_electrical,
            electrical_alpha=self._electrical_alpha,
        )
        next_minus = _predict_next_state(
            state,
            voltage - epsilon_voltage,
            motor=self._motor,
            dt=self.dt,
            internal_substeps=self.internal_substeps,
            robust_electrical=self.robust_electrical,
            electrical_alpha=self._electrical_alpha,
        )
        B = tuple(
            (next_plus[i] - next_minus[i]) / (2.0 * epsilon_voltage) for i in range(3)
        )

        return A, B

    def _design_stabilising_feedback(
        self, A: Matrix3, B: Vector3
    ) -> Tuple[float, float, float]:
        Q = (
            (self._lqr_state_weight[0], 0.0, 0.0),
            (0.0, self._lqr_state_weight[1], 0.0),
            (0.0, 0.0, self._lqr_state_weight[2]),
        )
        R = self._lqr_input_weight

        P = Q
        for _ in range(500):
            P_B = _mat_vec_mul(P, B)
            S = R + _vec_dot(B, P_B)
            S_inv = 1.0 / S
            P_A = _mat_mul(P, A)
            Bt_PA = [
                B[0] * P_A[0][i] + B[1] * P_A[1][i] + B[2] * P_A[2][i]
                for i in range(3)
            ]
            K = [S_inv * value for value in Bt_PA]

            At = _transpose(A)
            At_P = _mat_mul(At, P)
            At_P_A = _mat_mul(At, P_A)
            At_P_B = _mat_vec_mul(At, P_B)
            correction = _outer_product(At_P_B, Bt_PA)
            correction = tuple(
                tuple(S_inv * value for value in row) for row in correction
            )
            P_next = (
                tuple(At_P_A[0][j] - correction[0][j] + Q[0][j] for j in range(3)),
                tuple(At_P_A[1][j] - correction[1][j] + Q[1][j] for j in range(3)),
                tuple(At_P_A[2][j] - correction[2][j] + Q[2][j] for j in range(3)),
            )

            diff = max(
                abs(P_next[i][j] - P[i][j]) for i in range(3) for j in range(3)
            )
            P = P_next
            if diff < 1e-9:
                break

        feedback = tuple(-value for value in K)
        return feedback

    def _compute_closed_loop_matrix(
        self, A: Matrix3, B: Vector3, K: Tuple[float, float, float]
    ) -> Matrix3:
        return (
            (
                A[0][0] + B[0] * K[0],
                A[0][1] + B[0] * K[1],
                A[0][2] + B[0] * K[2],
            ),
            (
                A[1][0] + B[1] * K[0],
                A[1][1] + B[1] * K[1],
                A[1][2] + B[1] * K[2],
            ),
            (
                A[2][0] + B[2] * K[0],
                A[2][1] + B[2] * K[1],
                A[2][2] + B[2] * K[2],
            ),
        )

    def _estimate_disturbance_bounds(
        self,
        K: Tuple[float, float, float],
        A_K: Matrix3,
    ) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        error_bounds = self._error_bounds or ((-1.0, 1.0),) * 3
        current_samples = {error_bounds[0][0], 0.0, error_bounds[0][1]}
        speed_samples = {error_bounds[1][0], 0.0, error_bounds[1][1]}
        position_samples = {error_bounds[2][0], 0.0, error_bounds[2][1]}
        voltage_samples = {
            -self.voltage_limit,
            0.0,
            self.voltage_limit,
        }

        inductance_factors = [1.0]
        if self.inductance_rel_uncertainty > 0.0:
            lower = max(1.0 - self.inductance_rel_uncertainty, 1e-6)
            upper = 1.0 + self.inductance_rel_uncertainty
            if not math.isclose(lower, 1.0, rel_tol=0.0, abs_tol=1e-12):
                inductance_factors.append(lower)
            if not math.isclose(upper, 1.0, rel_tol=0.0, abs_tol=1e-12):
                inductance_factors.append(upper)

        min_bounds = [float("inf"), float("inf"), float("inf")]
        max_bounds = [float("-inf"), float("-inf"), float("-inf")]

        base_nominal = (0.0, 0.0, 0.0)

        for current in current_samples:
            for speed in speed_samples:
                for position in position_samples:
                    error = (current, speed, position)
                    for voltage in voltage_samples:
                        for factor in inductance_factors:
                            if math.isclose(factor, 1.0, rel_tol=0.0, abs_tol=1e-12):
                                motor = self._motor
                            else:
                                motor = _clone_motor_with_inductance(
                                    self._motor, inductance=self._motor.inductance * factor
                                )

                            real_state = (
                                base_nominal[0] + error[0],
                                base_nominal[1] + error[1],
                                base_nominal[2] + error[2],
                            )
                            feedback = sum(k * value for k, value in zip(K, error))
                            control = _clamp_symmetric(voltage + feedback, self.voltage_limit)

                            real_next = _predict_next_state(
                                real_state,
                                control,
                                motor=motor,
                                dt=self.dt,
                                internal_substeps=self.internal_substeps,
                                robust_electrical=self.robust_electrical,
                                electrical_alpha=self._electrical_alpha,
                            )
                            nominal_next = _predict_next_state(
                                base_nominal,
                                voltage,
                                motor=self._motor,
                                dt=self.dt,
                                internal_substeps=self.internal_substeps,
                                robust_electrical=self.robust_electrical,
                                electrical_alpha=self._electrical_alpha,
                            )

                            e_next = (
                                real_next[0] - nominal_next[0],
                                real_next[1] - nominal_next[1],
                                real_next[2] - nominal_next[2],
                            )
                            predicted = _mat_vec_mul(A_K, error)
                            disturbance = (
                                e_next[0] - predicted[0],
                                e_next[1] - predicted[1],
                                e_next[2] - predicted[2],
                            )

                            for index in range(3):
                                value = disturbance[index]
                                if value < min_bounds[index]:
                                    min_bounds[index] = value
                                if value > max_bounds[index]:
                                    max_bounds[index] = value

        results = []
        for index in range(3):
            width = max(abs(min_bounds[index]), abs(max_bounds[index]))
            if not math.isfinite(width):
                width = 0.0
            results.append((-width, width))

        return tuple(results)  # type: ignore[return-value]

    def _compute_rpi_set(
        self,
        A_K: Matrix3,
        disturbance_bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
    ) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        current = list(disturbance_bounds)

        for _ in range(self._tube_max_iterations):
            transformed = self._transform_box(A_K, current)
            next_box = self._minkowski_sum(transformed, disturbance_bounds)
            delta = max(
                abs(next_box[i][j] - current[i][j]) for i in range(3) for j in range(2)
            )
            current = list(next_box)
            if delta < self._tube_tolerance:
                break

        return tuple((float(bounds[0]), float(bounds[1])) for bounds in current)  # type: ignore[return-value]

    def _transform_box(
        self,
        matrix: Matrix3,
        box: Sequence[Tuple[float, float]],
    ) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        result = []
        for row in matrix:
            row_min = 0.0
            row_max = 0.0
            for coefficient, bounds in zip(row, box):
                candidates = (
                    coefficient * bounds[0],
                    coefficient * bounds[1],
                )
                row_min += min(candidates)
                row_max += max(candidates)
            result.append((row_min, row_max))
        return tuple(result)  # type: ignore[return-value]

    def _minkowski_sum(
        self,
        a: Sequence[Tuple[float, float]],
        b: Sequence[Tuple[float, float]],
    ) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        combined = []
        for first, second in zip(a, b):
            combined.append((first[0] + second[0], first[1] + second[1]))
        return tuple(combined)  # type: ignore[return-value]

    def _compute_tightened_voltage_limit(
        self,
        K: Tuple[float, float, float],
        tube: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] | None,
    ) -> float:
        if tube is None:
            return self.voltage_limit

        min_value = 0.0
        max_value = 0.0
        for gain, bounds in zip(K, tube):
            candidates = (gain * bounds[0], gain * bounds[1])
            min_value += min(candidates)
            max_value += max(candidates)
        delta = max(abs(min_value), abs(max_value))
        tightened = max(0.0, self.voltage_limit - delta)
        return tightened

    def _build_voltage_candidates(self, limit: float) -> Tuple[float, ...]:
        if limit <= 0.0:
            return (0.0,)

        half = self._candidate_count // 2
        step = limit / half
        values = {0.0}
        for index in range(1, half + 1):
            voltage = index * step
            values.add(min(voltage, limit))
            values.add(max(-voltage, -limit))
        return tuple(sorted(values))

    def _compute_nominal_position_bounds(
        self,
        tube: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] | None,
    ) -> Tuple[float, float]:
        if tube is None:
            return (-self._motor.lvdt_full_scale, self._motor.lvdt_full_scale)

        position_bounds = (-self._motor.lvdt_full_scale, self._motor.lvdt_full_scale)
        tube_position = tube[2]
        margin = max(abs(tube_position[0]), abs(tube_position[1]))
        tightened = (
            position_bounds[0] + margin,
            position_bounds[1] - margin,
        )
        if tightened[0] > tightened[1]:
            center = 0.5 * (tightened[0] + tightened[1])
            return (center, center)
        return tightened

    def _voltage_sequences(self) -> Iterable[Tuple[float, ...]]:
        return product(self._tightened_voltage_candidates, repeat=self.horizon)

    def _evaluate_nominal_sequence(
        self,
        initial_state: Vector3,
        sequence: Tuple[float, ...],
    ) -> Tuple[float, Vector3, bool]:
        state = initial_state
        cost = 0.0
        previous_voltage = self._last_nominal_voltage

        for voltage in sequence:
            state = _predict_next_state(
                state,
                voltage,
                motor=self._motor,
                dt=self.dt,
                internal_substeps=self.internal_substeps,
                robust_electrical=self.robust_electrical,
                electrical_alpha=self._electrical_alpha,
            )
            lvdt = _position_to_measurement_value(state[2], self._motor)
            position_error = self.target_lvdt - lvdt
            cost += self.weights.position * position_error * position_error
            cost += self.weights.speed * state[1] * state[1]
            cost += self.weights.voltage * voltage * voltage

            if previous_voltage is not None:
                delta = voltage - previous_voltage
                cost += self.weights.delta_voltage * delta * delta
            previous_voltage = voltage

            if (
                abs(state[1]) < self._motor.stop_speed_threshold
                and abs(position_error) > self.position_tolerance
                and abs(voltage) < self.friction_compensation
            ):
                cost += self.static_friction_penalty

            if not (
                self._nominal_position_bounds[0]
                <= state[2]
                <= self._nominal_position_bounds[1]
            ):
                return float("inf"), state, False

        terminal_error = self.target_lvdt - _position_to_measurement_value(
            state[2], self._motor
        )
        cost += self.weights.terminal_position * terminal_error * terminal_error

        return cost, state, True
