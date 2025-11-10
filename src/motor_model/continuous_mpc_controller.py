"""Continuous-action MPC controller for the brushed motor model."""

from __future__ import annotations

import math
from typing import List, Tuple

from ._mpc_common import (
    MPCWeights,
    _clamp_symmetric,
    _clone_motor_with_inductance,
    _measurement_to_position_value,
    _position_to_measurement_value,
    _predict_next_state,
)
from .brushed_motor import BrushedMotorModel


class ContMPCController:
    """Continuous-action MPC controller using LVDT feedback."""

    def __init__(
        self,
        motor: BrushedMotorModel,
        *,
        dt: float,
        horizon: int = 5,
        voltage_limit: float = 10.0,
        target_lvdt: float = 0.0,
        weights: MPCWeights | None = None,
        position_tolerance: float = 0.02,
        static_friction_penalty: float = 50.0,
        friction_compensation: float | None = None,
        auto_fc_gain: float = 0.4,
        auto_fc_floor: float = 0.0,
        auto_fc_cap: float | None = None,
        friction_blend_error_low: float = 0.2,
        friction_blend_error_high: float = 0.5,
        internal_substeps: int = 30,
        robust_electrical: bool = True,
        electrical_alpha: float | None = None,
        inductance_rel_uncertainty: float = 0.5,
        pd_blend: float = 0.7,
        pd_kp: float = 6.0,
        pd_kd: float = 0.4,
        pi_ki: float = 0.0,
        pi_limit: float = 5.0,
        pi_gate_saturation: bool = True,
        pi_gate_blocked: bool = True,
        pi_gate_error_band: bool = True,
        pi_leak_near_setpoint: bool = True,
        opt_iters: int = 10,
        opt_step: float = 0.1,
        opt_eps: float | None = 0.1,
    ) -> None:
        if dt <= 0:
            raise ValueError("dt must be positive")
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if voltage_limit <= 0:
            raise ValueError("voltage_limit must be positive")
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
        if not 0.0 <= pd_blend <= 1.0:
            raise ValueError("pd_blend must be within [0, 1]")
        if pd_kp <= 0.0:
            raise ValueError("pd_kp must be positive")
        if pd_kd < 0.0:
            raise ValueError("pd_kd must be non-negative")
        if pi_ki < 0.0:
            raise ValueError("pi_ki must be non-negative")
        if pi_limit <= 0.0:
            raise ValueError("pi_limit must be positive")
        if auto_fc_gain <= 0.0:
            raise ValueError("auto_fc_gain must be positive")
        if auto_fc_floor < 0.0:
            raise ValueError("auto_fc_floor must be non-negative")
        if auto_fc_cap is not None and auto_fc_cap <= 0.0:
            raise ValueError("auto_fc_cap must be positive when provided")
        if friction_blend_error_low < 0.0:
            raise ValueError("friction_blend_error_low must be non-negative")
        if friction_blend_error_high <= friction_blend_error_low:
            raise ValueError(
                "friction_blend_error_high must be greater than friction_blend_error_low"
            )
        if opt_iters <= 0:
            raise ValueError("opt_iters must be positive")
        if opt_step <= 0.0:
            raise ValueError("opt_step must be positive")

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

        self.pd_blend = pd_blend
        self.pd_kp = pd_kp
        self.pd_kd = pd_kd
        self.pi_ki = pi_ki
        self.pi_limit = abs(pi_limit)
        self.pi_gate_saturation = pi_gate_saturation
        self.pi_gate_blocked = pi_gate_blocked
        self.pi_gate_error_band = pi_gate_error_band
        self.pi_leak_near_setpoint = pi_leak_near_setpoint
        self._int_err = 0.0

        self.auto_fc_gain = auto_fc_gain
        self.auto_fc_floor = auto_fc_floor
        self.auto_fc_cap = auto_fc_cap
        self.friction_blend_error_low = friction_blend_error_low
        self.friction_blend_error_high = friction_blend_error_high

        self._opt_iters = opt_iters
        self._opt_step = opt_step
        self._opt_eps = opt_eps if opt_eps is not None else 0.05 * voltage_limit
        if self._opt_eps <= 0.0:
            raise ValueError("opt_eps must be positive")

        self._user_friction_compensation_request: float | None = friction_compensation
        self._auto_friction_compensation: float = 0.0
        self._active_user_friction_compensation: float | None = None
        self.friction_compensation: float = 0.0

        self._motor: BrushedMotorModel | None = None
        self._prediction_models: Tuple[BrushedMotorModel, ...] = tuple()

        self._u_seq: Tuple[float, ...] = (0.0,) * self.horizon
        self._state: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._last_voltage: float | None = None
        self._last_measured_position: float | None = None
        self._last_measurement_time: float | None = None

        self._apply_motor_parameters(motor, friction_compensation)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def adapt_to_motor(
        self,
        motor: BrushedMotorModel,
        *,
        voltage_limit: float | None = None,
        friction_compensation: float | None = None,
    ) -> None:
        if voltage_limit is not None:
            if voltage_limit <= 0:
                raise ValueError("voltage_limit must be positive")
            self.voltage_limit = voltage_limit

        self._apply_motor_parameters(motor, friction_compensation)

        if self._last_voltage is not None:
            self._last_voltage = self._clamp_voltage(self._last_voltage)

        if len(self._u_seq) != self.horizon:
            base = self._last_voltage or 0.0
            self._u_seq = (base,) * self.horizon

    def reset(
        self,
        *,
        initial_measurement: float,
        initial_current: float,
        initial_speed: float,
    ) -> None:
        position = self._measurement_to_position(initial_measurement)
        self._state = (initial_current, initial_speed, position)
        self._last_voltage = None
        self._last_measured_position = position
        self._last_measurement_time = None
        self._u_seq = (0.0,) * self.horizon
        self._int_err = 0.0

    def update(self, *, time: float, measurement: float) -> float:
        if self._motor is None:
            return 0.0

        measured_position = self._measurement_to_position(measurement)
        normalized_measurement = self._position_to_measurement(measured_position)

        current, _, _ = self._state

        estimated_speed = 0.0
        if (
            self._last_measured_position is not None
            and self._last_measurement_time is not None
        ):
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

        self._state = (current, estimated_speed, measured_position)
        self._last_measured_position = measured_position
        self._last_measurement_time = time

        u_seq = self._optimize_sequence(self._state)
        mpc_voltage = u_seq[0]

        position_error = self.target_lvdt - normalized_measurement
        friction_floor = self._dynamic_friction_compensation(position_error)

        if self.pi_ki > 0.0:
            # -------------------------------
            # Integral term update (conditional)
            # -------------------------------
            u_last = self._last_voltage or 0.0
            not_saturated = abs(u_last) < self.voltage_limit - 1e-6
            if not self.pi_gate_saturation:
                not_saturated = True

            e = position_error

            blocked = False
            if self.pi_gate_blocked and self._motor is not None:
                v_small = self._motor.stop_speed_threshold * 0.5
                u_block = 0.3 * self.voltage_limit
                blocked = abs(estimated_speed) < v_small and abs(u_last) > u_block

            allow_integration = not_saturated and not blocked
            if self.pi_gate_error_band:
                # Only suppress integration for truly tiny errors.
                # Anything outside the tolerance is allowed to drive I-term.
                allow_integration = allow_integration and abs(e) > self.position_tolerance

            if allow_integration:
                self._int_err += e * self.dt
            elif self.pi_leak_near_setpoint and abs(e) < self.position_tolerance:
                self._int_err *= 0.9

            max_int_state = self.pi_limit / self.pi_ki
            self._int_err = max(-max_int_state, min(max_int_state, self._int_err))

        # Integral contribution with hard output cap
        u_I = self.pi_ki * self._int_err
        if u_I > self.pi_limit:
            u_I = self.pi_limit
        elif u_I < -self.pi_limit:
            u_I = -self.pi_limit

        u_pd = (
            self.pd_kp * position_error
            - self.pd_kd * estimated_speed
            + u_I
        )
        u_raw = self.pd_blend * mpc_voltage + (1.0 - self.pd_blend) * u_pd

        if (
            abs(self._state[1]) < self._motor.stop_speed_threshold
            and abs(position_error) > self.position_tolerance
            and friction_floor > 0.0
            and abs(u_raw) < friction_floor
        ):
            u_raw = math.copysign(friction_floor, position_error)

        u = self._clamp_voltage(u_raw)

        self._state = self._predict_next(self._state, u)
        self._last_voltage = u

        return u

    # ------------------------------------------------------------------
    # Internal helpers: motor and friction
    # ------------------------------------------------------------------

    def _apply_motor_parameters(
        self,
        motor: BrushedMotorModel,
        friction_compensation: float | None,
    ) -> None:
        self._motor = motor
        (
            self._auto_friction_compensation,
            self._active_user_friction_compensation,
            self.friction_compensation,
        ) = self._determine_friction_compensation(motor, friction_compensation)
        self._prediction_models = self._build_prediction_models(motor)

    def _determine_friction_compensation(
        self,
        motor: BrushedMotorModel,
        friction_compensation: float | None,
    ) -> Tuple[float, float | None, float]:
        if (
            friction_compensation is None
            and self._user_friction_compensation_request is not None
        ):
            friction_compensation = self._user_friction_compensation_request
        elif friction_compensation is not None:
            self._user_friction_compensation_request = friction_compensation

        user_value: float | None = None
        if friction_compensation is not None:
            if friction_compensation <= 0:
                raise ValueError("friction_compensation must be positive")
            user_value = min(float(friction_compensation), self.voltage_limit)

        if motor._kt == 0.0:
            base = 0.0
        else:
            base = motor.static_friction * motor.resistance / motor._kt

        auto_value = base * self.auto_fc_gain
        if self.auto_fc_cap is not None:
            auto_value = min(auto_value, self.auto_fc_cap)
        auto_value = max(auto_value, self.auto_fc_floor)
        auto_value = min(auto_value, self.voltage_limit)

        effective = user_value if user_value is not None else auto_value

        return auto_value, user_value, effective

    def _dynamic_friction_compensation(self, position_error: float) -> float:
        e = abs(position_error)
        auto_value = self._auto_friction_compensation
        user_value = self._active_user_friction_compensation

        # 1) Never enforce a floor inside the tight tolerance band.
        if e <= self.position_tolerance:
            return 0.0

        # 2) No user value: ramp auto friction in with error, don't keep it constant.
        if user_value is None:
            # Below low threshold: no floor yet.
            if e <= self.friction_blend_error_low:
                return 0.0
            # Above high threshold: full auto_value.
            if e >= self.friction_blend_error_high:
                return min(auto_value, self.voltage_limit)

            # Between: linear ramp 0 -> auto_value.
            alpha = (e - self.friction_blend_error_low) / (
                self.friction_blend_error_high - self.friction_blend_error_low
            )
            return min(alpha * auto_value, self.voltage_limit)

        # 3) User + auto: blend between them based on error.
        if e <= self.friction_blend_error_low:
            # Near target: prefer smaller auto_value.
            return min(auto_value, self.voltage_limit)
        if e >= self.friction_blend_error_high:
            # Far away: use the stronger user value.
            return min(user_value, self.voltage_limit)

        # Between: interpolate.
        alpha = (e - self.friction_blend_error_low) / (
            self.friction_blend_error_high - self.friction_blend_error_low
        )
        blended = (1.0 - alpha) * auto_value + alpha * user_value
        return min(blended, self.voltage_limit)

    def _build_prediction_models(
        self,
        motor: BrushedMotorModel,
    ) -> Tuple[BrushedMotorModel, ...]:
        models: List[BrushedMotorModel] = [motor]

        if self.robust_electrical:
            return tuple(models)

        if self.inductance_rel_uncertainty > 0.0:
            lower_factor = max(1.0 - self.inductance_rel_uncertainty, 1e-6)
            upper_factor = 1.0 + self.inductance_rel_uncertainty

            if not math.isclose(lower_factor, 1.0, rel_tol=0.0, abs_tol=1e-12):
                models.append(
                    self._clone_motor(motor, inductance=motor.inductance * lower_factor)
                )
            if not math.isclose(upper_factor, 1.0, rel_tol=0.0, abs_tol=1e-12):
                models.append(
                    self._clone_motor(motor, inductance=motor.inductance * upper_factor)
                )

        return tuple(models)

    def _clone_motor(
        self,
        motor: BrushedMotorModel,
        *,
        inductance: float,
    ) -> BrushedMotorModel:
        return _clone_motor_with_inductance(motor, inductance=inductance)

    # ------------------------------------------------------------------
    # Internal helpers: optimisation and cost
    # ------------------------------------------------------------------

    def _evaluate_sequence(
        self,
        initial_state: Tuple[float, float, float],
        sequence: Tuple[float, ...],
    ) -> Tuple[float, Tuple[float, float, float]]:
        worst_cost = float("-inf")
        nominal_state: Tuple[float, float, float] | None = None

        for index, motor in enumerate(self._prediction_models):
            cost, state = self._evaluate_sequence_single(initial_state, sequence, motor)
            if index == 0:
                nominal_state = state
            if cost > worst_cost:
                worst_cost = cost

        if nominal_state is None:
            nominal_state = initial_state

        return worst_cost, nominal_state

    def _evaluate_sequence_single(
        self,
        initial_state: Tuple[float, float, float],
        sequence: Tuple[float, ...],
        motor: BrushedMotorModel,
    ) -> Tuple[float, Tuple[float, float, float]]:
        state = initial_state
        cost = 0.0
        previous_voltage = self._last_voltage

        for voltage in sequence:
            predicted_state = self._predict_next_for_model(state, voltage, motor)
            lvdt = self._position_to_measurement(predicted_state[2], motor=motor)
            position_error = self.target_lvdt - lvdt

            cost += self.weights.position * position_error * position_error
            cost += self.weights.speed * predicted_state[1] * predicted_state[1]
            cost += self.weights.voltage * voltage * voltage

            if previous_voltage is not None:
                delta = voltage - previous_voltage
                cost += self.weights.delta_voltage * delta * delta
            previous_voltage = voltage

            friction_floor = self._dynamic_friction_compensation(position_error)
            if (
                abs(predicted_state[1]) < motor.stop_speed_threshold
                and abs(position_error) > self.position_tolerance
                and friction_floor > 0.0
                and abs(voltage) < friction_floor
            ):
                cost += self.static_friction_penalty

            state = predicted_state

        terminal_error = self.target_lvdt - self._position_to_measurement(
            state[2], motor=motor
        )
        cost += self.weights.terminal_position * terminal_error * terminal_error

        return cost, state

    def _optimize_sequence(
        self,
        initial_state: Tuple[float, float, float],
    ) -> Tuple[float, ...]:
        horizon = self.horizon
        limit = self.voltage_limit
        eps = self._opt_eps
        step = self._opt_step

        if isinstance(self._u_seq, tuple) and len(self._u_seq) == horizon:
            last = self._u_seq
            u = [last[i + 1] if i + 1 < horizon else last[-1] for i in range(horizon)]
        else:
            base = self._last_voltage or 0.0
            u = [base for _ in range(horizon)]

        for i in range(horizon):
            u[i] = _clamp_symmetric(u[i], limit)

        for _ in range(self._opt_iters):
            for i in range(horizon):
                base = u[i]

                up = _clamp_symmetric(base + eps, limit)
                u[i] = up
                j_plus, _ = self._evaluate_sequence(initial_state, tuple(u))

                down = _clamp_symmetric(base - eps, limit)
                u[i] = down
                j_minus, _ = self._evaluate_sequence(initial_state, tuple(u))

                u[i] = base

                if math.isfinite(j_plus) and math.isfinite(j_minus):
                    grad = (j_plus - j_minus) / (2.0 * eps)
                    candidate = base - step * grad
                    if math.isfinite(candidate):
                        u[i] = _clamp_symmetric(candidate, limit)
                    else:
                        u[i] = base
                else:
                    u[i] = base

        self._u_seq = tuple(u)
        return self._u_seq

    # ------------------------------------------------------------------
    # Internal helpers: prediction and conversions
    # ------------------------------------------------------------------

    def _predict_next(
        self,
        state: Tuple[float, float, float],
        voltage: float,
    ) -> Tuple[float, float, float]:
        return self._predict_next_for_model(state, voltage, self._motor)

    def _predict_next_for_model(
        self,
        state: Tuple[float, float, float],
        voltage: float,
        motor: BrushedMotorModel | None,
    ) -> Tuple[float, float, float]:
        if motor is None:
            return state
        return _predict_next_state(
            state,
            voltage,
            motor=motor,
            dt=self.dt,
            internal_substeps=self.internal_substeps,
            robust_electrical=self.robust_electrical,
            electrical_alpha=self._electrical_alpha,
        )

    def _measurement_to_position(self, measurement: float) -> float:
        return _measurement_to_position_value(measurement, self._motor)

    def _position_to_measurement(
        self,
        position: float,
        *,
        motor: BrushedMotorModel | None = None,
    ) -> float:
        motor = motor or self._motor
        return _position_to_measurement_value(position, motor)

    def _clamp_voltage(self, voltage: float) -> float:
        return _clamp_symmetric(voltage, self.voltage_limit)
