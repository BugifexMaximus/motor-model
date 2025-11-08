"""Model predictive controller for the brushed motor model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product
from typing import Iterable, List, Tuple

from .brushed_motor import BrushedMotorModel


@dataclass
class MPCWeights:
    """Weights used by the MPC cost function."""

    position: float = 280.0
    speed: float = 1.0
    voltage: float = 0.02
    delta_voltage: float = 0.5
    terminal_position: float = 600.0


class LVDTMPCController:
    """Discrete-time MPC controller using LVDT feedback.

    The controller uses a simplified internal model of the :class:`BrushedMotorModel`
    to evaluate candidate voltage sequences. The first voltage of the sequence
    with the lowest predicted cost is applied to the motor, and the optimisation
    is repeated at the next control step (receding horizon).

    Parameters
    ----------
    motor:
        Instance of :class:`BrushedMotorModel` that provides the physical
        parameters used to build the internal model.
    dt:
        Controller time-step in seconds. This should match the integration step
        used for the plant simulation.
    horizon:
        Number of discrete steps used in the MPC look-ahead window. Larger
        horizons provide smoother behaviour at the cost of more computations.
    voltage_limit:
        Symmetric saturation limit (in Volts) applied to the controller output.
    target_lvdt:
        Desired LVDT reading expressed in the normalised range ``[-1, 1]``.
    candidate_count:
        Number of discrete voltage levels used during the optimisation. The
        value must be odd so that a zero control action exists in the grid.
    weights:
        Relative weighting applied to the MPC cost function terms.
    position_tolerance:
        Position error (normalised LVDT units) regarded as acceptable steady
        state. Errors inside this band do not trigger the static friction
        penalty.
    static_friction_penalty:
        Additional cost applied to candidate sequences that fail to command a
        voltage high enough to break static friction when the rotor is stuck
        outside the position tolerance band.
    friction_compensation:
        Minimum absolute voltage command (in Volts) used when the rotor is
        static and the error exceeds the tolerance. When ``None`` the value is
        derived from the motor parameters.
        internal_substeps:
            Number of internal integration slices used by the MPC prediction model
            within a single controller period. Setting this so that ``dt`` divided
            by ``internal_substeps`` matches the plant integration step improves
            model accuracy.
        robust_electrical:
            When ``True`` (default) the internal prediction model uses the
            quasi-static electrical approximation to eliminate inductance
            sensitivity. Setting it to ``False`` reverts to the inductive
            integration previously used.
        electrical_alpha:
            Optional smoothing factor applied when ``robust_electrical`` is
            enabled. A value of ``1.0`` snaps the predicted current to the
            steady-state, whereas smaller values low-pass filter the update.
            When omitted the controller uses ``1.0``.
        inductance_rel_uncertainty:
            Relative inductance spread used to create a min-max ensemble of
            prediction models. For example ``0.5`` evaluates each candidate
            sequence on 0.5×, 1× and 1.5× the nominal inductance and minimises
            the worst-case cost. This setting only affects the controller when
            ``robust_electrical`` is ``False`` because the quasi-static
            approximation ignores inductance dynamics.
        pd_blend:
            Weight applied to the MPC voltage before blending with the
            stabilising PD controller (``0`` = pure PD, ``1`` = pure MPC).
        pd_kp:
            Proportional gain used by the stabilising PD term.
        pd_kd:
            Derivative gain used by the stabilising PD term.
    """

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
        pd_blend: float = 0.7,
        pd_kp: float = 6.0,
        pd_kd: float = 0.4,
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

        if electrical_alpha is not None:
            if not 0.0 <= electrical_alpha <= 1.0:
                raise ValueError("electrical_alpha must be within [0, 1]")
        if inductance_rel_uncertainty < 0.0:
            raise ValueError("inductance_rel_uncertainty must be non-negative")
        if not 0.0 <= pd_blend <= 1.0:
            raise ValueError("pd_blend must be within [0, 1]")
        if pd_kp <= 0.0:
            raise ValueError("pd_kp must be positive")
        if pd_kd < 0.0:
            raise ValueError("pd_kd must be non-negative")

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

        self._candidate_count = candidate_count
        self._user_friction_compensation = friction_compensation
        self._prediction_models: Tuple[BrushedMotorModel, ...] = tuple()

        self._apply_motor_parameters(motor, friction_compensation)

        self._state: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._last_voltage: float | None = None
        self._last_measured_position: float | None = None
        self._last_measurement_time: float | None = None

    def adapt_to_motor(
        self,
        motor: BrushedMotorModel,
        *,
        candidate_count: int | None = None,
        voltage_limit: float | None = None,
        friction_compensation: float | None = None,
    ) -> None:
        """Update the internal model to match a new motor configuration.

        Parameters
        ----------
        motor:
            Instance describing the new motor parameters to track.
        candidate_count:
            Optional number of voltage candidates to use for the optimisation.
            When omitted the previous value is kept.
        voltage_limit:
            Optional new saturation limit for the controller output. When
            provided the limit is applied before recomputing the candidate set.
        friction_compensation:
            Optional manual friction compensation voltage. Passing ``None`` will
            keep the previous manual value when one was supplied during
            construction or a prior adaptation. When neither a previous value
            nor a new one are supplied the controller derives the compensation
            from the motor parameters.

        Notes
        -----
        The controller state (current estimate, speed estimate and position)
        is left untouched so that callers can manage re-initialisation
        explicitly when the operating conditions change.
        """

        if voltage_limit is not None:
            if voltage_limit <= 0:
                raise ValueError("voltage_limit must be positive")
            self.voltage_limit = voltage_limit

        if candidate_count is not None:
            if candidate_count < 3 or candidate_count % 2 == 0:
                raise ValueError("candidate_count must be an odd integer >= 3")
            self._candidate_count = candidate_count

        self._apply_motor_parameters(motor, friction_compensation)

        if self._last_voltage is not None:
            self._last_voltage = self._clamp_voltage(self._last_voltage)

    def reset(
        self,
        *,
        initial_measurement: float,
        initial_current: float,
        initial_speed: float,
    ) -> None:
        """Reset the controller internal state."""

        position = self._measurement_to_position(initial_measurement)
        self._state = (initial_current, initial_speed, position)
        self._last_voltage = None
        self._last_measured_position = position
        self._last_measurement_time = None

    def update(self, *, time: float, measurement: float) -> float:
        """Return the next control action using the latest LVDT measurement."""

        # Update the internally tracked position with the measured one.
        measured_position = self._measurement_to_position(measurement)
        normalized_measurement = self._position_to_measurement(measured_position)
        current, _, _ = self._state

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

        self._state = (current, estimated_speed, measured_position)
        self._last_measured_position = measured_position
        self._last_measurement_time = time

        best_sequence: Tuple[float, ...] | None = None
        best_cost = float("inf")

        for sequence in self._voltage_sequences():
            cost, _ = self._evaluate_sequence(self._state, sequence)
            if cost < best_cost:
                best_cost = cost
                best_sequence = sequence

        if best_sequence is None:
            return 0.0

        best_voltage = best_sequence[0]

        position_error = self.target_lvdt - normalized_measurement
        if (
            abs(self._state[1]) < self._motor.stop_speed_threshold
            and abs(position_error) > self.position_tolerance
            and abs(best_voltage) < self.friction_compensation
        ):
            direction = 1.0 if position_error >= 0.0 else -1.0
            best_voltage = direction * self.friction_compensation

        best_voltage = self._clamp_voltage(best_voltage)

        u_pd = self.pd_kp * position_error - self.pd_kd * estimated_speed
        blended_voltage = (
            self.pd_blend * best_voltage + (1.0 - self.pd_blend) * u_pd
        )
        blended_voltage = self._clamp_voltage(blended_voltage)

        if (
            abs(self._state[1]) < self._motor.stop_speed_threshold
            and abs(position_error) > self.position_tolerance
            and abs(blended_voltage) < self.friction_compensation
        ):
            direction = 1.0 if position_error >= 0.0 else -1.0
            blended_voltage = direction * self.friction_compensation
            blended_voltage = self._clamp_voltage(blended_voltage)

        self._state = self._predict_next(self._state, blended_voltage)
        self._last_voltage = blended_voltage

        return blended_voltage

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _voltage_sequences(self) -> Iterable[Tuple[float, ...]]:
        return product(self._voltage_candidates, repeat=self.horizon)

    def _apply_motor_parameters(
        self,
        motor: BrushedMotorModel,
        friction_compensation: float | None,
    ) -> None:
        self._motor = motor
        self.friction_compensation = self._determine_friction_compensation(
            motor, friction_compensation
        )
        self._voltage_candidates = self._build_voltage_candidates(self._candidate_count)
        self._prediction_models = self._build_prediction_models(motor)

    def _determine_friction_compensation(
        self,
        motor: BrushedMotorModel,
        friction_compensation: float | None,
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

    def _build_voltage_candidates(self, candidate_count: int) -> Tuple[float, ...]:
        half = candidate_count // 2
        step = self.voltage_limit / half
        values = {
            0.0,
        }
        for index in range(1, half + 1):
            voltage = index * step
            values.add(min(voltage, self.voltage_limit))
            values.add(max(-voltage, -self.voltage_limit))

        # Guarantee the availability of the minimum motion voltage in both
        # polarities to overcome static friction when necessary. This may grow
        # the candidate set beyond the requested ``candidate_count``.
        values.add(min(self.friction_compensation, self.voltage_limit))
        values.add(max(-self.friction_compensation, -self.voltage_limit))

        return tuple(sorted(values))

    def _build_prediction_models(
        self, motor: BrushedMotorModel
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
        self, motor: BrushedMotorModel, *, inductance: float
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
            lvdt_full_scale=motor.lvdt_full_scale,
            lvdt_noise_std=0.0,
            rng=motor._rng,
        )

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

            if (
                abs(predicted_state[1]) < motor.stop_speed_threshold
                and abs(position_error) > self.position_tolerance
                and abs(voltage) < self.friction_compensation
            ):
                cost += self.static_friction_penalty

            state = predicted_state

        terminal_error = self.target_lvdt - self._position_to_measurement(state[2], motor=motor)
        cost += self.weights.terminal_position * terminal_error * terminal_error

        return cost, state

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
        motor: BrushedMotorModel,
    ) -> Tuple[float, float, float]:
        current, speed, position = state

        sub_dt = self.dt / self.internal_substeps

        for _ in range(self.internal_substeps):
            if self.robust_electrical:
                back_emf = motor._ke * speed
                steady_state_current = (voltage - back_emf) / motor.resistance
                current += self._electrical_alpha * (steady_state_current - current)
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

    def _measurement_to_position(self, measurement: float) -> float:
        measurement = max(-1.0, min(1.0, measurement))
        return measurement * self._motor.lvdt_full_scale

    def _position_to_measurement(
        self,
        position: float,
        *,
        motor: BrushedMotorModel | None = None,
    ) -> float:
        motor = motor or self._motor
        normalized = position / motor.lvdt_full_scale
        return max(-1.0, min(1.0, normalized))

    def _clamp_voltage(self, voltage: float) -> float:
        return max(-self.voltage_limit, min(self.voltage_limit, voltage))
