"""Model predictive controller for the brushed motor model."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, Tuple

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

        self._motor = motor
        self.dt = dt
        self.horizon = horizon
        self.voltage_limit = voltage_limit
        self.target_lvdt = target_lvdt
        self.weights = weights or MPCWeights()
        self.position_tolerance = position_tolerance
        self.static_friction_penalty = static_friction_penalty
        self.internal_substeps = internal_substeps

        min_motion_voltage = motor.static_friction * motor.resistance / motor._kt
        if friction_compensation is not None:
            if friction_compensation <= 0:
                raise ValueError("friction_compensation must be positive")
            self.friction_compensation = min(friction_compensation, voltage_limit)
        else:
            self.friction_compensation = min(min_motion_voltage * 1.1, voltage_limit)

        self._voltage_candidates = self._build_voltage_candidates(candidate_count)

        self._state: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._last_voltage: float | None = None
        self._last_measured_position: float | None = None
        self._last_measurement_time: float | None = None

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
        self._last_measurement_time = 0.0

    def update(self, *, time: float, measurement: float) -> float:
        """Return the next control action using the latest LVDT measurement."""

        # Update the internally tracked position with the measured one.
        measured_position = self._measurement_to_position(measurement)
        current, _, _ = self._state

        estimated_speed = 0.0
        if self._last_measured_position is not None and self._last_measurement_time is not None:
            dt = time - self._last_measurement_time
            if dt > 0:
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

        position_error = self.target_lvdt - measurement
        if (
            abs(self._state[1]) < self._motor.stop_speed_threshold
            and abs(position_error) > self.position_tolerance
            and abs(best_voltage) < self.friction_compensation
        ):
            direction = 1.0 if position_error >= 0.0 else -1.0
            best_voltage = direction * self.friction_compensation

        best_voltage = self._clamp_voltage(best_voltage)
        self._state = self._predict_next(self._state, best_voltage)
        self._last_voltage = best_voltage

        return best_voltage

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _voltage_sequences(self) -> Iterable[Tuple[float, ...]]:
        return product(self._voltage_candidates, repeat=self.horizon)

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
        # polarities to overcome static friction when necessary.
        values.add(min(self.friction_compensation, self.voltage_limit))
        values.add(max(-self.friction_compensation, -self.voltage_limit))

        return tuple(sorted(values))

    def _evaluate_sequence(
        self,
        initial_state: Tuple[float, float, float],
        sequence: Tuple[float, ...],
    ) -> Tuple[float, Tuple[float, float, float]]:
        state = initial_state
        cost = 0.0
        previous_voltage = self._last_voltage

        for voltage in sequence:
            predicted_state = self._predict_next(state, voltage)
            lvdt = self._position_to_measurement(predicted_state[2])
            position_error = self.target_lvdt - lvdt
            cost += self.weights.position * position_error * position_error
            cost += self.weights.speed * predicted_state[1] * predicted_state[1]
            cost += self.weights.voltage * voltage * voltage

            if previous_voltage is not None:
                delta = voltage - previous_voltage
                cost += self.weights.delta_voltage * delta * delta
            previous_voltage = voltage

            if (
                abs(state[1]) < self._motor.stop_speed_threshold
                and abs(position_error) > self.position_tolerance
                and abs(voltage) <= self.friction_compensation
            ):
                cost += self.static_friction_penalty

            state = predicted_state

        terminal_error = self.target_lvdt - self._position_to_measurement(state[2])
        cost += self.weights.terminal_position * terminal_error * terminal_error

        return cost, state

    def _predict_next(
        self,
        state: Tuple[float, float, float],
        voltage: float,
    ) -> Tuple[float, float, float]:
        current, speed, position = state

        sub_dt = self.dt / self.internal_substeps

        for _ in range(self.internal_substeps):
            di_dt = (
                voltage
                - self._motor.resistance * current
                - self._motor._ke * speed
            ) / self._motor.inductance
            current += di_dt * sub_dt

            electromagnetic_torque = self._motor._kt * current
            spring_torque = self._motor._spring_torque(position)
            available_torque = electromagnetic_torque - spring_torque

            if (
                abs(speed) < self._motor.stop_speed_threshold
                and abs(available_torque) <= self._motor.static_friction
            ):
                speed = 0.0
            else:
                friction_direction = (
                    self._motor._sign(speed) or self._motor._sign(available_torque)
                )
                dynamic_friction = (
                    self._motor.coulomb_friction * friction_direction
                    + self._motor.viscous_friction * speed
                )
                net_torque = available_torque - dynamic_friction
                angular_acceleration = net_torque / self._motor.inertia
                speed += angular_acceleration * sub_dt

            position += speed * sub_dt

        return current, speed, position

    def _measurement_to_position(self, measurement: float) -> float:
        measurement = max(-1.0, min(1.0, measurement))
        return measurement * self._motor.lvdt_full_scale

    def _position_to_measurement(self, position: float) -> float:
        normalized = position / self._motor.lvdt_full_scale
        return max(-1.0, min(1.0, normalized))

    def _clamp_voltage(self, voltage: float) -> float:
        return max(-self.voltage_limit, min(self.voltage_limit, voltage))
