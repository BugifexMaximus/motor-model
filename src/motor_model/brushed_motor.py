"""A simple brushed DC motor model."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Protocol


class VoltageSource(Protocol):
    """Protocol for time-dependent voltage sources used in simulations."""

    def __call__(self, time: float) -> float:  # pragma: no cover - protocol definition
        ...


class FeedbackController(Protocol):
    """Protocol for controllers using feedback measurements to set voltage."""

    def reset(
        self,
        *,
        initial_measurement: float,
        initial_current: float,
        initial_speed: float,
    ) -> None:  # pragma: no cover - protocol definition
        ...

    def update(self, *, time: float, measurement: float) -> float:  # pragma: no cover
        ...


@dataclass(frozen=True)
class SimulationResult:
    """Stores time-series data produced by a motor simulation."""

    time: List[float]
    current: List[float]
    speed: List[float]
    position: List[float]
    torque: List[float]
    voltage: List[float]
    lvdt_time: List[float]
    lvdt: List[float]


class BrushedMotorModel:
    """A simple lumped-parameter brushed DC motor model.

    Parameters
    ----------
    resistance: float
        Winding resistance (Ohms).
    inductance: float
        Winding inductance (Henries).
    kv: float
        Speed constant in rad/s per Volt. The electrical constant (back-EMF)
        and torque constant are derived from this value.
    inertia: float
        Rotor inertia (kg*m^2).
    viscous_friction: float
        Viscous friction coefficient (N*m*s/rad).
    coulomb_friction: float
        Coulomb (dynamic) friction torque (N*m).
    static_friction: float
        Static friction torque threshold (N*m).
    stop_speed_threshold: float
        Speed (rad/s) below which the rotor is considered stationary for
        static-friction logic.
    """

    def __init__(
        self,
        *,
        resistance: float = 28.0,
        inductance: float = 16e-3,
        kv: float = 7.0,
        inertia: float = 5e-5,
        viscous_friction: float = 2e-5,
        coulomb_friction: float = 2.2e-3,
        static_friction: float = 2.5e-3,
        stop_speed_threshold: float = 1e-4,
        spring_constant: float = 1e-4,
        spring_compression_ratio: float = 0.4,
        lvdt_full_scale: float = 0.1,
        lvdt_noise_std: float = 5e-3,
        rng: random.Random | None = None,
    ) -> None:
        if kv <= 0:
            raise ValueError("kv must be positive")
        if resistance <= 0:
            raise ValueError("resistance must be positive")
        if inductance <= 0:
            raise ValueError("inductance must be positive")
        if inertia <= 0:
            raise ValueError("inertia must be positive")
        if spring_constant < 0:
            raise ValueError("spring_constant must be non-negative")
        if not 0.0 <= spring_compression_ratio <= 1.0:
            raise ValueError("spring_compression_ratio must be between 0 and 1")
        if lvdt_full_scale <= 0:
            raise ValueError("lvdt_full_scale must be positive")
        if lvdt_noise_std < 0:
            raise ValueError("lvdt_noise_std must be non-negative")

        self.resistance = resistance
        self.inductance = inductance
        self.kv = kv
        self.inertia = inertia
        self.viscous_friction = viscous_friction
        self.coulomb_friction = coulomb_friction
        self.static_friction = static_friction
        self.stop_speed_threshold = stop_speed_threshold
        self.spring_constant = spring_constant
        self.spring_compression_ratio = spring_compression_ratio
        self.lvdt_full_scale = lvdt_full_scale
        self.lvdt_noise_std = lvdt_noise_std
        self._rng = rng or random.Random()

        # Electrical constant ke and torque constant kt in SI units.
        self._ke = 1.0 / kv
        self._kt = 1.0 / kv

    @staticmethod
    def _sign(value: float) -> float:
        if value > 0:
            return 1.0
        if value < 0:
            return -1.0
        return 0.0

    def simulate(
        self,
        voltage: VoltageSource | float,
        *,
        duration: float,
        dt: float,
        initial_speed: float = 0.0,
        initial_current: float = 0.0,
        load_torque: VoltageSource | float = 0.0,
        measurement_period: float | None = None,
        controller_period: float | None = None,
    ) -> SimulationResult:
        """Simulate the motor response to a voltage source.

        Parameters
        ----------
        voltage:
            Either a callable ``voltage(t)`` or a constant voltage in Volts.
        duration:
            Total simulation time in seconds.
        dt:
            Integration time-step in seconds.
        initial_speed:
            Initial mechanical angular speed in rad/s.
        initial_current:
            Initial phase current in Amps.
        load_torque:
            External load torque opposing motion. Either a callable of time or
            a constant torque in N*m.
        measurement_period:
            Interval between LVDT samples recorded in the simulation results.
            Must be a positive multiple of ``dt``. Defaults to ``dt``.
        controller_period:
            Period between controller updates when ``voltage`` implements the
            :class:`FeedbackController` protocol. Must be a positive multiple
            of ``dt``. Defaults to ``measurement_period`` when omitted.
        """

        if dt <= 0:
            raise ValueError("dt must be positive")
        if duration <= 0:
            raise ValueError("duration must be positive")

        steps = int(round(duration / dt))
        if not math.isclose(steps * dt, duration, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("duration must be an integer multiple of dt")
        time_values: List[float] = []
        current_values: List[float] = []
        speed_values: List[float] = []
        position_values: List[float] = []
        torque_values: List[float] = []
        voltage_values: List[float] = []
        lvdt_values: List[float] = []

        current = initial_current
        speed = initial_speed
        position = 0.0

        feedback_controller: FeedbackController | None = None
        voltage_source: VoltageSource | None = None
        measurement_period = measurement_period or dt
        if measurement_period <= 0:
            raise ValueError("measurement_period must be positive")
        measurement_steps = int(round(measurement_period / dt))
        if measurement_steps <= 0 or not math.isclose(
            measurement_steps * dt, measurement_period, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError("measurement_period must be a multiple of dt")

        controller_steps: int | None = None

        initial_measurement = self._lvdt_measurement(position)

        if hasattr(voltage, "update") and callable(getattr(voltage, "update")):
            feedback_controller = voltage  # type: ignore[assignment]
            reset = getattr(feedback_controller, "reset", None)
            if callable(reset):
                reset(
                    initial_measurement=initial_measurement,
                    initial_current=current,
                    initial_speed=speed,
                )
            controller_period = controller_period or measurement_period
            if controller_period <= 0:
                raise ValueError("controller_period must be positive")
            controller_steps = int(round(controller_period / dt))
            if controller_steps <= 0 or not math.isclose(
                controller_steps * dt,
                controller_period,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError("controller_period must be a multiple of dt")
        else:
            voltage_source = self._as_callable(voltage)

        load_source = self._as_callable(load_torque)

        lvdt_time_values: List[float] = []
        lvdt_values: List[float] = []

        if feedback_controller is not None:
            assert controller_steps is not None
            voltage_command = feedback_controller.update(
                time=0.0, measurement=initial_measurement
            )
        else:
            assert voltage_source is not None
            voltage_command = voltage_source(0.0)

        voltage_values.append(voltage_command)
        lvdt_time_values.append(0.0)
        lvdt_values.append(initial_measurement)

        time_values.append(0.0)
        current_values.append(current)
        speed_values.append(speed)
        position_values.append(position)
        torque_values.append(self._kt * current)

        for step in range(steps):
            t = step * dt
            load = load_source(t)

            # Electrical subsystem: di/dt = (V - Ri - ke * omega) / L
            back_emf = self._ke * speed
            di_dt = (
                voltage_command - self.resistance * current - back_emf
            ) / self.inductance
            current += di_dt * dt

            electromagnetic_torque = self._kt * current
            spring_torque = self._spring_torque(position)
            available_torque = electromagnetic_torque - load - spring_torque

            if abs(speed) < self.stop_speed_threshold and abs(available_torque) <= self.static_friction:
                # Static friction prevents motion. Clamp speed output.
                speed = 0.0
            else:
                friction_direction = self._sign(speed) or self._sign(available_torque)
                dynamic_friction = self.coulomb_friction * friction_direction + self.viscous_friction * speed
                net_torque = available_torque - dynamic_friction
                angular_accel = net_torque / self.inertia
                speed += angular_accel * dt

            position += speed * dt

            next_time = (step + 1) * dt

            time_values.append(next_time)
            current_values.append(current)
            speed_values.append(speed)
            position_values.append(position)
            torque_values.append(electromagnetic_torque)
            voltage_values.append(voltage_command)

            if feedback_controller is not None:
                assert controller_steps is not None
                if (step + 1) % measurement_steps == 0:
                    measurement_time = next_time
                    measurement = self._lvdt_measurement(position)
                    lvdt_time_values.append(measurement_time)
                    lvdt_values.append(measurement)
                    if (step + 1) % controller_steps == 0:
                        voltage_command = feedback_controller.update(
                            time=measurement_time, measurement=measurement
                        )
            else:
                if (step + 1) % measurement_steps == 0:
                    measurement_time = next_time
                    measurement = self._lvdt_measurement(position)
                    lvdt_time_values.append(measurement_time)
                    lvdt_values.append(measurement)
                assert voltage_source is not None
                voltage_command = voltage_source(next_time)

        return SimulationResult(
            time=time_values,
            current=current_values,
            speed=speed_values,
            position=position_values,
            torque=torque_values,
            voltage=voltage_values,
            lvdt_time=lvdt_time_values,
            lvdt=lvdt_values,
        )

    @staticmethod
    def _as_callable(source: VoltageSource | float) -> VoltageSource:
        if callable(source):
            return source  # type: ignore[return-value]

        def constant(_: float, value: float = float(source)) -> float:
            return value

        return constant

    def _spring_torque(self, position: float) -> float:
        if self.spring_constant == 0.0:
            return 0.0
        if position >= 0.0:
            return self.spring_constant * position
        return self.spring_constant * self.spring_compression_ratio * position

    def _lvdt_measurement(self, position: float) -> float:
        normalized = position / self.lvdt_full_scale
        if self.lvdt_noise_std > 0.0:
            noise = self._rng.gauss(0.0, self.lvdt_noise_std)
            normalized += noise
        return max(-1.0, min(1.0, normalized))
