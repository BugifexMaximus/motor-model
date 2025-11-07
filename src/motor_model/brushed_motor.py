"""A simple brushed DC motor model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol


class VoltageSource(Protocol):
    """Protocol for time-dependent voltage sources used in simulations."""

    def __call__(self, time: float) -> float:  # pragma: no cover - protocol definition
        ...


@dataclass(frozen=True)
class SimulationResult:
    """Stores time-series data produced by a motor simulation."""

    time: List[float]
    current: List[float]
    speed: List[float]
    position: List[float]
    torque: List[float]


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
    ) -> None:
        if kv <= 0:
            raise ValueError("kv must be positive")
        if resistance <= 0:
            raise ValueError("resistance must be positive")
        if inductance <= 0:
            raise ValueError("inductance must be positive")
        if inertia <= 0:
            raise ValueError("inertia must be positive")

        self.resistance = resistance
        self.inductance = inductance
        self.kv = kv
        self.inertia = inertia
        self.viscous_friction = viscous_friction
        self.coulomb_friction = coulomb_friction
        self.static_friction = static_friction
        self.stop_speed_threshold = stop_speed_threshold

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
        """

        if dt <= 0:
            raise ValueError("dt must be positive")
        if duration <= 0:
            raise ValueError("duration must be positive")

        voltage_source = self._as_callable(voltage)
        load_source = self._as_callable(load_torque)

        steps = int(duration / dt)
        time_values: List[float] = []
        current_values: List[float] = []
        speed_values: List[float] = []
        position_values: List[float] = []
        torque_values: List[float] = []

        current = initial_current
        speed = initial_speed
        position = 0.0

        for step in range(steps + 1):
            t = step * dt
            v = voltage_source(t)
            load = load_source(t)

            # Electrical subsystem: di/dt = (V - Ri - ke * omega) / L
            back_emf = self._ke * speed
            di_dt = (v - self.resistance * current - back_emf) / self.inductance
            current += di_dt * dt

            electromagnetic_torque = self._kt * current
            available_torque = electromagnetic_torque - load

            if abs(speed) < self.stop_speed_threshold and abs(available_torque) <= self.static_friction:
                # Static friction prevents motion. Clamp speed and torque output.
                speed = 0.0
                net_torque = 0.0
            else:
                friction_direction = self._sign(speed) or self._sign(available_torque)
                dynamic_friction = self.coulomb_friction * friction_direction + self.viscous_friction * speed
                net_torque = available_torque - dynamic_friction
                angular_accel = net_torque / self.inertia
                speed += angular_accel * dt

            position += speed * dt

            time_values.append(t)
            current_values.append(current)
            speed_values.append(speed)
            position_values.append(position)
            torque_values.append(electromagnetic_torque)

        return SimulationResult(
            time=time_values,
            current=current_values,
            speed=speed_values,
            position=position_values,
            torque=torque_values,
        )

    @staticmethod
    def _as_callable(source: VoltageSource | float) -> VoltageSource:
        if callable(source):
            return source  # type: ignore[return-value]

        def constant(_: float, value: float = float(source)) -> float:
            return value

        return constant
