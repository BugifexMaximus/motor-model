"""Utility helpers for interactive and automated MPC simulations."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Tuple, Literal

from .brushed_motor import BrushedMotorModel, rpm_per_volt_to_rad_per_sec_per_volt
from .mpc_controller import LVDTMPCController, MPCWeights
from .tube_mpc_controller import TubeMPCController

ControllerName = Literal["lvdtnom", "tube"]

_CONTROLLER_CLASSES = {
    "lvdtnom": LVDTMPCController,
    "tube": TubeMPCController,
}


@dataclass(frozen=True)
class SimulationState:
    """Snapshot of the motor/controller state for quick inspection."""

    time: float
    current: float
    speed: float
    position: float
    voltage: float


@dataclass(frozen=True)
class SimulationHistory:
    """Container with the most recent time-series samples."""

    time: Tuple[float, ...]
    position: Tuple[float, ...]
    setpoint: Tuple[float, ...]
    voltage: Tuple[float, ...]
    speed: Tuple[float, ...]
    current: Tuple[float, ...]


class MotorSimulation:
    """Lightweight time-domain simulation suitable for interactive use.

    The class advances the :class:`~motor_model.brushed_motor.BrushedMotorModel`
    using an explicit Euler integrator and keeps a short rolling history of the
    state variables.  It can wrap either the standard
    :class:`~motor_model.mpc_controller.LVDTMPCController` or the
    :class:`~motor_model.tube_mpc_controller.TubeMPCController` depending on the
    selected ``controller_type``. When the controller and plant parameters
    match, the simulation mirrors the simplified internal model used by the
    MPC, providing an inexpensive environment for the GUI and unit tests to
    exercise the control loop.
    """

    def __init__(
        self,
        motor_kwargs: Dict[str, float],
        controller_kwargs: Dict[str, float],
        *,
        history_duration: float = 10.0,
        max_points: int = 8000,
        controller_motor_kwargs: Dict[str, float] | None = None,
        controller_type: ControllerName = "lvdtnom",
    ) -> None:
        self._motor_kwargs = dict(motor_kwargs)
        self._controller_kwargs = dict(controller_kwargs)
        self._controller_motor_kwargs = (
            dict(controller_motor_kwargs)
            if controller_motor_kwargs is not None
            else dict(motor_kwargs)
        )
        if controller_type not in _CONTROLLER_CLASSES:
            valid = ", ".join(sorted(_CONTROLLER_CLASSES))
            raise ValueError(f"Unsupported controller_type '{controller_type}'. Choose from: {valid}.")

        self.history_duration = history_duration
        self.max_points = max_points
        self._history_max_points = max_points
        self._controller_type: ControllerName = controller_type
        self.reset()

    # ------------------------------------------------------------------
    # Life-cycle helpers
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset the motor, controller and simulation history."""

        self.motor = BrushedMotorModel(**self._motor_kwargs)
        controller_motor = BrushedMotorModel(**self._controller_motor_kwargs)

        controller_kwargs = dict(self._controller_kwargs)
        weights = controller_kwargs.pop("weights", MPCWeights())
        if not isinstance(weights, MPCWeights):
            raise TypeError("weights must be an MPCWeights instance")
        controller_cls = _CONTROLLER_CLASSES[self._controller_type]
        self.controller = controller_cls(
            controller_motor, weights=weights, **controller_kwargs
        )

        substeps = max(1, controller_kwargs.get("internal_substeps", 1))
        self.plant_dt = self.controller.dt / substeps
        self.measurement_steps = max(1, int(round(self.controller.dt / self.plant_dt)))
        self._steps_since_measurement = 0

        required_points = max(2, int(math.ceil(self.history_duration / self.plant_dt)) + 1)
        self._history_max_points = max(self.max_points, required_points)

        self.time = 0.0
        self.current = 0.0
        self.speed = 0.0
        self.position = 0.0

        initial_measurement = self.motor._lvdt_measurement(self.position)
        self.controller.reset(
            initial_measurement=initial_measurement,
            initial_current=self.current,
            initial_speed=self.speed,
        )
        self.voltage = self.controller.update(time=0.0, measurement=initial_measurement)

        self.time_history: Deque[float] = deque([0.0], maxlen=self._history_max_points)
        self.position_history: Deque[float] = deque([self.position], maxlen=self._history_max_points)
        self.setpoint_history: Deque[float] = deque([self.target_position()], maxlen=self._history_max_points)
        self.voltage_history: Deque[float] = deque([self.voltage], maxlen=self._history_max_points)
        self.speed_history: Deque[float] = deque([self.speed], maxlen=self._history_max_points)
        self.current_history: Deque[float] = deque([self.current], maxlen=self._history_max_points)

    # ------------------------------------------------------------------
    # Public API used by the GUI and the unit tests
    # ------------------------------------------------------------------
    def plant_dt_ms(self) -> float:
        return self.plant_dt * 1000.0

    def target_position(self) -> float:
        return self.controller.target_lvdt * self.motor.lvdt_full_scale

    def set_target_position(self, position: float) -> None:
        if self.motor.lvdt_full_scale <= 0:
            return
        measurement = max(-1.0, min(1.0, position / self.motor.lvdt_full_scale))
        self.controller.target_lvdt = measurement
        if self.setpoint_history:
            self.setpoint_history[-1] = self.target_position()

    def step(self, steps: int) -> None:
        for _ in range(steps):
            self._single_step()

    def run_for(self, duration: float) -> None:
        """Advance the simulation for ``duration`` seconds."""

        if duration <= 0:
            raise ValueError("duration must be positive")
        steps = max(1, int(math.ceil(duration / self.plant_dt)))
        self.step(steps)

    def state(self) -> SimulationState:
        return SimulationState(
            time=self.time,
            current=self.current,
            speed=self.speed,
            position=self.position,
            voltage=self.voltage,
        )

    def history(self) -> SimulationHistory:
        return SimulationHistory(
            time=tuple(self.time_history),
            position=tuple(self.position_history),
            setpoint=tuple(self.setpoint_history),
            voltage=tuple(self.voltage_history),
            speed=tuple(self.speed_history),
            current=tuple(self.current_history),
        )

    # ------------------------------------------------------------------
    # Internal mechanics
    # ------------------------------------------------------------------
    def _single_step(self) -> None:
        dt = self.plant_dt

        back_emf = self.motor._ke * self.speed
        di_dt = (self.voltage - self.motor.resistance * self.current - back_emf) / self.motor.inductance
        self.current += di_dt * dt

        electromagnetic_torque = self.motor._kt * self.current
        spring_torque = self.motor._spring_torque(self.position)
        available_torque = electromagnetic_torque - spring_torque

        if (
            abs(self.speed) < self.motor.stop_speed_threshold
            and abs(available_torque) <= self.motor.static_friction
        ):
            self.speed = 0.0
        else:
            friction_direction = self.motor._sign(self.speed) or self.motor._sign(available_torque)
            dynamic_friction = (
                self.motor.coulomb_friction * friction_direction + self.motor.viscous_friction * self.speed
            )
            angular_accel = (available_torque - dynamic_friction) / self.motor.inertia
            self.speed += angular_accel * dt

        self.position += self.speed * dt
        self.time += dt

        self.time_history.append(self.time)
        self.position_history.append(self.position)
        self.setpoint_history.append(self.target_position())
        self.voltage_history.append(self.voltage)
        self.speed_history.append(self.speed)
        self.current_history.append(self.current)

        self._steps_since_measurement += 1
        if self._steps_since_measurement >= self.measurement_steps:
            measurement = self.motor._lvdt_measurement(self.position)
            self.voltage = self.controller.update(time=self.time, measurement=measurement)
            self._steps_since_measurement = 0

        self._trim_history()

    def _trim_history(self) -> None:
        if not self.time_history:
            return
        min_time = self.time - self.history_duration
        while self.time_history and self.time_history[0] < min_time:
            self.time_history.popleft()
            self.position_history.popleft()
            self.setpoint_history.popleft()
            self.voltage_history.popleft()
            self.speed_history.popleft()
            self.current_history.popleft()


def build_default_motor_kwargs(**overrides: float) -> Dict[str, float]:
    """Return a ``BrushedMotorModel`` kwargs dictionary useful for tests."""

    kwargs: Dict[str, float] = {
        "resistance": 28.0,
        "inductance": 16e-3,
        "kv": rpm_per_volt_to_rad_per_sec_per_volt(7.0),
        "inertia": 4.8e-4,
        "viscous_friction": 1.9e-4,
        "coulomb_friction": 2.1e-2,
        "static_friction": 2.4e-2,
        "stop_speed_threshold": 1e-4,
        "spring_constant": 9.5e-4,
        "spring_compression_ratio": 0.4,
        "lvdt_full_scale": math.radians(30.0),
        "lvdt_noise_std": 0.0,
    }
    kwargs.update(overrides)
    return kwargs


def build_default_controller_kwargs(**overrides: float) -> Dict[str, float]:
    """Return ``LVDTMPCController`` kwargs paired with the defaults above."""

    kwargs: Dict[str, float] = {
        "dt": 0.005,
        "horizon": 4,
        "voltage_limit": 10.0,
        "target_lvdt": 0.0,
        "candidate_count": 5,
        "position_tolerance": 0.02,
        "static_friction_penalty": 50.0,
        "internal_substeps": 5,
        "weights": MPCWeights(),
    }
    kwargs.update(overrides)
    return kwargs


def build_default_tube_controller_kwargs(**overrides: float) -> Dict[str, float]:
    """Return ``TubeMPCController`` kwargs paired with the defaults above."""

    kwargs: Dict[str, float] = {
        "dt": 0.005,
        "horizon": 4,
        "voltage_limit": 10.0,
        "target_lvdt": 0.0,
        "candidate_count": 5,
        "position_tolerance": 0.02,
        "static_friction_penalty": 50.0,
        "internal_substeps": 5,
        "inductance_rel_uncertainty": 0.5,
        "tube_tolerance": 1e-6,
        "tube_max_iterations": 500,
        "lqr_state_weight": (2.0, 0.2, 5.0),
        "lqr_input_weight": 0.5,
        "weights": MPCWeights(),
    }
    kwargs.update(overrides)
    return kwargs


__all__ = [
    "MotorSimulation",
    "SimulationState",
    "SimulationHistory",
    "build_default_controller_kwargs",
    "build_default_tube_controller_kwargs",
    "build_default_motor_kwargs",
]

