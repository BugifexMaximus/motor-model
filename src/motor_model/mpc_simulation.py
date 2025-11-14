"""Utility helpers for interactive and automated MPC simulations."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Literal, Tuple

from ._mpc_common import MPCWeights
from .brushed_motor import BrushedMotorModel, rpm_per_volt_to_rad_per_sec_per_volt
from .continuous_mpc_controller import ContMPCController
from .mpc_controller import LVDTMPCController
from .tube_mpc_controller import TubeMPCController
from ._native import load_brushed_motor, load_continuous_mpc

ControllerName = Literal["lvdtnom", "tube", "continuous", "continuous_native"]
MotorBackend = Literal["python", "native"]

_PYTHON_CONTROLLER_CLASSES = {
    "lvdtnom": LVDTMPCController,
    "tube": TubeMPCController,
    "continuous": ContMPCController,
}


@dataclass(frozen=True)
class SimulationState:
    """Snapshot of the motor/controller state for quick inspection."""

    time: float
    current: float
    speed: float
    position: float
    voltage: float
    disturbance: float


@dataclass(frozen=True)
class SimulationHistory:
    """Container with the most recent time-series samples."""

    time: Tuple[float, ...]
    position: Tuple[float, ...]
    setpoint: Tuple[float, ...]
    voltage: Tuple[float, ...]
    speed: Tuple[float, ...]
    current: Tuple[float, ...]
    disturbance: Tuple[float, ...]


@dataclass
class _TorqueDisturbance:
    """Internal representation of a scheduled torque disturbance."""

    start: float
    end: float
    torque: float


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
        controller_kwargs: Dict[str, object],
        *,
        history_duration: float = 10.0,
        max_points: int = 8000,
        controller_motor_kwargs: Dict[str, float] | None = None,
        controller_type: ControllerName = "lvdtnom",
        motor_backend: MotorBackend = "python",
    ) -> None:
        self._motor_kwargs = dict(motor_kwargs)
        self._controller_kwargs = dict(controller_kwargs)
        self._controller_motor_kwargs = (
            dict(controller_motor_kwargs)
            if controller_motor_kwargs is not None
            else dict(motor_kwargs)
        )
        supported = set(_PYTHON_CONTROLLER_CLASSES) | {"continuous_native"}
        if controller_type not in supported:
            valid = ", ".join(sorted(supported))
            raise ValueError(f"Unsupported controller_type '{controller_type}'. Choose from: {valid}.")

        if motor_backend not in {"python", "native"}:
            valid = ", ".join(sorted({"python", "native"}))
            raise ValueError(
                f"Unsupported motor_backend '{motor_backend}'. Choose from: {valid}."
            )

        self.history_duration = history_duration
        self.max_points = max_points
        self._history_max_points = max_points
        self._controller_type: ControllerName = controller_type
        self._native_manager: Any | None = None
        self._motor_backend: MotorBackend = motor_backend
        self.reset()

    # ------------------------------------------------------------------
    # Life-cycle helpers
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset the motor, controller and simulation history."""

        self._stop_native_manager()

        self._last_pi_integrator = math.nan
        self._last_model_integrator = math.nan
        self._last_planned_voltage: Tuple[float, ...] = ()

        self.motor = self._create_motor(self._motor_kwargs)
        controller_motor = BrushedMotorModel(**self._controller_motor_kwargs)

        controller_kwargs = dict(self._controller_kwargs)
        weights = controller_kwargs.pop("weights", MPCWeights())
        if not isinstance(weights, MPCWeights):
            raise TypeError("weights must be an MPCWeights instance")

        if self._controller_type == "continuous_native":
            try:
                native_module = load_continuous_mpc()
            except ImportError as exc:  # pragma: no cover - import failure is environmental
                raise RuntimeError(
                    "The native continuous MPC module is not available. "
                    "Build the extension before selecting the C++ controller."
                ) from exc

            self.controller = native_module.ContMPCController(
                motor=controller_motor,
                weights=weights,
                **controller_kwargs,
            )
            self._native_manager = native_module.ContMPCControllerManager(self.controller)
        else:
            controller_cls = _PYTHON_CONTROLLER_CLASSES[self._controller_type]
            self.controller = controller_cls(
                controller_motor, weights=weights, **controller_kwargs
            )
            self._native_manager = None

        substeps = max(
            1,
            getattr(
                self.controller,
                "internal_substeps",
                controller_kwargs.get("internal_substeps", 1),
            ),
        )
        self.motor.integration_substeps = substeps
        controller_motor.integration_substeps = substeps
        self.plant_dt = self.controller.dt / substeps
        self.measurement_steps = max(1, int(round(self.controller.dt / self.plant_dt)))
        self._steps_since_measurement = 0

        required_points = max(2, int(math.ceil(self.history_duration / self.plant_dt)) + 1)
        self._history_max_points = max(self.max_points, required_points)

        self.time = 0.0
        self.current = 0.0
        self.speed = 0.0
        self.position = 0.0
        self.disturbance_torque = 0.0
        self._manual_torque = 0.0

        initial_measurement = self.motor._lvdt_measurement(self.position)
        self.controller.reset(
            initial_measurement=initial_measurement,
            initial_current=self.current,
            initial_speed=self.speed,
        )

        if self._native_manager is not None:
            future = self._native_manager.submit_step(
                time=0.0, measurement=initial_measurement
            )
            self.voltage = future.result()
        else:
            self.voltage = self.controller.update(
                time=0.0, measurement=initial_measurement
            )

        self._capture_diagnostics()

        self.time_history: Deque[float] = deque([0.0], maxlen=self._history_max_points)
        self.position_history: Deque[float] = deque([self.position], maxlen=self._history_max_points)
        self.setpoint_history: Deque[float] = deque([self.target_position()], maxlen=self._history_max_points)
        self.voltage_history: Deque[float] = deque([self.voltage], maxlen=self._history_max_points)
        self.speed_history: Deque[float] = deque([self.speed], maxlen=self._history_max_points)
        self.current_history: Deque[float] = deque([self.current], maxlen=self._history_max_points)
        self.disturbance_history: Deque[float] = deque([self.disturbance_torque], maxlen=self._history_max_points)
        self.lvdt_time_history: Deque[float] = deque([0.0], maxlen=self._history_max_points)
        self.lvdt_history: Deque[float] = deque([initial_measurement], maxlen=self._history_max_points)
        self.pi_integrator_history: Deque[float] = deque([
            self._last_pi_integrator
        ], maxlen=self._history_max_points)
        self.model_integrator_history: Deque[float] = deque([
            self._last_model_integrator
        ], maxlen=self._history_max_points)
        self.planned_voltage_history: Deque[Tuple[float, ...]] = deque([
            self._last_planned_voltage
        ], maxlen=self._history_max_points)
        self._torque_disturbances: list[_TorqueDisturbance] = []

    def motor_backend(self) -> MotorBackend:
        return self._motor_backend

    def _create_motor(self, kwargs: Dict[str, float]) -> Any:
        if self._motor_backend == "native":
            try:
                native_module = load_brushed_motor()
            except ImportError as exc:  # pragma: no cover - import failure depends on build
                raise RuntimeError(
                    "The native brushed motor module is not available. "
                    "Build the extension before selecting the C++ motor."
                ) from exc

            native_kwargs = dict(kwargs)
            if "rng" in native_kwargs:
                raise ValueError("Native motor backend does not support custom RNG instances")
            motor = native_module.BrushedMotorModel(**native_kwargs)
            return motor
        return BrushedMotorModel(**kwargs)

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        try:
            self._stop_native_manager()
        except Exception:
            pass

    def _stop_native_manager(self) -> None:
        if self._native_manager is not None:
            try:
                self._native_manager.stop()
            finally:
                self._native_manager = None

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
            disturbance=self.disturbance_torque,
        )

    def history(self) -> SimulationHistory:
        return SimulationHistory(
            time=tuple(self.time_history),
            position=tuple(self.position_history),
            setpoint=tuple(self.setpoint_history),
            voltage=tuple(self.voltage_history),
            speed=tuple(self.speed_history),
            current=tuple(self.current_history),
            disturbance=tuple(self.disturbance_history),
        )

    def apply_torque_disturbance(self, torque: float, duration: float) -> None:
        """Schedule a constant torque disturbance applied immediately."""

        if not math.isfinite(torque):
            raise ValueError("torque must be a finite value")
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("duration must be a positive finite value")

        start = self.time
        disturbance = _TorqueDisturbance(start=start, end=start + duration, torque=torque)
        self._torque_disturbances.append(disturbance)

    def set_manual_torque(self, torque: float) -> None:
        """Set a continuously applied torque disturbance."""

        if not math.isfinite(torque):
            raise ValueError("torque must be a finite value")
        self._manual_torque = float(torque)

    # ------------------------------------------------------------------
    # Internal mechanics
    # ------------------------------------------------------------------
    def _single_step(self) -> None:
        dt = self.plant_dt

        self.disturbance_torque = self._active_disturbance_torque()

        back_emf = self.motor._ke * self.speed
        di_dt = (self.voltage - self.motor.resistance * self.current - back_emf) / self.motor.inductance
        self.current += di_dt * dt

        electromagnetic_torque = self.motor._kt * self.current
        spring_torque = self.motor._spring_torque(self.position)
        available_torque = electromagnetic_torque - spring_torque - self.disturbance_torque

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
        self.disturbance_history.append(self.disturbance_torque)

        self._steps_since_measurement += 1
        if self._steps_since_measurement >= self.measurement_steps:
            measurement = self.motor._lvdt_measurement(self.position)
            if self._native_manager is not None:
                future = self._native_manager.submit_step(
                    time=self.time, measurement=measurement
                )
                self.voltage = future.result()
            else:
                self.voltage = self.controller.update(
                    time=self.time, measurement=measurement
                )
            self._steps_since_measurement = 0
            self._capture_diagnostics()
            self._record_measurement(self.time, measurement)

        self._append_diagnostics_sample()
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
            self.disturbance_history.popleft()
            self.pi_integrator_history.popleft()
            self.model_integrator_history.popleft()
            self.planned_voltage_history.popleft()

        while self.lvdt_time_history and self.lvdt_time_history[0] < min_time:
            self.lvdt_time_history.popleft()
            self.lvdt_history.popleft()

    def _active_disturbance_torque(self) -> float:
        """Return the total torque of disturbances active at the current time."""

        scheduled = 0.0
        if self._torque_disturbances:
            now = self.time
            remaining: list[_TorqueDisturbance] = []
            for disturbance in self._torque_disturbances:
                if now >= disturbance.end:
                    continue
                if now >= disturbance.start:
                    scheduled += disturbance.torque
                remaining.append(disturbance)
            self._torque_disturbances = remaining

        return self._manual_torque + scheduled

    def _capture_diagnostics(self) -> None:
        controller = getattr(self, "controller", None)
        pi_value = math.nan
        model_value = math.nan
        planned: Tuple[float, ...] = ()

        if controller is not None:
            pi_raw = self._controller_attribute(controller, ("_int_err", "_position_integral"))
            gain = self._controller_attribute(controller, ("pi_ki", "_integral_gain"))
            if pi_raw is not None:
                try:
                    scale = float(gain) if gain is not None else 1.0
                    pi_value = float(pi_raw) * scale
                except (TypeError, ValueError):
                    pi_value = math.nan

            model_raw = self._controller_attribute(controller, ("_u_bias",))
            if model_raw is not None:
                try:
                    model_value = float(model_raw)
                except (TypeError, ValueError):
                    model_value = math.nan

            plan_attr = getattr(controller, "_u_seq", None)
            if plan_attr is not None:
                try:
                    planned = tuple(float(value) for value in plan_attr)
                except (TypeError, ValueError):
                    planned = ()

        self._last_pi_integrator = pi_value
        self._last_model_integrator = model_value
        self._last_planned_voltage = planned

    @staticmethod
    def _controller_attribute(controller: object, names: Tuple[str, ...]) -> float | None:
        for name in names:
            if hasattr(controller, name):
                try:
                    return float(getattr(controller, name))
                except (TypeError, ValueError):
                    return None
        return None

    def _append_diagnostics_sample(self) -> None:
        self.pi_integrator_history.append(self._last_pi_integrator)
        self.model_integrator_history.append(self._last_model_integrator)
        self.planned_voltage_history.append(self._last_planned_voltage)

    def _record_measurement(self, time: float, measurement: float) -> None:
        self.lvdt_time_history.append(time)
        self.lvdt_history.append(measurement)


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


def build_default_controller_kwargs(**overrides: object) -> Dict[str, object]:
    """Return ``LVDTMPCController`` kwargs paired with the defaults above."""

    kwargs: Dict[str, object] = {
        "dt": 0.005,
        "horizon": 3,
        "voltage_limit": 10.0,
        "target_lvdt": 0.0,
        "candidate_count": 5,
        "position_tolerance": 0.02,
        "static_friction_penalty": 50.0,
        "internal_substeps": 30,
        "weights": MPCWeights(),
        "auto_fc_gain": 2.5,
        "auto_fc_floor": 0.0,
        "auto_fc_cap": None,
        "friction_blend_error_low": 0.05,
        "friction_blend_error_high": 0.2,
        "pd_blend": 0.7,
        "pi_ki": 0.0,
        "pi_limit": 5.0,
        "pi_gate_saturation": True,
        "pi_gate_blocked": True,
        "pi_gate_error_band": True,
        "pi_leak_near_setpoint": True,
        "use_model_integrator": False,
    }
    kwargs.update(overrides)
    return kwargs


def build_default_continuous_controller_kwargs(**overrides: object) -> Dict[str, object]:
    """Return ``ContMPCController`` kwargs paired with the defaults above."""

    kwargs: Dict[str, object] = {
        "dt": 0.005,
        "horizon": 5,
        "voltage_limit": 10.0,
        "target_lvdt": 0.0,
        "position_tolerance": 0.02,
        "static_friction_penalty": 50.0,
        "internal_substeps": 30,
        "weights": MPCWeights(),
        "auto_fc_gain": 2.5,
        "auto_fc_floor": 0.0,
        "auto_fc_cap": None,
        "friction_blend_error_low": 0.2,
        "friction_blend_error_high": 0.5,
        "opt_iters": 10,
        "opt_step": 0.1,
        "opt_eps": 0.1,
        "pd_blend": 0.7,
        "pi_ki": 0.0,
        "pi_limit": 5.0,
        "pi_gate_saturation": True,
        "pi_gate_blocked": True,
        "pi_gate_error_band": True,
        "pi_leak_near_setpoint": True,
        "use_model_integrator": False,
    }
    kwargs.update(overrides)
    return kwargs


def build_default_tube_controller_kwargs(**overrides: object) -> Dict[str, object]:
    """Return ``TubeMPCController`` kwargs paired with the defaults above."""

    kwargs: Dict[str, object] = {
        "dt": 0.005,
        "horizon": 3,
        "voltage_limit": 10.0,
        "target_lvdt": 0.0,
        "candidate_count": 5,
        "position_tolerance": 0.02,
        "static_friction_penalty": 50.0,
        "internal_substeps": 30,
        "inductance_rel_uncertainty": 0.5,
        "tube_tolerance": 1e-6,
        "tube_max_iterations": 500,
        "lqr_state_weight": (2.0, 0.2, 5.0),
        "lqr_input_weight": 0.5,
        "integral_gain": 0.05,
        "integral_limit": 5.0,
        "weights": MPCWeights(),
    }
    kwargs.update(overrides)
    return kwargs


__all__ = [
    "MotorSimulation",
    "SimulationState",
    "SimulationHistory",
    "build_default_controller_kwargs",
    "build_default_continuous_controller_kwargs",
    "build_default_tube_controller_kwargs",
    "build_default_motor_kwargs",
]

