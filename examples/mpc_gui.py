"""Interactive GUI for exploring the LVDT-based MPC motor controller."""

from __future__ import annotations

import sys
from collections import deque
from dataclasses import dataclass
from typing import Dict

from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from motor_model.brushed_motor import BrushedMotorModel
from motor_model.mpc_controller import LVDTMPCController, MPCWeights


@dataclass
class DoubleParamConfig:
    """Configuration helper for numeric spin boxes."""

    label: str
    minimum: float
    maximum: float
    step: float
    decimals: int
    default: float
    suffix: str = ""

class MotorSimulation:
    """Lightweight time-domain simulation suitable for interactive use."""

    def __init__(self, motor_kwargs: Dict[str, float], controller_kwargs: Dict[str, float]) -> None:
        self._motor_kwargs = motor_kwargs
        self._controller_kwargs = controller_kwargs
        self.history_duration = 8.0
        self.max_points = 8000
        self.reset()

    def reset(self) -> None:
        self.motor = BrushedMotorModel(**self._motor_kwargs)

        controller_kwargs = dict(self._controller_kwargs)
        weights = controller_kwargs.pop("weights")
        if not isinstance(weights, MPCWeights):
            raise TypeError("weights must be an MPCWeights instance")
        self.controller = LVDTMPCController(self.motor, weights=weights, **controller_kwargs)

        self.plant_dt = self.controller.dt / max(1, controller_kwargs.get("internal_substeps", 1))
        self.measurement_steps = max(1, int(round(self.controller.dt / self.plant_dt)))
        self._steps_since_measurement = 0

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

        self.time_history: deque[float] = deque([0.0], maxlen=self.max_points)
        self.position_history: deque[float] = deque([self.position], maxlen=self.max_points)
        self.setpoint_history: deque[float] = deque([self.target_position()], maxlen=self.max_points)
        self.voltage_history: deque[float] = deque([self.voltage], maxlen=self.max_points)
        self.speed_history: deque[float] = deque([self.speed], maxlen=self.max_points)
        self.current_history: deque[float] = deque([self.current], maxlen=self.max_points)

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


class ControllerDemo(QtWidgets.QMainWindow):
    """Main application window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Motor MPC Visualiser")
        self.resize(1200, 720)

        self._setup_in_progress = True
        self._block_updates = False
        self.simulation: MotorSimulation | None = None

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        layout = QtWidgets.QHBoxLayout(central)

        self.figure = Figure(figsize=(6, 4))
        self.canvas = FigureCanvas(self.figure)
        self.axes = self.figure.add_subplot(111)
        self.axes.set_xlabel("Time [s]")
        self.axes.set_ylabel("Position [rad]")
        self.axes.grid(True)
        layout.addWidget(self.canvas, stretch=3)

        self.position_line, = self.axes.plot([], [], label="Rotor position", color="#1f77b4")
        self.setpoint_line, = self.axes.plot([], [], label="Setpoint", linestyle="--", color="#d62728")
        self.axes.legend(loc="upper right")

        self.canvas.mpl_connect("button_press_event", self._on_plot_clicked)

        controls_scroll = QtWidgets.QScrollArea()
        controls_scroll.setWidgetResizable(True)
        layout.addWidget(controls_scroll, stretch=2)

        controls_container = QtWidgets.QWidget()
        controls_scroll.setWidget(controls_container)

        controls_layout = QtWidgets.QVBoxLayout(controls_container)
        controls_layout.setSpacing(10)

        controls_layout.addWidget(self._create_target_section())
        controls_layout.addWidget(self._create_motor_section())
        controls_layout.addWidget(self._create_controller_section())
        controls_layout.addWidget(self._create_status_section())
        controls_layout.addStretch(1)

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(30)
        self.timer.timeout.connect(self._on_timer)

        self._setup_in_progress = False
        self.reset_simulation()
        self.timer.start()

    # ------------------------------------------------------------------
    # UI builders
    # ------------------------------------------------------------------
    def _create_target_section(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Target")
        layout = QtWidgets.QFormLayout(box)

        self.target_spin = QtWidgets.QDoubleSpinBox()
        self.target_spin.setDecimals(4)
        self.target_spin.setSingleStep(0.01)
        self.target_spin.setRange(-0.5, 0.5)
        self.target_spin.setValue(0.0)
        self.target_spin.valueChanged.connect(self._on_target_changed)
        layout.addRow("Setpoint [rad]", self.target_spin)

        note = QtWidgets.QLabel("Click the plot to set a new target position.")
        note.setWordWrap(True)
        layout.addRow(note)
        return box

    def _create_motor_section(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Motor parameters")
        form = QtWidgets.QFormLayout(box)

        self.motor_controls: Dict[str, QtWidgets.QDoubleSpinBox] = {}
        configs: Dict[str, DoubleParamConfig] = {
            "resistance": DoubleParamConfig("Resistance [Ω]", 1.0, 100.0, 0.1, 2, 28.0),
            "inductance": DoubleParamConfig("Inductance [H]", 1e-4, 0.2, 1e-4, 6, 16e-3),
            "kv": DoubleParamConfig("Speed constant [rad/s/V]", 0.1, 200.0, 0.1, 2, 7.0),
            "inertia": DoubleParamConfig("Inertia [kg·m²]", 1e-7, 1e-2, 1e-7, 7, 5e-5),
            "viscous_friction": DoubleParamConfig("Viscous friction [N·m·s/rad]", 0.0, 1e-2, 1e-6, 7, 2e-5),
            "coulomb_friction": DoubleParamConfig("Coulomb friction [N·m]", 0.0, 0.02, 1e-4, 4, 2.2e-3),
            "static_friction": DoubleParamConfig("Static friction [N·m]", 0.0, 0.02, 1e-4, 4, 2.5e-3),
            "stop_speed_threshold": DoubleParamConfig("Stop threshold [rad/s]", 0.0, 0.1, 1e-4, 6, 1e-4),
            "spring_constant": DoubleParamConfig("Spring constant [N·m/rad]", 0.0, 1e-1, 1e-4, 6, 1e-4),
            "spring_compression_ratio": DoubleParamConfig("Compression ratio", 0.0, 1.0, 0.01, 3, 0.4),
            "lvdt_full_scale": DoubleParamConfig("LVDT full scale [rad]", 0.01, 1.0, 0.01, 3, 0.1),
            "lvdt_noise_std": DoubleParamConfig("LVDT noise σ", 0.0, 0.1, 1e-4, 4, 5e-3),
        }

        for name, config in configs.items():
            spin = QtWidgets.QDoubleSpinBox()
            spin.setDecimals(config.decimals)
            spin.setRange(config.minimum, config.maximum)
            spin.setSingleStep(config.step)
            spin.setValue(config.default)
            if config.suffix:
                spin.setSuffix(config.suffix)
            spin.valueChanged.connect(self._on_parameters_changed)
            self.motor_controls[name] = spin
            form.addRow(config.label, spin)

        return box

    def _create_controller_section(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Controller parameters")
        layout = QtWidgets.QVBoxLayout(box)

        controller_form = QtWidgets.QFormLayout()
        layout.addLayout(controller_form)

        self.controller_controls: Dict[str, QtWidgets.QWidget] = {}

        dt_spin = QtWidgets.QDoubleSpinBox()
        dt_spin.setDecimals(5)
        dt_spin.setRange(1e-4, 0.1)
        dt_spin.setSingleStep(1e-4)
        dt_spin.setValue(0.005)
        dt_spin.valueChanged.connect(self._on_parameters_changed)
        controller_form.addRow("Controller dt [s]", dt_spin)
        self.controller_controls["dt"] = dt_spin

        horizon_spin = QtWidgets.QSpinBox()
        horizon_spin.setRange(1, 12)
        horizon_spin.setValue(4)
        horizon_spin.valueChanged.connect(self._on_parameters_changed)
        controller_form.addRow("Horizon", horizon_spin)
        self.controller_controls["horizon"] = horizon_spin

        voltage_spin = QtWidgets.QDoubleSpinBox()
        voltage_spin.setDecimals(2)
        voltage_spin.setRange(1.0, 60.0)
        voltage_spin.setSingleStep(0.5)
        voltage_spin.setValue(10.0)
        voltage_spin.valueChanged.connect(self._on_parameters_changed)
        controller_form.addRow("Voltage limit [V]", voltage_spin)
        self.controller_controls["voltage_limit"] = voltage_spin

        candidate_spin = QtWidgets.QSpinBox()
        candidate_spin.setRange(3, 15)
        candidate_spin.setSingleStep(2)
        candidate_spin.setValue(5)
        candidate_spin.valueChanged.connect(self._on_parameters_changed)
        controller_form.addRow("Candidate count", candidate_spin)
        self.controller_controls["candidate_count"] = candidate_spin

        tolerance_spin = QtWidgets.QDoubleSpinBox()
        tolerance_spin.setDecimals(3)
        tolerance_spin.setRange(0.0, 0.5)
        tolerance_spin.setSingleStep(0.005)
        tolerance_spin.setValue(0.02)
        tolerance_spin.valueChanged.connect(self._on_parameters_changed)
        controller_form.addRow("Position tolerance", tolerance_spin)
        self.controller_controls["position_tolerance"] = tolerance_spin

        penalty_spin = QtWidgets.QDoubleSpinBox()
        penalty_spin.setDecimals(2)
        penalty_spin.setRange(0.0, 1000.0)
        penalty_spin.setSingleStep(1.0)
        penalty_spin.setValue(50.0)
        penalty_spin.valueChanged.connect(self._on_parameters_changed)
        controller_form.addRow("Static friction penalty", penalty_spin)
        self.controller_controls["static_friction_penalty"] = penalty_spin

        self.auto_friction_check = QtWidgets.QCheckBox("Automatic friction compensation")
        self.auto_friction_check.setChecked(True)
        self.auto_friction_check.stateChanged.connect(self._on_parameters_changed)
        layout.addWidget(self.auto_friction_check)

        self.friction_spin = QtWidgets.QDoubleSpinBox()
        self.friction_spin.setDecimals(3)
        self.friction_spin.setRange(0.1, 30.0)
        self.friction_spin.setSingleStep(0.1)
        self.friction_spin.setValue(3.0)
        self.friction_spin.valueChanged.connect(self._on_parameters_changed)
        controller_form.addRow("Manual friction compensation [V]", self.friction_spin)

        substeps_spin = QtWidgets.QSpinBox()
        substeps_spin.setRange(1, 10)
        substeps_spin.setValue(1)
        substeps_spin.valueChanged.connect(self._on_parameters_changed)
        controller_form.addRow("Internal substeps", substeps_spin)
        self.controller_controls["internal_substeps"] = substeps_spin

        weights_box = QtWidgets.QGroupBox("MPC weights")
        weights_form = QtWidgets.QFormLayout(weights_box)
        layout.addWidget(weights_box)

        self.weight_controls: Dict[str, QtWidgets.QDoubleSpinBox] = {}
        default_weights = MPCWeights()
        weight_configs: Dict[str, DoubleParamConfig] = {
            "position": DoubleParamConfig("Position", 0.0, 2000.0, 1.0, 1, default_weights.position),
            "speed": DoubleParamConfig("Speed", 0.0, 100.0, 0.1, 1, default_weights.speed),
            "voltage": DoubleParamConfig("Voltage", 0.0, 10.0, 0.01, 2, default_weights.voltage),
            "delta_voltage": DoubleParamConfig("ΔVoltage", 0.0, 10.0, 0.01, 2, default_weights.delta_voltage),
            "terminal_position": DoubleParamConfig(
                "Terminal position", 0.0, 5000.0, 1.0, 1, default_weights.terminal_position
            ),
        }

        for name, config in weight_configs.items():
            spin = QtWidgets.QDoubleSpinBox()
            spin.setDecimals(config.decimals)
            spin.setRange(config.minimum, config.maximum)
            spin.setSingleStep(config.step)
            spin.setValue(config.default)
            spin.valueChanged.connect(self._on_parameters_changed)
            weights_form.addRow(config.label, spin)
            self.weight_controls[name] = spin

        return box

    def _create_status_section(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Live status")
        layout = QtWidgets.QFormLayout(box)

        self.time_label = QtWidgets.QLabel("0.000 s")
        self.position_label = QtWidgets.QLabel("0.000 rad")
        self.speed_label = QtWidgets.QLabel("0.000 rad/s")
        self.current_label = QtWidgets.QLabel("0.000 A")
        self.voltage_label = QtWidgets.QLabel("0.000 V")

        layout.addRow("Time", self.time_label)
        layout.addRow("Position", self.position_label)
        layout.addRow("Speed", self.speed_label)
        layout.addRow("Current", self.current_label)
        layout.addRow("Voltage", self.voltage_label)

        return box

    # ------------------------------------------------------------------
    # Simulation handling
    # ------------------------------------------------------------------
    def reset_simulation(self) -> None:
        if self._setup_in_progress:
            return

        motor_kwargs = {name: control.value() for name, control in self.motor_controls.items()}

        target_position = self.target_spin.value()
        lvdt_scale = motor_kwargs["lvdt_full_scale"]
        target_lvdt = 0.0 if lvdt_scale <= 0 else max(-1.0, min(1.0, target_position / lvdt_scale))

        controller_kwargs: Dict[str, float] = {
            "dt": float(self.controller_controls["dt"].value()),
            "horizon": int(self.controller_controls["horizon"].value()),
            "voltage_limit": float(self.controller_controls["voltage_limit"].value()),
            "target_lvdt": float(target_lvdt),
            "candidate_count": int(self.controller_controls["candidate_count"].value()),
            "position_tolerance": float(self.controller_controls["position_tolerance"].value()),
            "static_friction_penalty": float(self.controller_controls["static_friction_penalty"].value()),
            "internal_substeps": int(self.controller_controls["internal_substeps"].value()),
            "weights": MPCWeights(
                position=self.weight_controls["position"].value(),
                speed=self.weight_controls["speed"].value(),
                voltage=self.weight_controls["voltage"].value(),
                delta_voltage=self.weight_controls["delta_voltage"].value(),
                terminal_position=self.weight_controls["terminal_position"].value(),
            ),
        }

        if self.auto_friction_check.isChecked():
            controller_kwargs["friction_compensation"] = None
        else:
            controller_kwargs["friction_compensation"] = float(self.friction_spin.value())

        self.simulation = MotorSimulation(motor_kwargs, controller_kwargs)
        self.simulation.set_target_position(target_position)

        self._block_updates = True
        try:
            self.target_spin.setRange(-lvdt_scale, lvdt_scale)
            self.target_spin.setValue(self.simulation.target_position())
        finally:
            self._block_updates = False

        self._update_plot(force_limits=True)
        self._update_status_labels()

    def _on_parameters_changed(self) -> None:
        if self._setup_in_progress or self._block_updates:
            return
        self.reset_simulation()

    def _on_target_changed(self, value: float) -> None:
        if self._setup_in_progress or self._block_updates:
            return
        if not self.simulation:
            return
        self.simulation.set_target_position(value)

    def _on_plot_clicked(self, event) -> None:  # type: ignore[override]
        if event.inaxes != self.axes or not self.simulation:
            return
        self._block_updates = True
        try:
            self.target_spin.setValue(float(event.ydata))
        finally:
            self._block_updates = False
        self._on_target_changed(self.target_spin.value())

    def _on_timer(self) -> None:
        if not self.simulation:
            return
        dt_ms = max(1.0, self.simulation.plant_dt_ms())
        steps = max(1, int(self.timer.interval() / dt_ms))
        self.simulation.step(steps)
        self._update_plot()
        self._update_status_labels()

    def _update_plot(self, *, force_limits: bool = False) -> None:
        if not self.simulation:
            return
        times = list(self.simulation.time_history)
        if not times:
            return
        positions = list(self.simulation.position_history)
        setpoints = list(self.simulation.setpoint_history)

        self.position_line.set_data(times, positions)
        self.setpoint_line.set_data(times, setpoints)

        t_max = times[-1]
        t_min = max(0.0, t_max - self.simulation.history_duration)
        if force_limits:
            self.axes.set_xlim(t_min, t_max + 1e-9)
        else:
            current_limits = self.axes.get_xlim()
            if abs(current_limits[1] - current_limits[0]) < 1e-9:
                self.axes.set_xlim(t_min, t_max + 1e-9)
            else:
                self.axes.set_xlim(t_min, t_max + 1e-9)

        value_min = min(min(positions), min(setpoints))
        value_max = max(max(positions), max(setpoints))
        span = value_max - value_min
        margin = 0.05 * span if span > 1e-9 else 0.05
        self.axes.set_ylim(value_min - margin, value_max + margin)

        self.canvas.draw_idle()

    def _update_status_labels(self) -> None:
        if not self.simulation:
            return
        self.time_label.setText(f"{self.simulation.time:0.3f} s")
        self.position_label.setText(f"{self.simulation.position:0.4f} rad")
        self.speed_label.setText(f"{self.simulation.speed:0.4f} rad/s")
        self.current_label.setText(f"{self.simulation.current:0.3f} A")
        self.voltage_label.setText(f"{self.simulation.voltage:0.3f} V")


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    window = ControllerDemo()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
