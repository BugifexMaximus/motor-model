"""Interactive GUI for exploring the LVDT-based MPC motor controller."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Dict

from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from motor_model.mpc_controller import MPCWeights
from motor_model.mpc_simulation import (
    MotorSimulation,
    build_default_controller_kwargs,
    build_default_motor_kwargs,
)


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
        self.axes.set_ylabel("Position [deg]")
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
        self.target_spin.setDecimals(2)
        self.target_spin.setSingleStep(0.1)
        self.target_spin.setRange(-30.0, 30.0)
        self.target_spin.setSuffix("°")
        self.target_spin.setValue(0.0)
        self.target_spin.valueChanged.connect(self._on_target_changed)
        layout.addRow("Setpoint [deg]", self.target_spin)

        note = QtWidgets.QLabel("Click the plot to set a new target position.")
        note.setWordWrap(True)
        layout.addRow(note)
        return box

    def _create_motor_section(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Motor parameters")
        form = QtWidgets.QFormLayout(box)

        self.motor_controls: Dict[str, QtWidgets.QDoubleSpinBox] = {}
        defaults = build_default_motor_kwargs(lvdt_noise_std=5e-3)
        configs: Dict[str, DoubleParamConfig] = {
            "resistance": DoubleParamConfig("Resistance [Ω]", 1.0, 100.0, 0.1, 2, defaults["resistance"]),
            "inductance": DoubleParamConfig("Inductance [H]", 1e-4, 0.2, 1e-4, 6, defaults["inductance"]),
            "kv": DoubleParamConfig(
                "Speed constant [rad/s/V]", 0.1, 200.0, 0.1, 2, defaults["kv"]
            ),
            "inertia": DoubleParamConfig("Inertia [kg·m²]", 1e-7, 1e-2, 1e-7, 7, defaults["inertia"]),
            "viscous_friction": DoubleParamConfig(
                "Viscous friction [N·m·s/rad]", 0.0, 1e-2, 1e-6, 7, defaults["viscous_friction"]
            ),
            "coulomb_friction": DoubleParamConfig(
                "Coulomb friction [N·m]", 0.0, 0.02, 1e-4, 4, defaults["coulomb_friction"]
            ),
            "static_friction": DoubleParamConfig(
                "Static friction [N·m]", 0.0, 0.02, 1e-4, 4, defaults["static_friction"]
            ),
            "stop_speed_threshold": DoubleParamConfig(
                "Stop threshold [rad/s]", 0.0, 0.1, 1e-4, 6, defaults["stop_speed_threshold"]
            ),
            "spring_constant": DoubleParamConfig(
                "Spring constant [N·m/rad]", 0.0, 1e-1, 1e-4, 6, defaults["spring_constant"]
            ),
            "spring_compression_ratio": DoubleParamConfig(
                "Compression ratio", 0.0, 1.0, 0.01, 3, defaults["spring_compression_ratio"]
            ),
            "lvdt_full_scale": DoubleParamConfig(
                "LVDT full scale [rad]", 0.01, 1.0, 0.01, 3, defaults["lvdt_full_scale"]
            ),
            "lvdt_noise_std": DoubleParamConfig("LVDT noise σ", 0.0, 0.1, 1e-4, 4, defaults["lvdt_noise_std"]),
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

        defaults = build_default_controller_kwargs()

        dt_spin = QtWidgets.QDoubleSpinBox()
        dt_spin.setDecimals(5)
        dt_spin.setRange(1e-4, 0.1)
        dt_spin.setSingleStep(1e-4)
        dt_spin.setValue(defaults["dt"])
        dt_spin.valueChanged.connect(self._on_parameters_changed)
        controller_form.addRow("Controller dt [s]", dt_spin)
        self.controller_controls["dt"] = dt_spin

        horizon_spin = QtWidgets.QSpinBox()
        horizon_spin.setRange(1, 12)
        horizon_spin.setValue(int(defaults["horizon"]))
        horizon_spin.valueChanged.connect(self._on_parameters_changed)
        controller_form.addRow("Horizon", horizon_spin)
        self.controller_controls["horizon"] = horizon_spin

        voltage_spin = QtWidgets.QDoubleSpinBox()
        voltage_spin.setDecimals(2)
        voltage_spin.setRange(1.0, 60.0)
        voltage_spin.setSingleStep(0.5)
        voltage_spin.setValue(defaults["voltage_limit"])
        voltage_spin.valueChanged.connect(self._on_parameters_changed)
        controller_form.addRow("Voltage limit [V]", voltage_spin)
        self.controller_controls["voltage_limit"] = voltage_spin

        candidate_spin = QtWidgets.QSpinBox()
        candidate_spin.setRange(3, 15)
        candidate_spin.setSingleStep(2)
        candidate_spin.setValue(int(defaults["candidate_count"]))
        candidate_spin.valueChanged.connect(self._on_parameters_changed)
        controller_form.addRow("Candidate count", candidate_spin)
        self.controller_controls["candidate_count"] = candidate_spin

        tolerance_spin = QtWidgets.QDoubleSpinBox()
        tolerance_spin.setDecimals(3)
        tolerance_spin.setRange(0.0, 0.5)
        tolerance_spin.setSingleStep(0.005)
        tolerance_spin.setValue(defaults["position_tolerance"])
        tolerance_spin.valueChanged.connect(self._on_parameters_changed)
        controller_form.addRow("Position tolerance", tolerance_spin)
        self.controller_controls["position_tolerance"] = tolerance_spin

        penalty_spin = QtWidgets.QDoubleSpinBox()
        penalty_spin.setDecimals(2)
        penalty_spin.setRange(0.0, 1000.0)
        penalty_spin.setSingleStep(1.0)
        penalty_spin.setValue(defaults["static_friction_penalty"])
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
        substeps_spin.setValue(int(defaults["internal_substeps"]))
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
        self.position_label = QtWidgets.QLabel("0.00 deg")
        self.speed_label = QtWidgets.QLabel("0.00 deg/s")
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

        target_position_deg = self.target_spin.value()
        target_position = math.radians(target_position_deg)
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
            lvdt_scale_deg = math.degrees(lvdt_scale)
            max_deg = min(30.0, lvdt_scale_deg)
            self.target_spin.setRange(-max_deg, max_deg)
            self.target_spin.setValue(math.degrees(self.simulation.target_position()))
        finally:
            self._block_updates = False

        self._update_plot()
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
        self.simulation.set_target_position(math.radians(value))

    def _on_plot_clicked(self, event) -> None:  # type: ignore[override]
        if event.inaxes != self.axes or not self.simulation:
            return
        self._block_updates = True
        try:
            if event.ydata is None:
                return
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

    def _update_plot(self) -> None:
        if not self.simulation:
            return
        times = list(self.simulation.time_history)
        if not times:
            return
        positions_deg = [math.degrees(value) for value in self.simulation.position_history]
        setpoints_deg = [math.degrees(value) for value in self.simulation.setpoint_history]

        self.position_line.set_data(times, positions_deg)
        self.setpoint_line.set_data(times, setpoints_deg)

        t_max = max(10.0, times[-1])
        t_min = t_max - self.simulation.history_duration
        self.axes.set_xlim(t_min, t_max)
        self.axes.set_ylim(-30.0, 30.0)

        self.canvas.draw_idle()

    def _update_status_labels(self) -> None:
        if not self.simulation:
            return
        self.time_label.setText(f"{self.simulation.time:0.3f} s")
        self.position_label.setText(f"{math.degrees(self.simulation.position):0.2f} deg")
        self.speed_label.setText(f"{math.degrees(self.simulation.speed):0.2f} deg/s")
        self.current_label.setText(f"{self.simulation.current:0.3f} A")
        self.voltage_label.setText(f"{self.simulation.voltage:0.3f} V")


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    window = ControllerDemo()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
