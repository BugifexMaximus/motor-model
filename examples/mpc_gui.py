"""Interactive GUI for exploring the motor MPC controllers."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Dict, cast

from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from motor_model.brushed_motor import (
    rad_per_sec_per_volt_to_rpm_per_volt,
    rpm_per_volt_to_rad_per_sec_per_volt,
)
from motor_model.mpc_controller import MPCWeights
from motor_model.mpc_simulation import (
    MotorSimulation,
    build_default_controller_kwargs,
    build_default_tube_controller_kwargs,
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
    def _motor_param_configs(self, defaults: Dict[str, float]) -> Dict[str, DoubleParamConfig]:
        return {
            "resistance": DoubleParamConfig("Resistance [Ω]", 1.0, 100.0, 0.1, 2, defaults["resistance"]),
            "inductance": DoubleParamConfig("Inductance [H]", 1e-4, 0.2, 1e-4, 6, defaults["inductance"]),
            "kv": DoubleParamConfig(
                "Speed constant [RPM/V]",
                0.1,
                600.0,
                0.1,
                2,
                rad_per_sec_per_volt_to_rpm_per_volt(defaults["kv"]),
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
            "lvdt_noise_std": DoubleParamConfig(
                "LVDT noise σ", 0.0, 0.1, 1e-4, 4, defaults["lvdt_noise_std"]
            ),
        }

    def _add_param_control(
        self,
        form: QtWidgets.QFormLayout,
        name: str,
        config: DoubleParamConfig,
        storage: Dict[str, QtWidgets.QDoubleSpinBox],
    ) -> None:
        spin = QtWidgets.QDoubleSpinBox()
        spin.setDecimals(config.decimals)
        spin.setRange(config.minimum, config.maximum)
        spin.setSingleStep(config.step)
        spin.setValue(config.default)
        if config.suffix:
            spin.setSuffix(config.suffix)
        spin.valueChanged.connect(self._on_parameters_changed)
        storage[name] = spin
        form.addRow(config.label, spin)

    def _create_double_spin(
        self,
        minimum: float,
        maximum: float,
        step: float,
        decimals: int,
        value: float,
        *,
        suffix: str = "",
    ) -> QtWidgets.QDoubleSpinBox:
        spin = QtWidgets.QDoubleSpinBox()
        spin.setDecimals(decimals)
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setValue(value)
        if suffix:
            spin.setSuffix(suffix)
        spin.valueChanged.connect(self._on_parameters_changed)
        return spin

    def _create_int_spin(
        self,
        minimum: int,
        maximum: int,
        step: int,
        value: int,
    ) -> QtWidgets.QSpinBox:
        spin = QtWidgets.QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setValue(value)
        spin.valueChanged.connect(self._on_parameters_changed)
        return spin

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
        box = QtWidgets.QGroupBox("Physical motor model")
        form = QtWidgets.QFormLayout()
        box.setLayout(form)

        description = QtWidgets.QLabel(
            "These parameters describe the simulated physical motor and measurement system."
        )
        description.setWordWrap(True)
        form.addRow(description)

        self.motor_controls: Dict[str, QtWidgets.QDoubleSpinBox] = {}
        defaults = build_default_motor_kwargs(lvdt_noise_std=5e-3)
        configs = self._motor_param_configs(defaults)

        for name, config in configs.items():
            self._add_param_control(form, name, config, self.motor_controls)

        return box

    def _create_controller_section(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("MPC configuration")
        layout = QtWidgets.QVBoxLayout(box)

        algorithm_box = QtWidgets.QGroupBox("Algorithm tuning")
        algorithm_layout = QtWidgets.QVBoxLayout(algorithm_box)
        layout.addWidget(algorithm_box)

        self.controller_type_combo = QtWidgets.QComboBox()
        self.controller_type_combo.addItem("LVDT MPC", "lvdtnom")
        self.controller_type_combo.addItem("Tube MPC", "tube")
        self.controller_type_combo.currentIndexChanged.connect(
            self._on_controller_type_changed
        )
        algorithm_layout.addWidget(self.controller_type_combo)

        self.controller_stack = QtWidgets.QStackedWidget()
        algorithm_layout.addWidget(self.controller_stack)

        self.controller_controls_by_type: Dict[str, Dict[str, QtWidgets.QWidget]] = {}
        self._controller_stack_indices: Dict[str, int] = {}

        lvd_defaults = build_default_controller_kwargs()
        lvd_widget, lvd_controls = self._build_lvd_controller_controls(lvd_defaults)
        self.controller_controls_by_type["lvdtnom"] = lvd_controls
        self._controller_stack_indices["lvdtnom"] = self.controller_stack.addWidget(
            lvd_widget
        )

        tube_defaults = build_default_tube_controller_kwargs()
        tube_widget, tube_controls = self._build_tube_controller_controls(tube_defaults)
        self.controller_controls_by_type["tube"] = tube_controls
        self._controller_stack_indices["tube"] = self.controller_stack.addWidget(
            tube_widget
        )

        self.controller_stack.setCurrentIndex(
            self._controller_stack_indices[self._current_controller_type()]
        )

        self.auto_friction_check = QtWidgets.QCheckBox("Automatic friction compensation")
        self.auto_friction_check.setChecked(True)
        self.auto_friction_check.stateChanged.connect(self._on_parameters_changed)
        layout.addWidget(self.auto_friction_check)

        manual_box = QtWidgets.QGroupBox("Manual overrides")
        manual_form = QtWidgets.QFormLayout(manual_box)
        layout.addWidget(manual_box)

        self.friction_spin = QtWidgets.QDoubleSpinBox()
        self.friction_spin.setDecimals(3)
        self.friction_spin.setRange(0.1, 30.0)
        self.friction_spin.setSingleStep(0.1)
        self.friction_spin.setValue(3.0)
        self.friction_spin.valueChanged.connect(self._on_parameters_changed)
        manual_form.addRow("Manual friction compensation [V]", self.friction_spin)

        model_box = QtWidgets.QGroupBox("MPC internal model")
        model_layout = QtWidgets.QVBoxLayout(model_box)
        model_description = QtWidgets.QLabel(
            "Adjust the physical parameters used inside the controller's predictive model."
        )
        model_description.setWordWrap(True)
        model_layout.addWidget(model_description)

        model_form = QtWidgets.QFormLayout()
        model_layout.addLayout(model_form)
        layout.addWidget(model_box)

        self.controller_model_controls: Dict[str, QtWidgets.QDoubleSpinBox] = {}
        controller_motor_defaults = build_default_motor_kwargs(lvdt_noise_std=5e-3)
        controller_configs = self._motor_param_configs(controller_motor_defaults)
        for name, config in controller_configs.items():
            self._add_param_control(model_form, name, config, self.controller_model_controls)

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

    def _build_lvd_controller_controls(
        self, defaults: Dict[str, float]
    ) -> tuple[QtWidgets.QWidget, Dict[str, QtWidgets.QWidget]]:
        widget = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(widget)
        controls: Dict[str, QtWidgets.QWidget] = {}

        dt_spin = self._create_double_spin(1e-4, 0.1, 1e-4, 5, defaults["dt"])
        form.addRow("Controller dt [s]", dt_spin)
        controls["dt"] = dt_spin

        horizon_spin = self._create_int_spin(1, 12, 1, int(defaults["horizon"]))
        form.addRow("Horizon", horizon_spin)
        controls["horizon"] = horizon_spin

        voltage_spin = self._create_double_spin(1.0, 60.0, 0.5, 2, defaults["voltage_limit"])
        form.addRow("Voltage limit [V]", voltage_spin)
        controls["voltage_limit"] = voltage_spin

        candidate_spin = self._create_int_spin(3, 15, 2, int(defaults["candidate_count"]))
        form.addRow("Candidate count", candidate_spin)
        controls["candidate_count"] = candidate_spin

        tolerance_spin = self._create_double_spin(0.0, 0.5, 0.005, 3, defaults["position_tolerance"])
        form.addRow("Position tolerance", tolerance_spin)
        controls["position_tolerance"] = tolerance_spin

        penalty_spin = self._create_double_spin(0.0, 1000.0, 1.0, 2, defaults["static_friction_penalty"])
        form.addRow("Static friction penalty", penalty_spin)
        controls["static_friction_penalty"] = penalty_spin

        auto_gain_spin = self._create_double_spin(0.1, 5.0, 0.05, 2, defaults.get("auto_fc_gain", 1.1))
        form.addRow("Auto friction gain", auto_gain_spin)
        controls["auto_fc_gain"] = auto_gain_spin

        auto_floor_spin = self._create_double_spin(
            0.0,
            60.0,
            0.05,
            2,
            defaults.get("auto_fc_floor", 0.0),
        )
        form.addRow("Auto friction floor [V]", auto_floor_spin)
        controls["auto_fc_floor"] = auto_floor_spin

        auto_cap_value = defaults.get("auto_fc_cap")
        auto_cap_spin = self._create_double_spin(0.1, 60.0, 0.05, 2, defaults["voltage_limit"])
        if isinstance(auto_cap_value, (int, float)):
            auto_cap_spin.setValue(float(auto_cap_value))
        auto_cap_spin.setEnabled(isinstance(auto_cap_value, (int, float)))

        auto_cap_checkbox = QtWidgets.QCheckBox("Enable cap")
        auto_cap_checkbox.setChecked(isinstance(auto_cap_value, (int, float)))

        def _on_auto_cap_state_changed(state: int) -> None:
            enabled = state == QtCore.Qt.Checked
            auto_cap_spin.setEnabled(enabled)
            self._on_parameters_changed()

        auto_cap_checkbox.stateChanged.connect(_on_auto_cap_state_changed)

        auto_cap_container = QtWidgets.QWidget()
        auto_cap_layout = QtWidgets.QHBoxLayout(auto_cap_container)
        auto_cap_layout.setContentsMargins(0, 0, 0, 0)
        auto_cap_layout.setSpacing(6)
        auto_cap_layout.addWidget(auto_cap_checkbox)
        auto_cap_layout.addWidget(auto_cap_spin)
        auto_cap_layout.addStretch(1)

        form.addRow("Auto friction cap [V]", auto_cap_container)
        controls["auto_fc_cap_enabled"] = auto_cap_checkbox
        controls["auto_fc_cap"] = auto_cap_spin

        blend_low_spin = self._create_double_spin(
            0.0,
            1.0,
            0.005,
            3,
            defaults.get("friction_blend_error_low", 0.05),
        )
        form.addRow("Blend error low", blend_low_spin)
        controls["friction_blend_error_low"] = blend_low_spin

        blend_high_spin = self._create_double_spin(
            0.0,
            1.0,
            0.005,
            3,
            defaults.get("friction_blend_error_high", 0.2),
        )
        form.addRow("Blend error high", blend_high_spin)
        controls["friction_blend_error_high"] = blend_high_spin

        substeps_spin = self._create_int_spin(
            1, 50, 1, int(defaults["internal_substeps"])
        )
        form.addRow("Internal substeps", substeps_spin)
        controls["internal_substeps"] = substeps_spin

        return widget, controls

    def _build_tube_controller_controls(
        self, defaults: Dict[str, float]
    ) -> tuple[QtWidgets.QWidget, Dict[str, QtWidgets.QWidget]]:
        widget = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(widget)
        controls: Dict[str, QtWidgets.QWidget] = {}

        dt_spin = self._create_double_spin(1e-4, 0.1, 1e-4, 5, defaults["dt"])
        form.addRow("Controller dt [s]", dt_spin)
        controls["dt"] = dt_spin

        horizon_spin = self._create_int_spin(1, 12, 1, int(defaults["horizon"]))
        form.addRow("Horizon", horizon_spin)
        controls["horizon"] = horizon_spin

        voltage_spin = self._create_double_spin(1.0, 60.0, 0.5, 2, defaults["voltage_limit"])
        form.addRow("Voltage limit [V]", voltage_spin)
        controls["voltage_limit"] = voltage_spin

        candidate_spin = self._create_int_spin(3, 15, 2, int(defaults["candidate_count"]))
        form.addRow("Candidate count", candidate_spin)
        controls["candidate_count"] = candidate_spin

        tolerance_spin = self._create_double_spin(0.0, 0.5, 0.005, 3, defaults["position_tolerance"])
        form.addRow("Position tolerance", tolerance_spin)
        controls["position_tolerance"] = tolerance_spin

        penalty_spin = self._create_double_spin(0.0, 1000.0, 1.0, 2, defaults["static_friction_penalty"])
        form.addRow("Static friction penalty", penalty_spin)
        controls["static_friction_penalty"] = penalty_spin

        substeps_spin = self._create_int_spin(
            1, 50, 1, int(defaults["internal_substeps"])
        )
        form.addRow("Internal substeps", substeps_spin)
        controls["internal_substeps"] = substeps_spin

        inductance_spin = self._create_double_spin(
            0.0,
            2.0,
            0.05,
            2,
            defaults["inductance_rel_uncertainty"],
        )
        form.addRow("Inductance relative uncertainty", inductance_spin)
        controls["inductance_rel_uncertainty"] = inductance_spin

        tube_tol_spin = self._create_double_spin(
            1e-8,
            1e-2,
            1e-6,
            8,
            defaults["tube_tolerance"],
        )
        form.addRow("Tube tolerance", tube_tol_spin)
        controls["tube_tolerance"] = tube_tol_spin

        tube_iter_spin = self._create_int_spin(10, 5000, 10, int(defaults["tube_max_iterations"]))
        form.addRow("Tube max iterations", tube_iter_spin)
        controls["tube_max_iterations"] = tube_iter_spin

        integral_gain_spin = self._create_double_spin(
            0.0,
            2.0,
            0.01,
            2,
            defaults["integral_gain"],
        )
        form.addRow("Integral gain", integral_gain_spin)
        controls["integral_gain"] = integral_gain_spin

        integral_limit_spin = self._create_double_spin(
            0.1,
            50.0,
            0.1,
            1,
            defaults["integral_limit"],
        )
        form.addRow("Integral limit", integral_limit_spin)
        controls["integral_limit"] = integral_limit_spin

        lqr_current_spin = self._create_double_spin(0.0, 50.0, 0.1, 2, defaults["lqr_state_weight"][0])
        form.addRow("LQR weight (current)", lqr_current_spin)
        controls["lqr_state_weight_current"] = lqr_current_spin

        lqr_speed_spin = self._create_double_spin(0.0, 50.0, 0.1, 2, defaults["lqr_state_weight"][1])
        form.addRow("LQR weight (speed)", lqr_speed_spin)
        controls["lqr_state_weight_speed"] = lqr_speed_spin

        lqr_position_spin = self._create_double_spin(
            0.0,
            50.0,
            0.1,
            2,
            defaults["lqr_state_weight"][2],
        )
        form.addRow("LQR weight (position)", lqr_position_spin)
        controls["lqr_state_weight_position"] = lqr_position_spin

        lqr_input_spin = self._create_double_spin(0.01, 10.0, 0.01, 2, defaults["lqr_input_weight"])
        form.addRow("LQR input weight", lqr_input_spin)
        controls["lqr_input_weight"] = lqr_input_spin

        return widget, controls

    def _current_controller_type(self) -> str:
        data = self.controller_type_combo.currentData() if hasattr(self, "controller_type_combo") else None
        if not data:
            return "lvdtnom"
        return str(data)

    def _on_controller_type_changed(self, index: int) -> None:  # noqa: ARG002
        if self._setup_in_progress:
            return
        controller_type = self._current_controller_type()
        self.controller_stack.setCurrentIndex(
            self._controller_stack_indices[controller_type]
        )
        self.reset_simulation()

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
    def _controller_kwargs_for_type(self, controller_type: str) -> Dict[str, object]:
        controls = self.controller_controls_by_type[controller_type]

        kwargs: Dict[str, object] = {
            "dt": float(cast(QtWidgets.QDoubleSpinBox, controls["dt"]).value()),
            "horizon": int(cast(QtWidgets.QSpinBox, controls["horizon"]).value()),
            "voltage_limit": float(
                cast(QtWidgets.QDoubleSpinBox, controls["voltage_limit"]).value()
            ),
            "candidate_count": int(
                cast(QtWidgets.QSpinBox, controls["candidate_count"]).value()
            ),
            "position_tolerance": float(
                cast(QtWidgets.QDoubleSpinBox, controls["position_tolerance"]).value()
            ),
            "static_friction_penalty": float(
                cast(QtWidgets.QDoubleSpinBox, controls["static_friction_penalty"]).value()
            ),
            "internal_substeps": int(
                cast(QtWidgets.QSpinBox, controls["internal_substeps"]).value()
            ),
        }

        if "auto_fc_gain" in controls:
            kwargs.update(
                {
                    "auto_fc_gain": float(
                        cast(QtWidgets.QDoubleSpinBox, controls["auto_fc_gain"]).value()
                    ),
                    "auto_fc_floor": float(
                        cast(QtWidgets.QDoubleSpinBox, controls["auto_fc_floor"]).value()
                    ),
                    "friction_blend_error_low": float(
                        cast(
                            QtWidgets.QDoubleSpinBox,
                            controls["friction_blend_error_low"],
                        ).value()
                    ),
                    "friction_blend_error_high": float(
                        cast(
                            QtWidgets.QDoubleSpinBox,
                            controls["friction_blend_error_high"],
                        ).value()
                    ),
                }
            )

            cap_enabled = cast(QtWidgets.QCheckBox, controls["auto_fc_cap_enabled"]).isChecked()
            if cap_enabled:
                kwargs["auto_fc_cap"] = float(
                    cast(QtWidgets.QDoubleSpinBox, controls["auto_fc_cap"]).value()
                )
            else:
                kwargs["auto_fc_cap"] = None

        if controller_type == "tube":
            kwargs.update(
                {
                    "inductance_rel_uncertainty": float(
                        cast(
                            QtWidgets.QDoubleSpinBox,
                            controls["inductance_rel_uncertainty"],
                        ).value()
                    ),
                    "tube_tolerance": float(
                        cast(QtWidgets.QDoubleSpinBox, controls["tube_tolerance"]).value()
                    ),
                    "tube_max_iterations": int(
                        cast(QtWidgets.QSpinBox, controls["tube_max_iterations"]).value()
                    ),
                    "lqr_state_weight": (
                        float(
                            cast(
                                QtWidgets.QDoubleSpinBox,
                                controls["lqr_state_weight_current"],
                            ).value()
                        ),
                        float(
                            cast(
                                QtWidgets.QDoubleSpinBox,
                                controls["lqr_state_weight_speed"],
                            ).value()
                        ),
                        float(
                            cast(
                                QtWidgets.QDoubleSpinBox,
                                controls["lqr_state_weight_position"],
                            ).value()
                        ),
                    ),
                    "lqr_input_weight": float(
                        cast(QtWidgets.QDoubleSpinBox, controls["lqr_input_weight"]).value()
                    ),
                    "integral_gain": float(
                        cast(QtWidgets.QDoubleSpinBox, controls["integral_gain"]).value()
                    ),
                    "integral_limit": float(
                        cast(QtWidgets.QDoubleSpinBox, controls["integral_limit"]).value()
                    ),
                }
            )

        return kwargs

    def reset_simulation(self) -> None:
        if self._setup_in_progress:
            return

        motor_kwargs = {}
        for name, control in self.motor_controls.items():
            value = float(control.value())
            if name == "kv":
                value = rpm_per_volt_to_rad_per_sec_per_volt(value)
            motor_kwargs[name] = value

        controller_motor_kwargs = {}
        for name, control in self.controller_model_controls.items():
            value = float(control.value())
            if name == "kv":
                value = rpm_per_volt_to_rad_per_sec_per_volt(value)
            controller_motor_kwargs[name] = value

        controller_type = self._current_controller_type()
        controller_kwargs = self._controller_kwargs_for_type(controller_type)

        target_position_deg = self.target_spin.value()
        target_position = math.radians(target_position_deg)
        lvdt_scale = motor_kwargs["lvdt_full_scale"]
        target_lvdt = 0.0 if lvdt_scale <= 0 else max(-1.0, min(1.0, target_position / lvdt_scale))

        controller_kwargs["target_lvdt"] = float(target_lvdt)
        controller_kwargs["weights"] = MPCWeights(
            position=self.weight_controls["position"].value(),
            speed=self.weight_controls["speed"].value(),
            voltage=self.weight_controls["voltage"].value(),
            delta_voltage=self.weight_controls["delta_voltage"].value(),
            terminal_position=self.weight_controls["terminal_position"].value(),
        )

        if self.auto_friction_check.isChecked():
            controller_kwargs["friction_compensation"] = None
        else:
            controller_kwargs["friction_compensation"] = float(self.friction_spin.value())

        self.simulation = MotorSimulation(
            motor_kwargs,
            controller_kwargs,
            controller_motor_kwargs=controller_motor_kwargs,
            controller_type=controller_type,
        )
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
