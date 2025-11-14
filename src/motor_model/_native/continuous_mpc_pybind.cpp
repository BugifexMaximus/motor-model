#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <memory>

#include "continuous_mpc_core.h"

namespace py = pybind11;

void RegisterContMPCControllerManager(py::module_& m);

namespace {

constexpr double kTwoPi = 2.0 * 3.14159265358979323846;
constexpr double kDefaultKv = 7.0 * kTwoPi / 60.0;

motor_model::MotorParams MotorParamsFromPython(const py::object& motor_obj) {
    if (py::isinstance<motor_model::MotorParams>(motor_obj)) {
        return motor_obj.cast<motor_model::MotorParams>();
    }

    auto attr = [&motor_obj](const char* name) { return motor_obj.attr(name); };

    motor_model::MotorParams params;
    params.resistance = attr("resistance").cast<double>();
    params.inductance = attr("inductance").cast<double>();
    params.kv = attr("kv").cast<double>();
    params.inertia = attr("inertia").cast<double>();
    params.viscous_friction = attr("viscous_friction").cast<double>();
    params.coulomb_friction = attr("coulomb_friction").cast<double>();
    params.static_friction = attr("static_friction").cast<double>();
    params.stop_speed_threshold = attr("stop_speed_threshold").cast<double>();
    params.spring_constant = attr("spring_constant").cast<double>();
    params.spring_compression_ratio = attr("spring_compression_ratio").cast<double>();
    if (py::hasattr(motor_obj, "spring_zero_position")) {
        params.spring_zero_position = attr("spring_zero_position").cast<double>();
    }
    params.lvdt_full_scale = attr("lvdt_full_scale").cast<double>();
    params.integration_substeps = attr("integration_substeps").cast<int>();

    if (py::hasattr(motor_obj, "_ke")) {
        params.ke = attr("_ke").cast<double>();
    } else if (py::hasattr(motor_obj, "ke")) {
        params.ke = attr("ke").cast<double>();
    } else if (params.kv > 0.0) {
        params.ke = 1.0 / params.kv;
    }

    if (py::hasattr(motor_obj, "_kt")) {
        params.kt = attr("_kt").cast<double>();
    } else if (py::hasattr(motor_obj, "kt")) {
        params.kt = attr("kt").cast<double>();
    } else if (params.ke != 0.0) {
        params.kt = params.ke;
    }

    return params;
}

motor_model::MPCWeightsValues WeightsFromPython(const py::object& weights_obj) {
    if (py::isinstance<motor_model::MPCWeightsValues>(weights_obj)) {
        return weights_obj.cast<motor_model::MPCWeightsValues>();
    }

    auto attr = [&weights_obj](const char* name) { return weights_obj.attr(name); };

    motor_model::MPCWeightsValues weights;
    weights.position = attr("position").cast<double>();
    weights.speed = attr("speed").cast<double>();
    weights.voltage = attr("voltage").cast<double>();
    weights.delta_voltage = attr("delta_voltage").cast<double>();
    weights.terminal_position = attr("terminal_position").cast<double>();
    return weights;
}

motor_model::MPCWeightsValues ResolveWeights(const py::object& weights_obj) {
    if (weights_obj.is_none()) {
        return motor_model::MPCWeightsValues{};
    }
    return WeightsFromPython(weights_obj);
}

motor_model::MotorParams ResolveMotorParams(const py::object& motor_obj) {
    if (py::isinstance<motor_model::MotorParams>(motor_obj)) {
        return motor_obj.cast<motor_model::MotorParams>();
    }
    return MotorParamsFromPython(motor_obj);
}

}  // namespace

PYBIND11_MODULE(continuous_mpc, m) {
    m.doc() = "Native continuous MPC controller";

    namespace mm = motor_model;

    py::class_<mm::MPCWeightsValues>(m, "MPCWeights")
        .def(
            py::init<double, double, double, double, double>(),
            py::kw_only(),
            py::arg("position") = 300.0,
            py::arg("speed") = 0.5,
            py::arg("voltage") = 0.02,
            py::arg("delta_voltage") = 0.25,
            py::arg("terminal_position") = 700.0)
        .def_readwrite("position", &mm::MPCWeightsValues::position)
        .def_readwrite("speed", &mm::MPCWeightsValues::speed)
        .def_readwrite("voltage", &mm::MPCWeightsValues::voltage)
        .def_readwrite("delta_voltage", &mm::MPCWeightsValues::delta_voltage)
        .def_readwrite("terminal_position", &mm::MPCWeightsValues::terminal_position);

    py::class_<mm::MotorParams>(m, "MotorParameters")
        .def(
            py::init([](double resistance,
                        double inductance,
                        double kv,
                        double inertia,
                        double viscous_friction,
                        double coulomb_friction,
                        double static_friction,
                        double stop_speed_threshold,
                        double spring_constant,
                        double spring_compression_ratio,
                        double spring_zero_position,
                        double lvdt_full_scale,
                        int integration_substeps) {
                double ke = kv > 0.0 ? 1.0 / kv : 0.0;
                double kt = ke;
                return mm::MotorParams(
                    resistance,
                    inductance,
                    kv,
                    inertia,
                    viscous_friction,
                    coulomb_friction,
                    static_friction,
                    stop_speed_threshold,
                    spring_constant,
                    spring_compression_ratio,
                    spring_zero_position,
                    lvdt_full_scale,
                    ke,
                    kt,
                    integration_substeps);
            }),
            py::kw_only(),
            py::arg("resistance") = 28.0,
            py::arg("inductance") = 16e-3,
            py::arg("kv") = kDefaultKv,
            py::arg("inertia") = 4.8e-4,
            py::arg("viscous_friction") = 1.9e-4,
            py::arg("coulomb_friction") = 2.1e-2,
            py::arg("static_friction") = 2.4e-2,
            py::arg("stop_speed_threshold") = 1e-4,
            py::arg("spring_constant") = 9.5e-4,
            py::arg("spring_compression_ratio") = 0.4,
            py::arg("spring_zero_position") = 0.0,
            py::arg("lvdt_full_scale") = 0.1,
            py::arg("integration_substeps") = 1)
        .def_readwrite("resistance", &mm::MotorParams::resistance)
        .def_readwrite("inductance", &mm::MotorParams::inductance)
        .def_readwrite("kv", &mm::MotorParams::kv)
        .def_readwrite("inertia", &mm::MotorParams::inertia)
        .def_readwrite("viscous_friction", &mm::MotorParams::viscous_friction)
        .def_readwrite("coulomb_friction", &mm::MotorParams::coulomb_friction)
        .def_readwrite("static_friction", &mm::MotorParams::static_friction)
        .def_readwrite("stop_speed_threshold", &mm::MotorParams::stop_speed_threshold)
        .def_readwrite("spring_constant", &mm::MotorParams::spring_constant)
        .def_readwrite("spring_compression_ratio", &mm::MotorParams::spring_compression_ratio)
        .def_readwrite("spring_zero_position", &mm::MotorParams::spring_zero_position)
        .def_readwrite("lvdt_full_scale", &mm::MotorParams::lvdt_full_scale)
        .def_readwrite("ke", &mm::MotorParams::ke)
        .def_readwrite("kt", &mm::MotorParams::kt)
        .def_readwrite("integration_substeps", &mm::MotorParams::integration_substeps)
        .def("with_inductance", &mm::MotorParams::WithInductance);

    py::class_<mm::ContMPCController, std::shared_ptr<mm::ContMPCController>>(m, "ContMPCController")
        .def(
            py::init([](const mm::MotorParams& motor_params,
                        double dt,
                        int horizon,
                        double voltage_limit,
                        double target_lvdt,
                        std::optional<mm::MPCWeightsValues> weights,
                        double position_tolerance,
                        double static_friction_penalty,
                        std::optional<double> friction_compensation,
                        double auto_fc_gain,
                        double auto_fc_floor,
                        std::optional<double> auto_fc_cap,
                        double friction_blend_error_low,
                        double friction_blend_error_high,
                        int internal_substeps,
                        bool robust_electrical,
                        std::optional<double> electrical_alpha,
                        double inductance_rel_uncertainty,
                        double pd_blend,
                        double pd_kp,
                        double pd_kd,
                        double pi_ki,
                        double pi_limit,
                        bool pi_gate_saturation,
                        bool pi_gate_blocked,
                        bool pi_gate_error_band,
                        bool pi_leak_near_setpoint,
                        bool use_model_integrator,
                        int opt_iters,
                        double opt_step,
                        std::optional<double> opt_eps) {
                mm::MPCWeightsValues resolved = weights.value_or(mm::MPCWeightsValues{});
                return std::make_shared<mm::ContMPCController>(
                    motor_params,
                    dt,
                    horizon,
                    voltage_limit,
                    target_lvdt,
                    resolved,
                    position_tolerance,
                    static_friction_penalty,
                    friction_compensation,
                    auto_fc_gain,
                    auto_fc_floor,
                    auto_fc_cap,
                    friction_blend_error_low,
                    friction_blend_error_high,
                    internal_substeps,
                    robust_electrical,
                    electrical_alpha,
                    inductance_rel_uncertainty,
                    pd_blend,
                    pd_kp,
                    pd_kd,
                    pi_ki,
                    pi_limit,
                    pi_gate_saturation,
                    pi_gate_blocked,
                    pi_gate_error_band,
                    pi_leak_near_setpoint,
                    use_model_integrator,
                    opt_iters,
                    opt_step,
                    opt_eps);
            }),
            py::arg("motor_params"),
            py::kw_only(),
            py::arg("dt"),
            py::arg("horizon") = 5,
            py::arg("voltage_limit") = 10.0,
            py::arg("target_lvdt") = 0.0,
            py::arg("weights") = std::optional<mm::MPCWeightsValues>(),
            py::arg("position_tolerance") = 0.02,
            py::arg("static_friction_penalty") = 50.0,
            py::arg("friction_compensation") = std::optional<double>(),
            py::arg("auto_fc_gain") = 2.5,
            py::arg("auto_fc_floor") = 0.0,
            py::arg("auto_fc_cap") = std::optional<double>(),
            py::arg("friction_blend_error_low") = 0.2,
            py::arg("friction_blend_error_high") = 0.5,
            py::arg("internal_substeps") = 30,
            py::arg("robust_electrical") = true,
            py::arg("electrical_alpha") = std::optional<double>(),
            py::arg("inductance_rel_uncertainty") = 0.5,
            py::arg("pd_blend") = 0.7,
            py::arg("pd_kp") = 6.0,
            py::arg("pd_kd") = 0.4,
            py::arg("pi_ki") = 0.0,
            py::arg("pi_limit") = 5.0,
            py::arg("pi_gate_saturation") = true,
            py::arg("pi_gate_blocked") = true,
            py::arg("pi_gate_error_band") = false,
            py::arg("pi_leak_near_setpoint") = true,
            py::arg("use_model_integrator") = false,
            py::arg("opt_iters") = 10,
            py::arg("opt_step") = 0.1,
            py::arg("opt_eps") = std::optional<double>())
        .def(
            py::init([](py::object motor,
                        double dt,
                        int horizon,
                        double voltage_limit,
                        double target_lvdt,
                        py::object weights,
                        double position_tolerance,
                        double static_friction_penalty,
                        std::optional<double> friction_compensation,
                        double auto_fc_gain,
                        double auto_fc_floor,
                        std::optional<double> auto_fc_cap,
                        double friction_blend_error_low,
                        double friction_blend_error_high,
                        int internal_substeps,
                        bool robust_electrical,
                        std::optional<double> electrical_alpha,
                        double inductance_rel_uncertainty,
                        double pd_blend,
                        double pd_kp,
                        double pd_kd,
                        double pi_ki,
                        double pi_limit,
                        bool pi_gate_saturation,
                        bool pi_gate_blocked,
                        bool pi_gate_error_band,
                        bool pi_leak_near_setpoint,
                        bool use_model_integrator,
                        int opt_iters,
                        double opt_step,
                        std::optional<double> opt_eps) {
                mm::MotorParams params = ResolveMotorParams(motor);
                mm::MPCWeightsValues resolved = ResolveWeights(weights);
                return std::make_shared<mm::ContMPCController>(
                    params,
                    dt,
                    horizon,
                    voltage_limit,
                    target_lvdt,
                    resolved,
                    position_tolerance,
                    static_friction_penalty,
                    friction_compensation,
                    auto_fc_gain,
                    auto_fc_floor,
                    auto_fc_cap,
                    friction_blend_error_low,
                    friction_blend_error_high,
                    internal_substeps,
                    robust_electrical,
                    electrical_alpha,
                    inductance_rel_uncertainty,
                    pd_blend,
                    pd_kp,
                    pd_kd,
                    pi_ki,
                    pi_limit,
                    pi_gate_saturation,
                    pi_gate_blocked,
                    pi_gate_error_band,
                    pi_leak_near_setpoint,
                    use_model_integrator,
                    opt_iters,
                    opt_step,
                    opt_eps);
            }),
            py::arg("motor"),
            py::kw_only(),
            py::arg("dt"),
            py::arg("horizon") = 5,
            py::arg("voltage_limit") = 10.0,
            py::arg("target_lvdt") = 0.0,
            py::arg("weights") = py::none(),
            py::arg("position_tolerance") = 0.02,
            py::arg("static_friction_penalty") = 50.0,
            py::arg("friction_compensation") = std::optional<double>(),
            py::arg("auto_fc_gain") = 2.5,
            py::arg("auto_fc_floor") = 0.0,
            py::arg("auto_fc_cap") = std::optional<double>(),
            py::arg("friction_blend_error_low") = 0.2,
            py::arg("friction_blend_error_high") = 0.5,
            py::arg("internal_substeps") = 30,
            py::arg("robust_electrical") = true,
            py::arg("electrical_alpha") = std::optional<double>(),
            py::arg("inductance_rel_uncertainty") = 0.5,
            py::arg("pd_blend") = 0.7,
            py::arg("pd_kp") = 6.0,
            py::arg("pd_kd") = 0.4,
            py::arg("pi_ki") = 0.0,
            py::arg("pi_limit") = 5.0,
            py::arg("pi_gate_saturation") = true,
            py::arg("pi_gate_blocked") = true,
            py::arg("pi_gate_error_band") = false,
            py::arg("pi_leak_near_setpoint") = true,
            py::arg("use_model_integrator") = false,
            py::arg("opt_iters") = 10,
            py::arg("opt_step") = 0.1,
            py::arg("opt_eps") = std::optional<double>())
        .def_property(
            "weights",
            [](mm::ContMPCController& self) -> mm::MPCWeightsValues& {
                return self.mutable_weights();
            },
            [](mm::ContMPCController& self, py::object weights_obj) {
                if (weights_obj.is_none()) {
                    self.set_weights(mm::MPCWeightsValues{});
                } else if (py::isinstance<mm::MPCWeightsValues>(weights_obj)) {
                    self.set_weights(weights_obj.cast<mm::MPCWeightsValues>());
                } else {
                    self.set_weights(WeightsFromPython(weights_obj));
                }
            },
            py::return_value_policy::reference_internal)
        .def_property_readonly("_motor", [](const mm::ContMPCController& self) {
            return self.motor_params();
        })
        .def_property_readonly("_prediction_models", [](const mm::ContMPCController& self) {
            return self.prediction_models();
        })
        .def_property_readonly("_state", [](const mm::ContMPCController& self) {
            const auto& state = self.state();
            return py::make_tuple(state[0], state[1], state[2]);
        })
        .def_property_readonly("_nominal_state", [](const mm::ContMPCController& self) {
            const auto& state = self.nominal_state();
            return py::make_tuple(state[0], state[1], state[2]);
        })
        .def_property_readonly("_u_seq", [](const mm::ContMPCController& self) {
            return self.u_sequence();
        })
        .def(
            "adapt_to_motor",
            [](mm::ContMPCController& self,
               py::object motor,
               std::optional<double> voltage_limit,
               std::optional<double> friction_compensation) {
                self.adapt_to_motor(ResolveMotorParams(motor), voltage_limit, friction_compensation);
            },
            py::arg("motor"),
            py::kw_only(),
            py::arg("voltage_limit") = std::optional<double>(),
            py::arg("friction_compensation") = std::optional<double>())
        .def(
            "reset",
            &mm::ContMPCController::reset,
            py::kw_only(),
            py::arg("initial_measurement"),
            py::arg("initial_current"),
            py::arg("initial_speed"))
        .def(
            "update",
            &mm::ContMPCController::update,
            py::kw_only(),
            py::arg("time"),
            py::arg("measurement"))
        .def_readwrite("dt", &mm::ContMPCController::dt)
        .def_readwrite("horizon", &mm::ContMPCController::horizon)
        .def_readwrite("voltage_limit", &mm::ContMPCController::voltage_limit)
        .def_readwrite("target_lvdt", &mm::ContMPCController::target_lvdt)
        .def_readwrite("position_tolerance", &mm::ContMPCController::position_tolerance)
        .def_readwrite("static_friction_penalty", &mm::ContMPCController::static_friction_penalty)
        .def_readwrite("internal_substeps", &mm::ContMPCController::internal_substeps)
        .def_readwrite("robust_electrical", &mm::ContMPCController::robust_electrical)
        .def_readwrite("inductance_rel_uncertainty", &mm::ContMPCController::inductance_rel_uncertainty)
        .def_readwrite("pd_blend", &mm::ContMPCController::pd_blend)
        .def_readwrite("pd_kp", &mm::ContMPCController::pd_kp)
        .def_readwrite("pd_kd", &mm::ContMPCController::pd_kd)
        .def_readwrite("pi_ki", &mm::ContMPCController::pi_ki)
        .def_readwrite("pi_limit", &mm::ContMPCController::pi_limit)
        .def_readwrite("pi_gate_saturation", &mm::ContMPCController::pi_gate_saturation)
        .def_readwrite("pi_gate_blocked", &mm::ContMPCController::pi_gate_blocked)
        .def_readwrite("pi_gate_error_band", &mm::ContMPCController::pi_gate_error_band)
        .def_readwrite("pi_leak_near_setpoint", &mm::ContMPCController::pi_leak_near_setpoint)
        .def_readwrite("use_model_integrator", &mm::ContMPCController::use_model_integrator)
        .def_readwrite("auto_fc_gain", &mm::ContMPCController::auto_fc_gain)
        .def_readwrite("auto_fc_floor", &mm::ContMPCController::auto_fc_floor)
        .def_readwrite("auto_fc_cap", &mm::ContMPCController::auto_fc_cap)
        .def_readwrite("friction_blend_error_low", &mm::ContMPCController::friction_blend_error_low)
        .def_readwrite("friction_blend_error_high", &mm::ContMPCController::friction_blend_error_high)
        .def_readwrite("model_bias_limit", &mm::ContMPCController::model_bias_limit)
        .def_readwrite("friction_compensation", &mm::ContMPCController::friction_compensation)
        .def_readwrite("output_duty_cycle", &mm::ContMPCController::output_duty_cycle)
        .def_readwrite("_opt_iters", &mm::ContMPCController::_opt_iters)
        .def_readwrite("_opt_step", &mm::ContMPCController::_opt_step)
        .def_readwrite("_opt_eps", &mm::ContMPCController::_opt_eps)
        .def_readwrite("_electrical_alpha", &mm::ContMPCController::_electrical_alpha)
        .def_readwrite("_int_err", &mm::ContMPCController::_int_err)
        .def_readwrite("_u_bias", &mm::ContMPCController::_u_bias)
        .def_readwrite("_user_friction_compensation_request", &mm::ContMPCController::_user_friction_compensation_request)
        .def_readwrite("_auto_friction_compensation", &mm::ContMPCController::_auto_friction_compensation)
        .def_readwrite("_active_user_friction_compensation", &mm::ContMPCController::_active_user_friction_compensation)
        .def_readwrite("_last_voltage", &mm::ContMPCController::_last_voltage)
        .def_readwrite("_last_measured_position", &mm::ContMPCController::_last_measured_position)
        .def_readwrite("_last_measurement_time", &mm::ContMPCController::_last_measurement_time);

    RegisterContMPCControllerManager(m);
}

