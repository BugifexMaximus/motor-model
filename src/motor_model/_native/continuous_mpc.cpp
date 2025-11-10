#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "continuous_mpc_core.h"

PYBIND11_MODULE(continuous_mpc, m) {
    m.doc() = "Native continuous MPC controller";

    namespace mm = motor_model;

    py::class_<mm::ContMPCController>(m, "ContMPCController")
        .def(
            py::init<
                py::object,
                double,
                int,
                double,
                double,
                py::object,
                double,
                double,
                std::optional<double>,
                double,
                double,
                std::optional<double>,
                double,
                double,
                int,
                bool,
                std::optional<double>,
                double,
                double,
                double,
                double,
                double,
                double,
                bool,
                bool,
                bool,
                bool,
                bool,
                int,
                double,
                std::optional<double>>(),
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
            py::arg("pi_gate_error_band") = true,
            py::arg("pi_leak_near_setpoint") = true,
            py::arg("use_model_integrator") = false,
            py::arg("opt_iters") = 10,
            py::arg("opt_step") = 0.1,
            py::arg("opt_eps") = std::optional<double>())
        .def_property("weights", &mm::ContMPCController::get_weights, &mm::ContMPCController::set_weights)
        .def_property_readonly("_motor", &mm::ContMPCController::motor)
        .def_property_readonly("_prediction_models", &mm::ContMPCController::prediction_models)
        .def_property_readonly("_state", &mm::ContMPCController::state_tuple)
        .def_property_readonly("_nominal_state", &mm::ContMPCController::nominal_state_tuple)
        .def_property_readonly("_u_seq", &mm::ContMPCController::u_sequence_tuple)
        .def(
            "adapt_to_motor",
            &mm::ContMPCController::adapt_to_motor,
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
        .def_readwrite(
            "_user_friction_compensation_request",
            &mm::ContMPCController::_user_friction_compensation_request)
        .def_readwrite(
            "_auto_friction_compensation", &mm::ContMPCController::_auto_friction_compensation)
        .def_readwrite(
            "_active_user_friction_compensation",
            &mm::ContMPCController::_active_user_friction_compensation)
        .def_readwrite("_last_voltage", &mm::ContMPCController::_last_voltage)
        .def_readwrite(
            "_last_measured_position", &mm::ContMPCController::_last_measured_position)
        .def_readwrite(
            "_last_measurement_time", &mm::ContMPCController::_last_measurement_time);
}

