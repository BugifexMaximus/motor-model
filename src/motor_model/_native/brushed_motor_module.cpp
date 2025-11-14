#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <utility>

#include "brushed_motor_core.h"

#include <pybind11/pytypes.h>

namespace py = pybind11;
using namespace py::literals;
using motor_model::native::BrushedMotorModel;
using motor_model::native::ControllerDiagnostics;
using motor_model::native::FeedbackController;
using motor_model::native::SimulationResult;
using motor_model::native::VoltageSource;

namespace {

struct SourceWrapper {
  VoltageSource function;
  bool uses_python;
};

bool is_callable(const py::object &obj) {
  return PyCallable_Check(obj.ptr());
}

SourceWrapper make_voltage_source(const py::object &obj, const char *name) {
  if (py::isinstance<py::float_>(obj) || py::isinstance<py::int_>(obj)) {
    const double value = obj.cast<double>();
    return {BrushedMotorModel::constant_voltage_source(value), false};
  }
  if (obj.is_none()) {
    throw py::type_error(std::string(name) + " cannot be None");
  }
  if (is_callable(obj)) {
    py::function func = obj;
    return { [func](double t) {
               py::gil_scoped_acquire gil;
               return func(t).cast<double>();
             },
             true };
  }
  throw py::type_error(std::string("Unsupported ") + name + " type");
}

std::optional<double> maybe_double(const py::object &obj, const char *name) {
  if (obj.is_none()) {
    return std::nullopt;
  }
  try {
    return obj.cast<double>();
  } catch (const py::cast_error &) {
    throw py::type_error(std::string(name) + " must be a float or None");
  }
}

class PyFeedbackController : public FeedbackController {
 public:
  explicit PyFeedbackController(py::object controller) : controller_(std::move(controller)) {}

  void reset(double initial_measurement, double initial_current, double initial_speed) override {
    if (py::hasattr(controller_, "reset")) {
      controller_.attr("reset")("initial_measurement"_a = initial_measurement,
                                 "initial_current"_a = initial_current,
                                 "initial_speed"_a = initial_speed);
    }
  }

  double update(double time, double measurement) override {
    py::object result = controller_.attr("update")("time"_a = time,
                                                   "measurement"_a = measurement);
    return result.cast<double>();
  }

  std::optional<ControllerDiagnostics> diagnostics() const override {
    py::gil_scoped_acquire gil;

    ControllerDiagnostics diagnostics;
    bool has_value = false;

    if (py::hasattr(controller_, "_int_err")) {
      diagnostics.pi_integrator = controller_.attr("_int_err").cast<double>();
      has_value = true;
    }

    if (py::hasattr(controller_, "_u_bias")) {
      diagnostics.model_integrator = controller_.attr("_u_bias").cast<double>();
      has_value = true;
    }

    if (py::hasattr(controller_, "_u_seq")) {
      diagnostics.planned_voltage = controller_.attr("_u_seq").cast<std::vector<double>>();
      if (!diagnostics.planned_voltage.empty()) {
        has_value = true;
      }
    }

    if (!has_value) {
      return std::nullopt;
    }

    return diagnostics;
  }

 private:
  py::object controller_;
};

bool has_update_callable(const py::object &obj) {
  if (!py::hasattr(obj, "update")) {
    return false;
  }
  py::object update = obj.attr("update");
  return PyCallable_Check(update.ptr());
}

}  // namespace

PYBIND11_MODULE(brushed_motor, m) {
  m.doc() = "Native brushed DC motor model";

  m.def("rpm_per_volt_to_rad_per_sec_per_volt",
        &motor_model::native::rpm_per_volt_to_rad_per_sec_per_volt,
        "Convert RPM/V to rad/s/V");
  m.def("rad_per_sec_per_volt_to_rpm_per_volt",
        &motor_model::native::rad_per_sec_per_volt_to_rpm_per_volt,
        "Convert rad/s/V to RPM/V");

  py::class_<SimulationResult>(m, "SimulationResult")
      .def_readonly("time", &SimulationResult::time)
      .def_readonly("current", &SimulationResult::current)
      .def_readonly("speed", &SimulationResult::speed)
      .def_readonly("position", &SimulationResult::position)
      .def_readonly("torque", &SimulationResult::torque)
      .def_readonly("voltage", &SimulationResult::voltage)
      .def_readonly("lvdt_time", &SimulationResult::lvdt_time)
      .def_readonly("lvdt", &SimulationResult::lvdt)
      .def_readonly("pi_integrator", &SimulationResult::pi_integrator)
      .def_readonly("model_integrator", &SimulationResult::model_integrator)
      .def_readonly("planned_voltage", &SimulationResult::planned_voltage);

  py::class_<BrushedMotorModel>(m, "BrushedMotorModel")
      .def(py::init<double, double, double, double, double, double, double, double, double, double, double, double, int, std::optional<std::uint32_t>>(),
           py::kw_only(),
           py::arg("resistance") = 28.0,
           py::arg("inductance") = 16e-3,
           py::arg("kv") = motor_model::native::rpm_per_volt_to_rad_per_sec_per_volt(7.0),
           py::arg("inertia") = 4.8e-4,
           py::arg("viscous_friction") = 1.9e-4,
           py::arg("coulomb_friction") = 2.1e-2,
           py::arg("static_friction") = 2.4e-2,
           py::arg("stop_speed_threshold") = 1e-4,
           py::arg("spring_constant") = 9.5e-4,
           py::arg("spring_compression_ratio") = 0.4,
           py::arg("lvdt_full_scale") = 0.1,
           py::arg("lvdt_noise_std") = 5e-3,
           py::arg("integration_substeps") = 1,
           py::arg("rng_seed") = std::optional<std::uint32_t>())
      .def_property_readonly("speed_constant_rpm_per_volt", &BrushedMotorModel::speed_constant_rpm_per_volt)
      .def_property("integration_substeps",
                    &BrushedMotorModel::integration_substeps,
                    &BrushedMotorModel::set_integration_substeps)
      .def_property_readonly("resistance", &BrushedMotorModel::resistance)
      .def_property_readonly("inductance", &BrushedMotorModel::inductance)
      .def_property_readonly("kv", &BrushedMotorModel::kv)
      .def_property_readonly("inertia", &BrushedMotorModel::inertia)
      .def_property_readonly("viscous_friction", &BrushedMotorModel::viscous_friction)
      .def_property_readonly("coulomb_friction", &BrushedMotorModel::coulomb_friction)
      .def_property_readonly("static_friction", &BrushedMotorModel::static_friction)
      .def_property_readonly("stop_speed_threshold", &BrushedMotorModel::stop_speed_threshold)
      .def_property_readonly("spring_constant", &BrushedMotorModel::spring_constant)
      .def_property_readonly("spring_compression_ratio", &BrushedMotorModel::spring_compression_ratio)
      .def_property_readonly("lvdt_full_scale", &BrushedMotorModel::lvdt_full_scale)
      .def_property_readonly("lvdt_noise_std", &BrushedMotorModel::lvdt_noise_std)
      .def_property_readonly("_ke", &BrushedMotorModel::ke)
      .def_property_readonly("_kt", &BrushedMotorModel::kt)
      .def("_lvdt_measurement", &BrushedMotorModel::lvdt_measurement)
      .def("_spring_torque", &BrushedMotorModel::spring_torque)
      .def_static("_sign", &BrushedMotorModel::sign)
      .def("simulate",
           [](BrushedMotorModel &self,
              py::object voltage_obj,
              double duration,
              double dt,
              double initial_speed,
              double initial_current,
              py::object load_obj,
              py::object measurement_period_obj,
              py::object controller_period_obj) {
             std::unique_ptr<PyFeedbackController> controller;
             bool uses_controller = false;

             SourceWrapper load_wrapper = make_voltage_source(load_obj, "load_torque");

             SourceWrapper voltage_wrapper{};
             if (has_update_callable(voltage_obj)) {
               controller = std::make_unique<PyFeedbackController>(voltage_obj);
               uses_controller = true;
               voltage_wrapper = {BrushedMotorModel::constant_voltage_source(0.0), false};
             } else {
               voltage_wrapper = make_voltage_source(voltage_obj, "voltage");
             }

             std::optional<double> measurement_period = maybe_double(measurement_period_obj, "measurement_period");
             std::optional<double> controller_period = maybe_double(controller_period_obj, "controller_period");

             if (!uses_controller && controller_period.has_value()) {
               throw py::value_error("controller_period provided without a feedback controller");
             }

             bool release_gil = !voltage_wrapper.uses_python && !load_wrapper.uses_python && !uses_controller;

             SimulationResult result;
             if (release_gil) {
               py::gil_scoped_release release;
               result = self.simulate(voltage_wrapper.function,
                                      duration,
                                      dt,
                                      initial_speed,
                                      initial_current,
                                      std::move(load_wrapper.function),
                                      measurement_period,
                                      nullptr,
                                      controller_period);
             } else {
               if (uses_controller) {
                 result = self.simulate(voltage_wrapper.function,
                                        duration,
                                        dt,
                                        initial_speed,
                                        initial_current,
                                        std::move(load_wrapper.function),
                                        measurement_period,
                                        controller.get(),
                                        controller_period);
               } else {
                 result = self.simulate(voltage_wrapper.function,
                                        duration,
                                        dt,
                                        initial_speed,
                                        initial_current,
                                        std::move(load_wrapper.function),
                                        measurement_period,
                                        nullptr,
                                        controller_period);
               }
             }

             return result;
           },
           py::arg("voltage"),
           py::kw_only(),
           py::arg("duration"),
           py::arg("dt"),
           py::arg("initial_speed") = 0.0,
           py::arg("initial_current") = 0.0,
           py::arg("load_torque") = 0.0,
           py::arg("measurement_period") = py::none(),
           py::arg("controller_period") = py::none());
}
