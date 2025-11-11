#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <chrono>
#include <future>
#include <memory>
#include <optional>
#include <utility>

#include "continuous_mpc_core.h"
#include "continuous_mpc_manager.h"

namespace py = pybind11;
namespace mm = motor_model;

namespace {

class ContMPCStepFuture {
   public:
    explicit ContMPCStepFuture(std::future<double>&& future)
        : future_(future.share()) {}

    double result(std::optional<double> timeout_seconds = std::nullopt) {
        bool timed_out = false;
        double value = 0.0;
        {
            py::gil_scoped_release release;
            if (timeout_seconds && *timeout_seconds >= 0.0) {
                const auto status = future_.wait_for(std::chrono::duration<double>(*timeout_seconds));
                if (status == std::future_status::timeout) {
                    timed_out = true;
                } else {
                    value = future_.get();
                }
            } else {
                future_.wait();
                value = future_.get();
            }
        }
        if (timed_out) {
            PyErr_SetString(PyExc_TimeoutError, "ContMPCStepFuture timed out while waiting for result");
            throw py::error_already_set();
        }
        return value;
    }

    bool done() const {
        py::gil_scoped_release release;
        return future_.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
    }

   private:
    std::shared_future<double> future_;
};

class PyContMPCControllerManager {
   public:
    explicit PyContMPCControllerManager(std::shared_ptr<mm::ContMPCController> controller)
        : manager_(std::move(controller)) {}

    void start_realtime(py::function provider, py::object callback = py::none(), double frequency_hz = 0.0) {
        auto provider_wrapper = [provider = std::move(provider)]() -> std::pair<double, double> {
            py::gil_scoped_acquire gil;
            py::object result = provider();
            return result.cast<std::pair<double, double>>();
        };

        mm::ContMPCControllerManager::StepCallback callback_wrapper;
        if (!callback.is_none()) {
            py::function cb = callback.cast<py::function>();
            callback_wrapper = [cb = std::move(cb)](double time, double measurement, double control) {
                py::gil_scoped_acquire gil;
                cb(time, measurement, control);
            };
        }

        py::gil_scoped_release release;
        manager_.StartRealTime(std::move(provider_wrapper), std::move(callback_wrapper), frequency_hz);
    }

    ContMPCStepFuture submit_step(double time, double measurement, py::object callback = py::none()) {
        mm::ContMPCControllerManager::StepCallback callback_wrapper;
        if (!callback.is_none()) {
            py::function cb = callback.cast<py::function>();
            callback_wrapper = [cb = std::move(cb)](double step_time, double measurement_value, double control) {
                py::gil_scoped_acquire gil;
                cb(step_time, measurement_value, control);
            };
        }

        std::future<double> future;
        {
            py::gil_scoped_release release;
            future = manager_.SubmitStep(time, measurement, std::move(callback_wrapper));
        }
        return ContMPCStepFuture(std::move(future));
    }

    void stop() {
        py::gil_scoped_release release;
        manager_.Stop();
    }

   private:
    mm::ContMPCControllerManager manager_;
};

}  // namespace

void RegisterContMPCControllerManager(py::module_& m) {
    py::class_<ContMPCStepFuture>(m, "ContMPCStepFuture")
        .def("result", &ContMPCStepFuture::result, py::arg("timeout") = std::nullopt)
        .def("done", &ContMPCStepFuture::done);

    py::class_<PyContMPCControllerManager>(m, "ContMPCControllerManager")
        .def(py::init<std::shared_ptr<mm::ContMPCController>>(), py::arg("controller"))
        .def("start_realtime", &PyContMPCControllerManager::start_realtime, py::arg("provider"), py::arg("callback") = py::none(), py::arg("frequency_hz") = 0.0)
        .def("submit_step", &PyContMPCControllerManager::submit_step, py::arg("time"), py::arg("measurement"), py::arg("callback") = py::none())
        .def("stop", &PyContMPCControllerManager::stop);
}

