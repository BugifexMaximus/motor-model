#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace motor_model {
namespace native {

int add(int lhs, int rhs) {
    return lhs + rhs;
}

}  // namespace native
}  // namespace motor_model

PYBIND11_MODULE(test_module, m) {
    m.doc() = "Test pybind11 module for motor_model";

    m.def(
        "add",
        &motor_model::native::add,
        "Add two integers using the native extension.",
        py::arg("lhs"),
        py::arg("rhs"));
}
