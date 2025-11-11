from setuptools import setup

from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "motor_model._native.test_module",
        ["src/motor_model/_native/test_module.cpp"],
    ),
    Pybind11Extension(
        "motor_model._native.continuous_mpc",
        [
            "src/motor_model/_native/continuous_mpc_pybind.cpp",
            "src/motor_model/_native/continuous_mpc_core.cpp",
        ],
    ),
    Pybind11Extension(
        "motor_model._native.brushed_motor",
        [
            "src/motor_model/_native/brushed_motor_module.cpp",
            "src/motor_model/_native/brushed_motor_core.cpp",
        ],
    ),
]

setup(ext_modules=ext_modules, cmdclass={"build_ext": build_ext})
