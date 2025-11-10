"""Native extension modules for :mod:`motor_model`."""

from importlib import import_module
from types import ModuleType
from typing import Any

__all__ = ["load_test_module", "load_continuous_mpc"]


def load_test_module() -> ModuleType:
    """Load and return the compiled pybind11 test module.

    Returns
    -------
    ModuleType
        The loaded :mod:`motor_model._native.test_module` module.
    """

    return import_module("motor_model._native.test_module")


def load_continuous_mpc() -> ModuleType:
    """Load and return the native continuous MPC module."""

    return import_module("motor_model._native.continuous_mpc")


Module = Any  # backwards compat placeholder for future typing helpers
