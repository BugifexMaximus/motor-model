"""Tests for the experimental pybind11-backed module."""

from importlib import import_module

import pytest


@pytest.fixture(scope="module")
def native_module():
    return import_module("motor_model._native.test_module")


def test_add(native_module):
    """The native add function should add two integers."""

    assert native_module.add(2, 3) == 5
    assert native_module.add(-1, 1) == 0
