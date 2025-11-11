"""Tests for the native continuous MPC controller manager."""

from __future__ import annotations

import math
import threading

import pytest

from motor_model._native import load_continuous_mpc


@pytest.fixture()
def native_module():
    return load_continuous_mpc()


def _make_controller(module) -> object:
    motor = module.MotorParameters(
        resistance=24.0,
        inductance=14e-3,
        kv=8.5,
        inertia=4.8e-4,
        viscous_friction=1.9e-4,
        coulomb_friction=2.1e-2,
        static_friction=2.4e-2,
        stop_speed_threshold=1e-4,
        spring_constant=9.5e-4,
        spring_compression_ratio=0.4,
        lvdt_full_scale=0.1,
        integration_substeps=4,
    )
    weights = module.MPCWeights(
        position=250.0,
        speed=0.6,
        voltage=0.03,
        delta_voltage=0.2,
        terminal_position=500.0,
    )
    controller = module.ContMPCController(
        motor_params=motor,
        dt=0.01,
        horizon=4,
        voltage_limit=6.5,
        target_lvdt=0.0,
        weights=weights,
        position_tolerance=0.01,
        static_friction_penalty=30.0,
        friction_compensation=None,
        auto_fc_gain=1.2,
        auto_fc_floor=0.02,
        auto_fc_cap=None,
        friction_blend_error_low=0.05,
        friction_blend_error_high=0.3,
        internal_substeps=8,
        robust_electrical=True,
        electrical_alpha=None,
        inductance_rel_uncertainty=0.0,
        pd_blend=0.55,
        pd_kp=4.0,
        pd_kd=0.3,
        pi_ki=0.4,
        pi_limit=2.5,
        pi_gate_saturation=True,
        pi_gate_blocked=True,
        pi_gate_error_band=True,
        pi_leak_near_setpoint=True,
        use_model_integrator=False,
        opt_iters=5,
        opt_step=0.1,
        opt_eps=None,
    )
    controller.reset(
        initial_measurement=0.0,
        initial_current=0.0,
        initial_speed=0.0,
    )
    return controller


def test_manager_realtime_mode_invokes_callbacks(native_module) -> None:
    controller = _make_controller(native_module)
    dt = controller.dt
    manager = native_module.ContMPCControllerManager(controller)

    callback_event = threading.Event()
    callback_count = 0
    provider_calls = 0
    lock = threading.Lock()

    def provider() -> tuple[float, float]:
        nonlocal provider_calls
        with lock:
            provider_calls += 1
            current_call = provider_calls
        if current_call >= 3:
            callback_event.set()
        return current_call * dt, 0.0

    def callback(time_value: float, measurement: float, control: float) -> None:
        nonlocal callback_count
        assert measurement == pytest.approx(0.0, abs=1e-12)
        assert math.isfinite(control)
        callback_count += 1

    try:
        manager.start_realtime(provider, callback, frequency_hz=1.0 / dt)
        assert callback_event.wait(0.5), "Manager did not invoke provider sufficiently fast"
    finally:
        manager.stop()

    assert provider_calls >= 3
    assert callback_count >= 1


def test_manager_triggered_mode_returns_future(native_module) -> None:
    controller = _make_controller(native_module)
    manager = native_module.ContMPCControllerManager(controller)

    callback_event = threading.Event()
    recorded_control: list[float] = []
    result: float | None = None

    def callback(time_value: float, measurement: float, control: float) -> None:
        recorded_control.append(control)
        callback_event.set()

    try:
        future = manager.submit_step(time=0.01, measurement=0.0, callback=callback)
        result = future.result(timeout=1.0)
        assert future.done()
    finally:
        manager.stop()

    assert callback_event.wait(0.1)
    assert recorded_control, "Callback should record at least one control value"
    assert result is not None
    assert result == pytest.approx(recorded_control[0], rel=1e-12, abs=1e-12)
    assert math.isfinite(result)

