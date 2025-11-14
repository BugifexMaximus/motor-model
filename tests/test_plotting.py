import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from motor_model import SimulationResult, plot_simulation


def make_result(**overrides: object) -> SimulationResult:
    base = dict(
        time=[0.0, 0.1, 0.2, 0.3],
        current=[0.0, 0.2, 0.4, 0.6],
        speed=[0.0, 0.5, 1.0, 1.5],
        position=[0.0, 0.01, 0.02, 0.03],
        torque=[0.0, 0.05, 0.08, 0.1],
        voltage=[0.0, 1.0, 1.5, 2.0],
        lvdt_time=[0.0, 0.1, 0.2, 0.3],
        lvdt=[0.0, 0.2, 0.4, 0.6],
    )
    base.update(overrides)
    return SimulationResult(**base)  # type: ignore[arg-type]


def test_plot_simulation_excludes_lvdt_by_default() -> None:
    result = make_result()
    fig = plot_simulation(result)
    try:
        axes = fig.axes
        assert len(axes) == 4
        assert all(ax.get_ylabel() != "LVDT (normalized)" for ax in axes)
    finally:
        plt.close(fig)


def test_plot_simulation_includes_lvdt_when_requested() -> None:
    result = make_result()
    fig = plot_simulation(result, include_lvdt=True)
    try:
        lvdt_axes = [ax for ax in fig.axes if ax.get_ylabel() == "LVDT (normalized)"]
        assert len(lvdt_axes) == 1
        lvdt_axis = lvdt_axes[0]
        line = lvdt_axis.lines[0]
        assert list(line.get_xdata()) == result.lvdt_time
        assert list(line.get_ydata()) == result.lvdt
    finally:
        plt.close(fig)


def test_plot_simulation_adds_integrator_axis() -> None:
    samples = [0.0, 0.05, 0.1, 0.08]
    result = make_result(pi_integrator=samples, model_integrator=[s * 0.5 for s in samples])
    fig = plot_simulation(result, include_integrator=True)
    try:
        assert len(fig.axes) == 5
        integrator_ax = fig.axes[4]
        assert integrator_ax.get_ylabel() == "Integrator [V]"
        assert len(integrator_ax.lines) == 2
    finally:
        plt.close(fig)


def test_plot_simulation_adds_planned_voltage_axis() -> None:
    plan = [
        [0.0, 0.1, 0.2],
        [0.1, 0.2, 0.3],
        [0.2, 0.3, 0.4],
        [0.3, 0.4, 0.5],
    ]
    result = make_result(planned_voltage=plan)
    fig = plot_simulation(result, include_planned_voltage=True)
    try:
        assert len(fig.axes) == 5
        plan_ax = fig.axes[4]
        assert plan_ax.get_ylabel() == "Planned V [V]"
        assert len(plan_ax.lines) == len(plan[0])
    finally:
        plt.close(fig)


def test_plot_simulation_requires_matching_axes_count() -> None:
    result = make_result()
    fig, axes = plt.subplots(4, 1)
    try:
        with pytest.raises(ValueError):
            plot_simulation(result, axes=axes, include_integrator=True)
    finally:
        plt.close(fig)

