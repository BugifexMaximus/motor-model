"""Visualization helpers for simulation results."""

from __future__ import annotations

import math
from typing import Any, Sequence

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .brushed_motor import SimulationResult


def _as_floats(values: Any) -> list[float] | None:
    if values is None:
        return None
    try:
        return [float(value) for value in values]
    except TypeError:
        return None


def _as_sequences(values: Any) -> list[list[float]] | None:
    if values is None:
        return None
    sequences: list[list[float]] = []
    try:
        for entry in values:
            sequences.append([float(value) for value in entry])
    except TypeError:
        return None
    return sequences


def plot_simulation(
    result: SimulationResult | Any,
    *,
    axes: Sequence[Axes] | None = None,
    include_lvdt: bool = False,
    include_integrator: bool = False,
    include_planned_voltage: bool = False,
) -> Figure:
    """Plot the main time-series stored in a :class:`SimulationResult`.

    Parameters
    ----------
    result:
        Simulation outcome produced by :meth:`BrushedMotorModel.simulate`.
    axes:
        Optional sequence of Matplotlib axes used for plotting. When omitted a
        new figure with four vertically stacked subplots is created. Additional
        subplots are appended when ``include_integrator`` and
        ``include_planned_voltage`` are set.
        ``axes`` must contain the exact number of axes required for the
        requested plots ordered as position, speed, current, voltage and any
        additional diagnostics.
    include_lvdt:
        When ``True`` the LVDT measurement samples recorded in ``result.lvdt``
        are plotted alongside the position trace using a secondary y-axis. The
        measurements are expressed in their normalized ``[-1, 1]`` form.

    Returns
    -------
    matplotlib.figure.Figure
        The figure hosting the plots. This is the newly created figure when
        ``axes`` is not provided, otherwise it corresponds to ``axes[0].figure``.
    """

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        raise RuntimeError("Plotting requires matplotlib to be installed.") from exc

    extra_axes = int(include_integrator) + int(include_planned_voltage)
    axes_required = 4 + extra_axes

    if axes is None:
        base_height = 8.0
        height = base_height + 1.6 * extra_axes
        fig, axes = plt.subplots(axes_required, 1, sharex=True, figsize=(8.0, height))
    else:
        if len(axes) != axes_required:
            raise ValueError(
                f"axes must contain {axes_required} Matplotlib Axes objects when diagnostics are requested"
            )
        fig = axes[0].figure

    axes = list(axes)

    position_ax, speed_ax, current_ax, voltage_ax = axes[:4]

    time = result.time
    position_ax.plot(time, result.position, label="Position", color="#1f77b4")
    position_ax.set_ylabel("Position [rad]")
    position_ax.grid(True)

    speed_ax.plot(time, result.speed, label="Speed", color="#ff7f0e")
    speed_ax.set_ylabel("Speed [rad/s]")
    speed_ax.grid(True)

    current_ax.plot(time, result.current, label="Current", color="#2ca02c")
    current_ax.set_ylabel("Current [A]")
    current_ax.grid(True)

    voltage_ax.plot(time, result.voltage, label="Voltage", color="#d62728")
    voltage_ax.set_ylabel("Voltage [V]")
    voltage_ax.grid(True)

    if include_lvdt and result.lvdt and result.lvdt_time:
        lvdt_ax = position_ax.twinx()
        lvdt_ax.plot(
            result.lvdt_time,
            result.lvdt,
            label="LVDT measurement",
            color="#9467bd",
            linestyle="--",
            marker="o",
            markersize=3,
        )
        lvdt_ax.set_ylabel("LVDT (normalized)")
        lvdt_ax.set_ylim(-1.05, 1.05)
        lvdt_ax.grid(False)
        if position_ax.legend(loc="upper left") is None:
            position_ax.legend(loc="upper left")
        lvdt_ax.legend(loc="upper right")
    else:
        position_ax.legend(loc="upper right")

    next_axis_index = 4
    integrator_ax: Axes | None = None
    if include_integrator:
        integrator_ax = axes[next_axis_index]
        next_axis_index += 1
        pi_data = _as_floats(getattr(result, "pi_integrator", None))
        model_data = _as_floats(getattr(result, "model_integrator", None))

        plotted = False
        if pi_data and len(pi_data) == len(time):
            integrator_ax.plot(time, pi_data, label="PI integrator", color="#17becf")
            plotted = True
        if model_data and len(model_data) == len(time):
            integrator_ax.plot(time, model_data, label="Model bias", color="#8c564b")
            plotted = True

        integrator_ax.set_ylabel("Integrator [V]")
        integrator_ax.grid(True)
        if plotted:
            integrator_ax.legend(loc="upper right")
        else:
            integrator_ax.text(
                0.5,
                0.5,
                "Integrator data unavailable",
                transform=integrator_ax.transAxes,
                ha="center",
                va="center",
                fontsize="small",
                color="#666666",
            )

    if include_planned_voltage:
        plan_ax = axes[next_axis_index]
        plan_sequences = _as_sequences(getattr(result, "planned_voltage", None))
        if plan_sequences and len(plan_sequences) == len(time):
            horizon = max((len(seq) for seq in plan_sequences), default=0)
            cmap = plt.get_cmap("viridis_r")
            for step in range(horizon):
                series = []
                for seq in plan_sequences:
                    if step < len(seq):
                        series.append(seq[step])
                    else:
                        series.append(math.nan)
                color_value = cmap(step / max(horizon - 1, 1)) if horizon > 1 else cmap(0.5)
                plan_ax.plot(time, series, label=f"Step {step}", color=color_value)
            if horizon <= 5:
                plan_ax.legend(loc="upper right")
            plan_ax.set_ylabel("Planned V [V]")
            plan_ax.grid(True)
        else:
            plan_ax.text(
                0.5,
                0.5,
                "Planned path unavailable",
                transform=plan_ax.transAxes,
                ha="center",
                va="center",
                fontsize="small",
                color="#666666",
            )

    axes[-1].set_xlabel("Time [s]")

    return fig

