"""Analyse MPC performance under varying friction parameters.

This script sweeps combinations of Coulomb (kinetic) and static friction
coefficients for the :class:`~motor_model.BrushedMotorModel` while keeping the
MPC design fixed at the nominal friction values. It produces two SVG figures in
the ``figures`` directory:

* ``mpc_friction_responses.svg`` – position, speed, and torque trajectories for
  each Coulomb friction setting with multiple static friction levels overlaid.
* ``mpc_friction_metrics.svg`` – heat-map style summaries of steady-state error,
  settling time, and peak speed for the same scenarios.

Run the script directly (e.g. ``PYTHONPATH=src python examples/mpc_friction_analysis.py``)
to regenerate the figures.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

from motor_model import BrushedMotorModel, LVDTMPCController, SimulationResult

BASE_COULOMB_FRICTION = 2.2e-3
BASE_STATIC_FRICTION = 2.5e-3

COULOMB_SCALES = (0.5, 1.0, 1.5)
STATIC_SCALES = (0.5, 1.0, 1.5)

TARGET_LVDT = 0.2
SIM_DURATION = 0.5
SIM_DT = 0.001
CONTROLLER_DT = 0.01

STATIC_COLORS = {
    0.5: "#1f77b4",
    1.0: "#ff7f0e",
    1.5: "#2ca02c",
}


def build_motor(*, coulomb_friction: float, static_friction: float) -> BrushedMotorModel:
    """Return a deterministic motor instance for the given friction values."""

    return BrushedMotorModel(
        coulomb_friction=coulomb_friction,
        static_friction=static_friction,
        lvdt_noise_std=0.0,
    )


@dataclass
class ScenarioResult:
    """Container for the raw simulation result and derived metrics."""

    label: str
    coulomb_friction: float
    static_friction: float
    steady_state_error: float
    overshoot: float
    settling_time: float
    rise_time: float
    peak_speed: float
    peak_torque: float
    peak_voltage: float
    result: SimulationResult


def run_scenario(*, coulomb_scale: float, static_scale: float) -> ScenarioResult:
    coulomb_friction = BASE_COULOMB_FRICTION * coulomb_scale
    static_friction = BASE_STATIC_FRICTION * static_scale

    plant = build_motor(
        coulomb_friction=coulomb_friction,
        static_friction=static_friction,
    )
    controller = LVDTMPCController(
        build_motor(
            coulomb_friction=BASE_COULOMB_FRICTION,
            static_friction=BASE_STATIC_FRICTION,
        ),
        dt=CONTROLLER_DT,
        horizon=4,
        voltage_limit=8.0,
        target_lvdt=TARGET_LVDT,
        candidate_count=5,
        position_tolerance=0.01,
        static_friction_penalty=80.0,
        internal_substeps=10,
    )

    result = plant.simulate(
        controller,
        duration=SIM_DURATION,
        dt=SIM_DT,
        measurement_period=CONTROLLER_DT,
        controller_period=CONTROLLER_DT,
    )

    target_position = TARGET_LVDT * plant.lvdt_full_scale
    position = result.position
    time = result.time

    errors = [p - target_position for p in position]
    steady_state_error = errors[-1]
    overshoot = max(0.0, max(position) - target_position)

    tolerance_band = max(0.02 * abs(target_position), 1e-4)
    settling_time = math.nan
    for index, _ in enumerate(errors):
        if all(abs(e) <= tolerance_band for e in errors[index:]):
            settling_time = time[index]
            break

    rise_time = math.nan
    target_threshold = target_position * 0.9
    for index, pos in enumerate(position):
        if pos >= target_threshold:
            rise_time = time[index]
            break

    peak_speed = max(abs(v) for v in result.speed)
    peak_torque = max(abs(t) for t in result.torque)
    peak_voltage = max(abs(v) for v in result.voltage)

    label = f"C:{coulomb_scale:.1f}× S:{static_scale:.1f}×"

    return ScenarioResult(
        label=label,
        coulomb_friction=coulomb_friction,
        static_friction=static_friction,
        steady_state_error=steady_state_error,
        overshoot=overshoot,
        settling_time=settling_time,
        rise_time=rise_time,
        peak_speed=peak_speed,
        peak_torque=peak_torque,
        peak_voltage=peak_voltage,
        result=result,
    )


def collect_results() -> Dict[Tuple[float, float], ScenarioResult]:
    data: Dict[Tuple[float, float], ScenarioResult] = {}
    for coulomb_scale in COULOMB_SCALES:
        for static_scale in STATIC_SCALES:
            scenario = run_scenario(
                coulomb_scale=coulomb_scale, static_scale=static_scale
            )
            data[(coulomb_scale, static_scale)] = scenario
    return data


class SvgCanvas:
    """Utility to help build simple SVG drawings."""

    def __init__(self, width: float, height: float) -> None:
        self.width = width
        self.height = height
        self._elements: list[str] = []

    def add(self, element: str) -> None:
        self._elements.append(element)

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        stroke: str = "#000000",
        stroke_width: float = 1.0,
        dash: str | None = None,
    ) -> None:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" '
            f'y2="{y2:.2f}" stroke="{stroke}" stroke-width="{stroke_width}"{dash_attr}/>'
        )

    def rect(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        *,
        stroke: str = "#000000",
        stroke_width: float = 1.0,
        fill: str = "none",
    ) -> None:
        self.add(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" '
            f'stroke="{stroke}" stroke-width="{stroke_width}" fill="{fill}"/>'
        )

    def polyline(
        self,
        points: str,
        *,
        stroke: str,
        stroke_width: float = 1.5,
    ) -> None:
        self.add(
            f'<polyline points="{points}" fill="none" stroke="{stroke}" '
            f'stroke-width="{stroke_width}"/>'
        )

    def text(
        self,
        x: float,
        y: float,
        content: str,
        *,
        size: float = 14.0,
        anchor: str = "middle",
        weight: str | None = None,
    ) -> None:
        weight_attr = f' font-weight="{weight}"' if weight else ""
        self.add(
            f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" '
            f'text-anchor="{anchor}"{weight_attr}>{content}</text>'
        )

    def save(self, path: Path) -> None:
        with path.open("w", encoding="utf-8") as handle:
            handle.write("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n")
            handle.write(
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" '
                f'height="{self.height}" viewBox="0 0 {self.width} {self.height}">\n'
            )
            for element in self._elements:
                handle.write(f"  {element}\n")
            handle.write("</svg>\n")


def _series_range(
    scenarios: Iterable[ScenarioResult],
    extractor: callable[[ScenarioResult], Iterable[float]],
) -> tuple[float, float]:
    min_value = float("inf")
    max_value = float("-inf")
    for scenario in scenarios:
        values = list(extractor(scenario))
        if not values:
            continue
        min_value = min(min_value, min(values))
        max_value = max(max_value, max(values))
    if min_value == float("inf"):
        min_value, max_value = 0.0, 1.0
    if math.isclose(min_value, max_value, rel_tol=0.0, abs_tol=1e-12):
        padding = 0.1 if math.isclose(min_value, 0.0, abs_tol=1e-12) else abs(min_value) * 0.1
        min_value -= padding
        max_value += padding
    return min_value, max_value


def _format_float(value: float, fmt: str) -> str:
    return "n/a" if math.isnan(value) else format(value, fmt)


def _interpolate(value: float, start: float, end: float) -> float:
    if math.isclose(start, end):
        return 0.5
    return (value - start) / (end - start)


def _value_to_color(value: float, minimum: float, maximum: float) -> str:
    if math.isnan(value):
        return "#9e9e9e"
    ratio = max(0.0, min(1.0, _interpolate(value, minimum, maximum)))
    red = int(40 + ratio * 200)
    green = int(50 + (1.0 - ratio) * 160)
    blue = int(120 + (1.0 - ratio) * 80)
    return f"#{red:02x}{green:02x}{blue:02x}"


def _series_polyline(
    time: Iterable[float],
    values: Iterable[float],
    *,
    time_end: float,
    value_min: float,
    value_max: float,
    x: float,
    y: float,
    width: float,
    height: float,
) -> str:
    points = []
    for t, value in zip(time, values):
        x_pos = x + (t / time_end) * width if time_end > 0 else x
        if math.isclose(value_min, value_max):
            y_pos = y + height / 2.0
        else:
            ratio = (value - value_min) / (value_max - value_min)
            y_pos = y + height - ratio * height
        points.append(f"{x_pos:.2f},{y_pos:.2f}")
    return " ".join(points)


def plot_time_responses(
    data: Dict[Tuple[float, float], ScenarioResult],
    *,
    output_path: Path,
) -> None:
    width, height = 1200.0, 900.0
    outer_margin_x, outer_margin_y = 90.0, 110.0
    gap_x, gap_y = 60.0, 80.0
    columns = len(COULOMB_SCALES)
    rows = 3
    cell_width = (width - 2 * outer_margin_x - (columns - 1) * gap_x) / columns
    cell_height = (height - 2 * outer_margin_y - (rows - 1) * gap_y) / rows

    canvas = SvgCanvas(width, height)
    canvas.text(width / 2.0, 40.0, "MPC response for varying friction", size=24, weight="bold")

    scenarios = list(data.values())
    position_range = _series_range(scenarios, lambda s: s.result.position)
    speed_range = _series_range(scenarios, lambda s: s.result.speed)
    torque_range = _series_range(scenarios, lambda s: s.result.torque)

    target_position = TARGET_LVDT * BrushedMotorModel(lvdt_noise_std=0.0).lvdt_full_scale

    row_configs = [
        ("Position [m]", position_range, target_position),
        ("Speed [rad/s]", speed_range, 0.0),
        ("Torque [N·m]", torque_range, 0.0),
    ]

    for column_index, coulomb_scale in enumerate(COULOMB_SCALES):
        column_x = outer_margin_x + column_index * (cell_width + gap_x)
        column_results = [
            data[(coulomb_scale, static_scale)] for static_scale in STATIC_SCALES
        ]
        column_label = (
            f"Coulomb friction {column_results[0].coulomb_friction * 1e3:.2f} mN·m"
        )
        canvas.text(column_x + cell_width / 2.0, 70.0, column_label, size=16)

        for row_index, (title, value_range, reference) in enumerate(row_configs):
            row_y = outer_margin_y + row_index * (cell_height + gap_y)
            canvas.rect(column_x, row_y, cell_width, cell_height, stroke="#b0b0b0")

            if column_index == 0:
                canvas.text(column_x - 20.0, row_y - 15.0, title, size=14, anchor="end")

            canvas.line(column_x, row_y + cell_height, column_x + cell_width, row_y + cell_height, stroke="#808080")
            canvas.line(column_x, row_y, column_x, row_y + cell_height, stroke="#808080")

            value_min, value_max = value_range
            if row_index == 0:
                ref_value = reference
                ratio = (ref_value - value_min) / (value_max - value_min)
                y_ref = row_y + cell_height - ratio * cell_height
                canvas.line(
                    column_x,
                    y_ref,
                    column_x + cell_width,
                    y_ref,
                    stroke="#555555",
                    stroke_width=1.0,
                    dash="6 4",
                )
            elif row_index in (1, 2):
                if value_min < 0 < value_max:
                    zero_ratio = (0.0 - value_min) / (value_max - value_min)
                    y_zero = row_y + cell_height - zero_ratio * cell_height
                    canvas.line(
                        column_x,
                        y_zero,
                        column_x + cell_width,
                        y_zero,
                        stroke="#555555",
                        stroke_width=1.0,
                        dash="6 4",
                    )

            tick_count = 4
            for tick_index in range(tick_count):
                if math.isclose(value_max, value_min):
                    tick_value = value_min
                else:
                    tick_value = value_min + (value_max - value_min) * tick_index / (tick_count - 1)
                y_tick = row_y + cell_height - (
                    (tick_value - value_min) / (value_max - value_min)
                ) * cell_height if not math.isclose(value_max, value_min) else row_y + cell_height / 2.0
                canvas.line(column_x - 5, y_tick, column_x, y_tick, stroke="#808080")
                canvas.text(column_x - 10, y_tick + 4, f"{tick_value:.3f}", anchor="end", size=10)

            for static_scale in STATIC_SCALES:
                scenario = data[(coulomb_scale, static_scale)]
                result = scenario.result
                values = (
                    result.position
                    if row_index == 0
                    else result.speed if row_index == 1 else result.torque
                )
                points = _series_polyline(
                    result.time,
                    values,
                    time_end=SIM_DURATION,
                    value_min=value_min,
                    value_max=value_max,
                    x=column_x,
                    y=row_y,
                    width=cell_width,
                    height=cell_height,
                )
                canvas.polyline(points, stroke=STATIC_COLORS[static_scale], stroke_width=1.8)

            for tick_index in range(4):
                t = SIM_DURATION * tick_index / 3
                x_tick = column_x + (t / SIM_DURATION) * cell_width if SIM_DURATION > 0 else column_x
                canvas.line(x_tick, row_y + cell_height, x_tick, row_y + cell_height + 5, stroke="#808080")
                canvas.text(x_tick, row_y + cell_height + 20, f"{t:.2f}", size=10)

            if column_index == 0 and row_index == 0:
                legend_x = column_x + 10
                legend_y = row_y + 25
                for static_scale in STATIC_SCALES:
                    color = STATIC_COLORS[static_scale]
                    label = f"Static {BASE_STATIC_FRICTION * static_scale * 1e3:.2f} mN·m"
                    canvas.line(legend_x, legend_y - 5, legend_x + 20, legend_y - 5, stroke=color, stroke_width=2.0)
                    canvas.text(legend_x + 25, legend_y, label, anchor="start", size=12)
                    legend_y += 18

    canvas.save(output_path)


def plot_metric_heatmaps(
    data: Dict[Tuple[float, float], ScenarioResult],
    *,
    output_path: Path,
) -> None:
    width, height = 900.0, 360.0
    outer_margin_x, outer_margin_y = 80.0, 70.0
    gap_x = 60.0
    columns = len(COULOMB_SCALES)
    metrics = (
        ("steady_state_error", "Steady-state error [m]"),
        ("settling_time", "Settling time [s]"),
        ("peak_speed", "Peak speed [rad/s]"),
    )
    cell_width = (width - 2 * outer_margin_x - (len(metrics) - 1) * gap_x) / len(metrics)
    cell_height = (height - 2 * outer_margin_y) / len(STATIC_SCALES)

    canvas = SvgCanvas(width, height)
    canvas.text(width / 2.0, 40.0, "Performance metrics", size=22, weight="bold")

    for metric_index, (metric_key, title) in enumerate(metrics):
        grid = _metric_grid(data, metric_key)
        flattened = [value for row in grid for value in row if not math.isnan(value)]
        if flattened:
            min_val = min(flattened)
            max_val = max(flattened)
            if math.isclose(min_val, max_val):
                min_val -= 0.5
                max_val += 0.5
        else:
            min_val, max_val = 0.0, 1.0

        origin_x = outer_margin_x + metric_index * (cell_width + gap_x)
        canvas.text(origin_x + cell_width / 2.0, outer_margin_y - 20.0, title, size=14)

        for row_index, static_scale in enumerate(STATIC_SCALES):
            y = outer_margin_y + row_index * cell_height
            label = f"Static {static_scale:.1f}×"
            if metric_index == 0:
                canvas.text(origin_x - 10.0, y + cell_height / 2.0 + 5.0, label, anchor="end", size=12)

            for col_index, coulomb_scale in enumerate(COULOMB_SCALES):
                value = grid[row_index][col_index]
                x = origin_x + col_index * (cell_width / columns)
                rect_width = cell_width / columns - 8.0
                rect_height = cell_height - 8.0
                rect_x = x + 4.0
                rect_y = y + 4.0
                color = _value_to_color(value, min_val, max_val)
                canvas.rect(rect_x, rect_y, rect_width, rect_height, fill=color, stroke="#404040", stroke_width=0.8)
                canvas.text(
                    rect_x + rect_width / 2.0,
                    rect_y + rect_height / 2.0 + 5.0,
                    _format_float(value, ".3f"),
                    size=11,
                )

        if metric_index == 0:
            for col_index, coulomb_scale in enumerate(COULOMB_SCALES):
                x = origin_x + col_index * (cell_width / columns) + (cell_width / columns) / 2.0
                canvas.text(x, height - outer_margin_y + 25.0, f"Coulomb {coulomb_scale:.1f}×", size=12)

    canvas.save(output_path)


def _metric_grid(
    data: Dict[Tuple[float, float], ScenarioResult],
    metric: str,
) -> list[list[float]]:
    grid: list[list[float]] = []
    for static_scale in STATIC_SCALES:
        row = []
        for coulomb_scale in COULOMB_SCALES:
            value = getattr(data[(coulomb_scale, static_scale)], metric)
            row.append(value)
        grid.append(row)
    return grid


def main() -> None:
    output_dir = Path("figures")
    output_dir.mkdir(exist_ok=True)
    data = collect_results()

    responses_path = output_dir / "mpc_friction_responses.svg"
    metrics_path = output_dir / "mpc_friction_metrics.svg"

    plot_time_responses(data, output_path=responses_path)
    plot_metric_heatmaps(data, output_path=metrics_path)

    print("Generated:")
    for path in (responses_path, metrics_path):
        print(f"  {path}")

    print("\nScenario summary:")
    for coulomb_scale in COULOMB_SCALES:
        for static_scale in STATIC_SCALES:
            scenario = data[(coulomb_scale, static_scale)]
            print(
                "  "
                f"{scenario.label}: error={_format_float(scenario.steady_state_error, '.4f')} m, "
                f"settling={_format_float(scenario.settling_time, '.3f')} s, "
                f"peak speed={_format_float(scenario.peak_speed, '.2f')} rad/s, "
                f"peak voltage={_format_float(scenario.peak_voltage, '.2f')} V"
            )


if __name__ == "__main__":
    main()
