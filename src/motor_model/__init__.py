"""Motor modeling utilities."""

from ._mpc_common import MPCWeights
from .brushed_motor import (
    BrushedMotorModel,
    SimulationResult,
    rad_per_sec_per_volt_to_rpm_per_volt,
    rpm_per_volt_to_rad_per_sec_per_volt,
)
from .continuous_mpc_controller import ContMPCController
from .mpc_controller import LVDTMPCController, TubeMPCController
from .mpc_simulation import (
    MotorSimulation,
    SimulationHistory,
    SimulationState,
    build_default_controller_kwargs,
    build_default_continuous_controller_kwargs,
    build_default_tube_controller_kwargs,
    build_default_motor_kwargs,
)
from .plotting import plot_simulation

__all__ = [
    "BrushedMotorModel",
    "SimulationResult",
    "LVDTMPCController",
    "TubeMPCController",
    "ContMPCController",
    "MPCWeights",
    "MotorSimulation",
    "SimulationHistory",
    "SimulationState",
    "rad_per_sec_per_volt_to_rpm_per_volt",
    "rpm_per_volt_to_rad_per_sec_per_volt",
    "build_default_controller_kwargs",
    "build_default_continuous_controller_kwargs",
    "build_default_tube_controller_kwargs",
    "build_default_motor_kwargs",
    "plot_simulation",
]
