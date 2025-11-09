"""Motor modeling utilities."""

from .brushed_motor import (
    BrushedMotorModel,
    SimulationResult,
    rad_per_sec_per_volt_to_rpm_per_volt,
    rpm_per_volt_to_rad_per_sec_per_volt,
)
from .mpc_controller import LVDTMPCController, MPCWeights, TubeMPCController
from .mpc_simulation import (
    MotorSimulation,
    SimulationHistory,
    SimulationState,
    build_default_controller_kwargs,
    build_default_tube_controller_kwargs,
    build_default_motor_kwargs,
)

__all__ = [
    "BrushedMotorModel",
    "SimulationResult",
    "LVDTMPCController",
    "TubeMPCController",
    "MPCWeights",
    "MotorSimulation",
    "SimulationHistory",
    "SimulationState",
    "rad_per_sec_per_volt_to_rpm_per_volt",
    "rpm_per_volt_to_rad_per_sec_per_volt",
    "build_default_controller_kwargs",
    "build_default_tube_controller_kwargs",
    "build_default_motor_kwargs",
]
