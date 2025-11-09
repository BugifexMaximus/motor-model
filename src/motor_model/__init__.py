"""Motor modeling utilities."""

from .brushed_motor import BrushedMotorModel, SimulationResult
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
    "build_default_controller_kwargs",
    "build_default_tube_controller_kwargs",
    "build_default_motor_kwargs",
]
