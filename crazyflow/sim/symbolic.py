from __future__ import annotations

from typing import TYPE_CHECKING

from crazyflow.dynamics import Dynamics, parametrize
from crazyflow.dynamics.first_principles import (
    symbolic_dynamics as first_principles_symbolic_dynamics,
)
from crazyflow.dynamics.so_rpy import symbolic_dynamics as so_rpy_symbolic_dynamics
from crazyflow.sim.data import Control

if TYPE_CHECKING:
    import casadi as cs

    from crazyflow.sim import Sim


def symbolic_from_sim(
    sim: Sim, model_rotor_vel: bool = False, model_dist_f: bool = False, model_dist_t: bool = False
) -> tuple[cs.MX, cs.MX, cs.MX, cs.MX]:
    """Create a symbolic model from a simulation object.

    Args:
        sim: The simulation object.
        model_rotor_vel: Flag to model the rotor velocity.
        model_dist_f: Flag to model the distributed force.
        model_dist_t: Flag to model the distributed torque.

    Returns:
        The four symbolic expressions for X_dot, X, U, Y.
    """
    if sim.control != Control.attitude:
        raise ValueError("Symbolic dynamics only support attitude control")
    match sim.dynamics:
        case Dynamics.first_principles:
            return parametrize(first_principles_symbolic_dynamics, sim.drone)(
                model_rotor_vel=model_rotor_vel,
                model_dist_f=model_dist_f,
                model_dist_t=model_dist_t,
            )
        case Dynamics.so_rpy:
            return parametrize(so_rpy_symbolic_dynamics, sim.drone)(
                model_rotor_vel=model_rotor_vel,
                model_dist_f=model_dist_f,
                model_dist_t=model_dist_t,
            )
        case _:
            raise ValueError(f"Dynamics mode {sim.dynamics} not supported")
