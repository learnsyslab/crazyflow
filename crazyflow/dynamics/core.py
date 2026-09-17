"""Core tools for registering and capability checking for the drone dynamics."""

from __future__ import annotations

import tomllib
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, ParamSpec, TypeVar

import numpy as np

from crazyflow.utils import filter_to_signature, to_xp
from crazyflow.utils import parametrize as _parametrize

if TYPE_CHECKING:
    from types import ModuleType

F = TypeVar("F", bound=Callable[..., Any])
P = ParamSpec("P")
R = TypeVar("R")


class Dynamics(StrEnum):
    """Dynamics mode for the simulation."""

    first_principles = "first_principles"
    so_rpy = "so_rpy"
    so_rpy_rotor = "so_rpy_rotor"
    so_rpy_rotor_drag = "so_rpy_rotor_drag"
    default = first_principles


def supports(rotor_dynamics: bool = True) -> Callable[[F], F]:
    """Decorator that declares which optional inputs a dynamics function supports.

    The decorator attaches a ``__dynamics_features__`` attribute to the wrapper, which
    [dynamics_features][crazyflow.dynamics.dynamics_features] reads.

    Args:
        rotor_dynamics: Whether the decorated function models rotor velocity dynamics. Set to
            ``False`` for models that do not accept or integrate ``rotor_vel`` (e.g. ``so_rpy``).
            Defaults to ``True``.

    Returns:
        The function decorated with capability flags.
    """

    def decorator(fn: F) -> F:
        fn.__dynamics_features__ = {"rotor_dynamics": rotor_dynamics}
        return fn

    return decorator


def parametrize(
    fn: Callable[P, R], drone: str, xp: ModuleType | None = None, device: str | None = None
) -> Callable[P, R]:
    """Parametrize a dynamics function with the default dynamics parameters for a drone.

    Args:
        fn: The dynamics function to parametrize.
        drone: The drone to use.
        xp: The array API module to use. If not provided, numpy is used.
        device: The device to use. If none, the device is inferred from the xp module.

    Example:
    ```python
    import numpy as np
    from crazyflow.dynamics.core import parametrize
    from crazyflow.dynamics.first_principles import dynamics

    dynamics_fn = parametrize(dynamics, drone="cf2x_L250")
    pos, quat = np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0])
    vel, ang_vel = np.zeros(3), np.zeros(3)
    rotor_vel, cmd = np.zeros(4), np.zeros(4)
    pos_dot, quat_dot, vel_dot, ang_vel_dot, rotor_vel_dot = dynamics_fn(
        pos=pos, quat=quat, vel=vel, ang_vel=ang_vel, cmd=cmd, rotor_vel=rotor_vel
    )
    ```

    Returns:
        The parametrized dynamics function with all keyword argument only parameters filled in.
    """
    return _parametrize(fn, drone, load_params, xp=xp, device=device)


def load_params(
    dynamics: Callable | Dynamics | str,
    drone: str,
    xp: ModuleType | None = None,
    device: str | None = None,
) -> dict:
    """Load and merge core and dynamics-specific parameters for a drone configuration.

    Reads parameters from two TOML files:

    * ``crazyflow/drones/params.toml`` — core physical parameters shared across all dynamics (mass,
      inertia, gravity, thrust limits).
    * ``crazyflow/dynamics/<dynamics>/params.toml`` — everything else the dynamics needs (e.g.
      thrust curves for ``first_principles``, fitted RPY coefficients for ``so_rpy``).

    The two dicts are merged (dynamics-specific values take precedence), and ``J_inv`` is computed
    from ``J`` and added to the result.

    Args:
        dynamics: A dynamics function, or a dynamics mode. For a function, the result only contains
            the parameters in its signature. For a mode, all parameters of both files are returned.
        drone: Name of the drone configuration, e.g. ``"cf2x_L250"``. Must exist as a section in
            both TOML files.
        xp: Array API module used to convert parameter values. If ``None``, NumPy is used.
        device: The device to use for the arrays. If ``None``, the device is inferred from the xp
            module.

    Returns:
        A flat dict mapping parameter names to arrays (or scalars) in the requested array namespace.
        Always contains at least ``mass``, ``J``, ``J_inv``, ``gravity_vec``, and the
        dynamics-specific coefficients for ``dynamics``.

    Raises:
        KeyError: If ``drone`` is not found in either TOML file, or if ``dynamics`` does not
            correspond to a known sub-package.
    """
    fn = dynamics if callable(dynamics) else None
    name = fn.__module__.split(".")[-2] if fn is not None else str(dynamics)
    if name not in tuple(Dynamics):
        raise KeyError(f"Dynamics `{name}` not found. Available dynamics: {tuple(Dynamics)}")
    with open(Path(__file__).parents[1] / "drones/params.toml", "rb") as f:
        core_params = tomllib.load(f)
    if drone not in core_params:
        raise KeyError(f"Drone `{drone}` not found in drones/params.toml")
    with open(Path(__file__).parent / f"{name}/params.toml", "rb") as f:
        dynamics_params = tomllib.load(f)
    if drone not in dynamics_params:
        raise KeyError(f"Drone `{drone}` not found in {name}/params.toml")
    params = core_params[drone] | dynamics_params[drone]
    # Make sure J_inv does not have a dtype fixed before conversion to xp arrays to avoid fixing it
    # to np.float64 when other frameworks might prefer a different dtype.
    params["J_inv"] = np.linalg.inv(params["J"]).tolist()
    if fn is not None:
        params = filter_to_signature(params, fn)
    return to_xp(params, xp=xp, device=device)
