"""Core tools for registering and capability checking for the drone dynamics."""

from __future__ import annotations

import inspect
import tomllib
from enum import Enum
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, ParamSpec, TypeVar

import numpy as np

if TYPE_CHECKING:
    from types import ModuleType


F = TypeVar("F", bound=Callable[..., Any])
P = ParamSpec("P")
R = TypeVar("R")


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
        ```{ .python notest }
        from crazyflow.dynamics.core import parametrize
        from crazyflow.dynamics.first_principles import dynamics

        dynamics_fn = parametrize(dynamics, drone="cf2x_L250")
        pos_dot, quat_dot, vel_dot, ang_vel_dot, rotor_vel_dot = dynamics_fn(
            pos=pos, quat=quat, vel=vel, ang_vel=ang_vel, cmd=cmd, rotor_vel=rotor_vel
        )
        ```

    Returns:
        The parametrized dynamics function with all keyword argument only parameters filled in.
    """
    try:
        xp = np if xp is None else xp
        dynamics = fn.__module__.split(".")[-2]
        sig = inspect.signature(fn)
        kwonly_params = [
            name
            for name, param in sig.parameters.items()
            if param.kind == inspect.Parameter.KEYWORD_ONLY
        ]
        params = load_params(dynamics, drone, xp=xp)
        params = {k: xp.asarray(v, device=device) for k, v in params.items() if k in kwonly_params}
    except KeyError as e:
        raise KeyError(
            f"Dynamics `{dynamics}` does not exist in the parameter registry for drone `{drone}`"
        ) from e
    except ValueError as e:
        raise ValueError(f"Drone `{drone}` not supported for dynamics `{dynamics}`") from e
    return partial(fn, **params)


def load_params(dynamics: str, drone: str, xp: ModuleType | None = None) -> dict:
    """Load and merge physical and dynamics-specific parameters for a drone configuration.

    Reads parameters from two TOML files:

    * ``crazyflow/drones/params.toml`` — physical parameters shared across all dynamics (mass,
      inertia, thrust curves, …).
    * ``crazyflow/dynamics/<dynamics>/params.toml`` — dynamics-specific coefficients (e.g. fitted
      RPY coefficients for ``so_rpy``).

    The two dicts are merged (dynamics-specific values take precedence), and ``J_inv`` is computed
    from ``J`` and added to the result.

    Args:
        dynamics: Name of the dynamics sub-package, e.g. ``"first_principles"``, ``"so_rpy"``,
            ``"so_rpy_rotor"``, or ``"so_rpy_rotor_drag"``.
        drone: Name of the drone configuration, e.g. ``"cf2x_L250"``. Must exist as a section in
            both TOML files.
        xp: Array API module used to convert parameter values. If ``None``, NumPy is used.

    Returns:
        A flat dict mapping parameter names to arrays (or scalars) in the requested array namespace.
        Always contains at least ``mass``, ``J``, ``J_inv``, ``gravity_vec``, and the
        dynamics-specific coefficients for ``dynamics``.

    Raises:
        KeyError: If ``drone`` is not found in either TOML file, or if ``dynamics`` does not
        correspond to a known sub-package.
    """
    xp = np if xp is None else xp
    with open(Path(__file__).parents[1] / "drones/params.toml", "rb") as f:
        physical_params = tomllib.load(f)
    if drone not in physical_params:
        raise KeyError(f"Drone `{drone}` not found in drones/params.toml")
    with open(Path(__file__).parent / f"{dynamics}/params.toml", "rb") as f:
        dynamics_params = tomllib.load(f)
    if drone not in dynamics_params:
        raise KeyError(f"Drone `{drone}` not found in {dynamics}/params.toml")
    params = physical_params[drone] | dynamics_params[drone]
    # Make sure J_inv does not have a dtype fixed before conversion to xp arrays to avoid fixing it
    # to np.float64 when other frameworks might prefer a different dtype.
    params["J_inv"] = np.linalg.inv(params["J"]).tolist()
    params = {k: xp.asarray(v) for k, v in params.items()}  # if k in fields
    return params


class Dynamics(str, Enum):
    """Dynamics mode for the simulation."""

    first_principles = "first_principles"
    so_rpy = "so_rpy"
    so_rpy_rotor = "so_rpy_rotor"
    so_rpy_rotor_drag = "so_rpy_rotor_drag"
    default = first_principles
