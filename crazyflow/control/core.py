"""Core functionalities for controller parametrization."""

from __future__ import annotations

import tomllib
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Callable, ParamSpec, TypeVar

import jax
from jax import Array

from crazyflow.utils import filter_to_signature, to_xp
from crazyflow.utils import parametrize as _parametrize

if TYPE_CHECKING:
    from types import ModuleType

    from crazyflow._typing import Array  # To be changed to array_api_typing later

P = ParamSpec("P")
R = TypeVar("R")


def parametrize(
    fn: Callable[P, R], drone: str, xp: ModuleType | None = None, device: str | None = None
) -> Callable[P, R]:
    """Parametrize a controller function with the default controller parameters for a drone.

    Args:
        fn: The controller function to parametrize.
        drone: The drone to use.
        xp: The array API module to use. If not provided, numpy is used.
        device: The device to use. If None, the device is inferred from the xp module.

    Example:
    ```python
    import numpy as np
    from crazyflow.control import parametrize
    from crazyflow.control.mellinger import state2attitude

    ctrl = parametrize(state2attitude, "cf2x_L250")
    pos, quat = np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0])
    vel, cmd = np.zeros(3), np.zeros(13)
    rpyt, int_pos_err = ctrl(pos, quat, vel, cmd)
    ```

    Returns:
        The parametrized controller function with all keyword argument only parameters filled in.
    """
    return _parametrize(fn, drone, load_params, xp=xp, device=device)


def load_params(
    fn: Callable, drone: str, xp: ModuleType | None = None, device: str | None = None
) -> dict[str, Array]:
    """Load the parameters a specific controller function accepts.

    Merges the ``"core"`` section with the function's ``[drone.<fn_name>]`` section (function values
    take precedence), then keeps only the parameters in ``fn``'s signature.

    Args:
        fn: The controller function for which to load parameters.
        drone: Name of the drone configuration, e.g. ``"cf2x_L250"``.
        xp: The array API module to use. If not provided, numpy is used.
        device: The device to use. If None, the device is inferred from the xp module.

    Returns:
        A flat dict mapping parameter names to arrays in the requested array namespace.
    """
    assert isinstance(fn, Callable), f"Expected a function, got {type(fn)}"
    controller = fn.__module__.split(".")[-2]
    params_path = Path(__file__).parent / f"{controller}/params.toml"
    if not params_path.exists():
        raise KeyError(f"`{controller}` not found. Available controllers: {tuple(Control)}")
    with open(params_path, "rb") as f:
        params = tomllib.load(f)
    if drone not in params:
        raise KeyError(f"Drone `{drone}` not found in {controller}/params.toml")
    merged = params[drone].get("core", {}) | params[drone].get(fn.__name__, {})
    return to_xp(filter_to_signature(merged, fn), xp=xp, device=device)


class Control(str, Enum):
    """Control type of the simulated onboard controller."""

    state = "state"
    """State control takes [x, y, z, vx, vy, vz, ax, ay, az, yaw, roll_rate, pitch_rate, yaw_rate].
    
    Note:
        Recommended frequency is >=20 Hz.

    Warning:
        Currently, we only use positions, velocities, and yaw. The rest of the state is ignored.
        This is subject to change in the future.
    """
    attitude = "attitude"
    """Attitude control takes [roll, pitch, yaw, collective thrust].

    Note:
        Recommended frequency is >=100 Hz.
    """
    force_torque = "force_torque"
    """Force and torque control takes [fc, tx, ty, tz].

    Note:
        Recommended frequency is >=500 Hz.
    """
    rotor_vel = "rotor_vel"
    """Rotor velocity control takes [w1, w2, w3, w4] in RPMs.

    Note:
        Recommended frequency is >=500 Hz.
    """
    default = attitude


@jax.jit
def controllable(step: Array, freq: int, control_steps: Array, control_freq: int) -> Array:
    """Check which worlds can currently update their controllers.

    Args:
        step: The current step of the simulation.
        freq: The frequency of the simulation.
        control_steps: The steps at which the controllers were last updated.
        control_freq: The frequency of the controllers.

    Returns:
        A boolean mask of shape (n_worlds,) that is True at the worlds where the controllers can be
        updated.
    """
    return ((step - control_steps) >= (freq / control_freq)) | (control_steps == -1)
