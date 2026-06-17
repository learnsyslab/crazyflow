"""Core functionalities for controller parametrization."""

from __future__ import annotations

import inspect
import tomllib
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Callable, ParamSpec, TypeVar

import numpy as np

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
        pos = np.zeros(3)
        quat = np.array([0.0, 0.0, 0.0, 1.0])
        vel = np.zeros(3)
        cmd = np.zeros(13)
        rpyt, int_pos_err = ctrl(pos, quat, vel, cmd)
        ```

    Returns:
        The parametrized controller function with all keyword argument only parameters filled in.
    """
    try:
        params = load_fn_params(fn, drone, xp=xp, device=device)
    except KeyError as e:
        controller = fn.__module__.split(".")[-2]
        raise KeyError(
            f"Controller `{controller}.{fn.__name__}` not found for drone `{drone}`"
        ) from e
    return partial(fn, **params)


def load_params(controller: str, drone: str) -> dict[str, dict]:
    """Load the raw parameter table for a controller and drone.

    Reads ``crazyflow/control/<controller>/params.toml`` and returns the drone's table, nested by
    section (``"core"`` and one per controller function). Use [load_fn_params][load_fn_params] to
    select and convert the parameters a specific controller function accepts.

    Args:
        controller: Name of the controller sub-package, e.g. ``"mellinger"``.
        drone: Name of the drone configuration, e.g. ``"cf2x_L250"``.

    Returns:
        The raw, section-nested parameter dict for the drone.

    Raises:
        KeyError: If ``drone`` is not found in the params.toml file.
    """
    with open(Path(__file__).parent / f"{controller}/params.toml", "rb") as f:
        params = tomllib.load(f)
    if drone not in params:
        raise KeyError(f"Drone `{drone}` not found in {controller}/params.toml")
    return params[drone]


def load_fn_params(
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
    xp = np if xp is None else xp
    drone_params = load_params(fn.__module__.split(".")[-2], drone)
    merged = drone_params.get("core", {}) | drone_params.get(fn.__name__, {})
    accepted = set(inspect.signature(fn).parameters.keys())
    return {k: xp.asarray(v, device=device) for k, v in merged.items() if k in accepted}
