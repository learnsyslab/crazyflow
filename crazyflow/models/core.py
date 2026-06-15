"""Core tools for registering and capability checking for the drone models."""

from __future__ import annotations

import inspect
import tomllib
import warnings
from functools import partial, wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, ParamSpec, TypeVar

import numpy as np

if TYPE_CHECKING:
    from types import ModuleType

    from crazyflow.models._typing import Array  # To be changed to array_api_typing later


F = TypeVar("F", bound=Callable[..., Any])
P = ParamSpec("P")
R = TypeVar("R")


def supports(rotor_dynamics: bool = True) -> Callable[[F], F]:
    """Decorator that declares which optional inputs a dynamics function supports.

    Wraps the decorated function so that:

    * If ``rotor_dynamics=False`` and the caller passes ``rotor_vel``, a
      ``ValueError`` is raised immediately.
    * If ``rotor_dynamics=True`` and the caller omits ``rotor_vel``, a
      ``UserWarning`` is issued and the commanded value is used directly.

    The decorator also attaches a ``__drone_model_features__`` attribute to the
    wrapper, which [model_features][crazyflow.models.model_features] reads.

    Args:
        rotor_dynamics: Whether the decorated function models rotor velocity
            dynamics. Set to ``False`` for models that do not accept or integrate
            ``rotor_vel`` (e.g. ``so_rpy``). Defaults to ``True``.

    Returns:
        A decorator that wraps the dynamics function with the capability checks
        described above.
    """

    def decorator(fn: F) -> F:
        @wraps(fn)
        def wrapper(
            pos: Array,
            quat: Array,
            vel: Array,
            ang_vel: Array,
            cmd: Array,
            rotor_vel: Array | None = None,
            *args: Any,
            **kwargs: Any,
        ) -> tuple[Array, Array, Array, Array, Array | None]:
            if not rotor_dynamics and rotor_vel is not None:
                raise ValueError("Rotor dynamics not supported, but rotor_vel is provided.")
            if rotor_dynamics and rotor_vel is None:
                warnings.warn("Rotor velocity not provided, using commanded rotor velocity.")
            return fn(pos, quat, vel, ang_vel, cmd, rotor_vel, *args, **kwargs)

        wrapper.__drone_model_features__ = {"rotor_dynamics": rotor_dynamics}

        return wrapper  # type: ignore

    return decorator


def parametrize(
    fn: Callable[P, R], drone_model: str, xp: ModuleType | None = None, device: str | None = None
) -> Callable[P, R]:
    """Parametrize a dynamics function with the default dynamics parameters for a drone model.

    Args:
        fn: The dynamics function to parametrize.
        drone_model: The drone model to use.
        xp: The array API module to use. If not provided, numpy is used.
        device: The device to use. If none, the device is inferred from the xp module.

    Example:
        ```{ .python notest }
        from crazyflow.models.core import parametrize
        from crazyflow.models.first_principles import dynamics

        dynamics_fn = parametrize(dynamics, drone_model="cf2x_L250")
        pos_dot, quat_dot, vel_dot, ang_vel_dot, rotor_vel_dot = dynamics_fn(
            pos=pos, quat=quat, vel=vel, ang_vel=ang_vel, cmd=cmd, rotor_vel=rotor_vel
        )
        ```

    Returns:
        The parametrized dynamics function with all keyword argument only parameters filled in.
    """
    try:
        xp = np if xp is None else xp
        physics = fn.__module__.split(".")[-2]
        sig = inspect.signature(fn)
        kwonly_params = [
            name
            for name, param in sig.parameters.items()
            if param.kind == inspect.Parameter.KEYWORD_ONLY
        ]
        params = load_params(physics, drone_model, xp=xp)
        params = {k: xp.asarray(v, device=device) for k, v in params.items() if k in kwonly_params}
    except KeyError as e:
        raise KeyError(
            f"Model `{physics}` does not exist in the parameter registry for drone `{drone_model}`"
        ) from e
    except ValueError as e:
        raise ValueError(f"Drone model `{drone_model}` not supported for `{physics}`") from e
    return partial(fn, **params)


def load_params(physics: str, drone_model: str, xp: ModuleType | None = None) -> dict:
    """Load and merge physical and model-specific parameters for a drone configuration.

    Reads parameters from two TOML files:

    * ``crazyflow/models/data/params.toml`` — physical parameters shared across all
      models (mass, inertia, thrust curves, …).
    * ``crazyflow/models/<physics>/params.toml`` — model-specific coefficients
      (e.g. fitted RPY coefficients for ``so_rpy``).

    The two dicts are merged (model-specific values take precedence), and
    ``J_inv`` is computed from ``J`` and added to the result.

    Args:
        physics: Name of the model sub-package, e.g. ``"first_principles"``,
            ``"so_rpy"``, ``"so_rpy_rotor"``, or ``"so_rpy_rotor_drag"``.
        drone_model: Name of the drone configuration, e.g. ``"cf2x_L250"``.
            Must exist as a section in both TOML files.
        xp: Array API module used to convert parameter values. If ``None``,
            NumPy is used.

    Returns:
        A flat dict mapping parameter names to arrays (or scalars) in the
        requested array namespace.  Always contains at least ``mass``, ``J``,
        ``J_inv``, ``gravity_vec``, and the model-specific coefficients for
        ``physics``.

    Raises:
        KeyError: If ``drone_model`` is not found in either TOML file, or if
            ``physics`` does not correspond to a known sub-package.
    """
    xp = np if xp is None else xp
    with open(Path(__file__).parent / "data/params.toml", "rb") as f:
        physical_params = tomllib.load(f)
    if drone_model not in physical_params:
        raise KeyError(f"Drone model `{drone_model}` not found in data/params.toml")
    with open(Path(__file__).parent / f"{physics}/params.toml", "rb") as f:
        model_params = tomllib.load(f)
    if drone_model not in model_params:
        raise KeyError(f"Drone model `{drone_model}` not found in model params.toml")
    params = physical_params[drone_model] | model_params[drone_model]
    # Make sure J_inv does not have a dtype fixed before conversion to xp arrays to avoid fixing it
    # to np.float64 when other frameworks might prefer a different dtype.
    params["J_inv"] = np.linalg.inv(params["J"]).tolist()
    params = {k: xp.asarray(v) for k, v in params.items()}  # if k in fields
    return params
