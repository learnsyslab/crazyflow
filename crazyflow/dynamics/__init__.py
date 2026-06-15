"""crazyflow.dynamics: quadrotor dynamics models for estimation, control, and simulation.

This package provides numeric and symbolic quadrotor dynamics models at multiple
fidelity levels. The models are pure functions compatible with any Array API backend
(NumPy, JAX, PyTorch, etc.) and with CasADi for symbolic computation.

Use [parametrize][crazyflow.dynamics.parametrize] to bind a dynamics function to a named drone
configuration, and [available_dynamics][crazyflow.dynamics.available_dynamics] to enumerate all
registered dynamics models.
"""

import os
import sys
from typing import Callable

# SciPy array API check. We use the most recent array API features, which require the
# SCIPY_ARRAY_API environment variable to be set to "1". This flag MUST be set before importing
# scipy, because scipy's C extensions cannot be unloaded once they have been imported. Therefore, we
# have to error out if the flag is not set. Otherwise, we immediately import scipy to ensure that no
# other package sets the flag to a different value before importing scipy.

if "scipy" in sys.modules and os.environ.get("SCIPY_ARRAY_API") != "1":
    msg = """scipy has already been imported and the 'SCIPY_ARRAY_API' environment variable has not
    been set. Please restart your Python session and set SCIPY_ARRAY_API="1" before importing any
    packages that depend on scipy, or import this package first to automatically set the flag."""
    raise RuntimeError(msg)

os.environ["SCIPY_ARRAY_API"] = "1"
import scipy  # noqa: F401, ensure scipy uses array API features

from crazyflow.dynamics.core import Dynamics, parametrize
from crazyflow.dynamics.first_principles import dynamics as _first_principles_dynamics
from crazyflow.dynamics.so_rpy import dynamics as _so_rpy_dynamics
from crazyflow.dynamics.so_rpy_rotor import dynamics as _so_rpy_rotor_dynamics
from crazyflow.dynamics.so_rpy_rotor_drag import dynamics as _so_rpy_rotor_drag_dynamics

__all__ = ["parametrize", "available_dynamics", "dynamics_features", "Dynamics"]


available_dynamics: dict[str, Callable] = {
    "first_principles": _first_principles_dynamics,
    "so_rpy": _so_rpy_dynamics,
    "so_rpy_rotor": _so_rpy_rotor_dynamics,
    "so_rpy_rotor_drag": _so_rpy_rotor_drag_dynamics,
}


def dynamics_features(dynamics: Callable) -> dict[str, bool]:
    """Return the feature flags declared by a dynamics function.

    Feature flags are set by the [supports][crazyflow.dynamics.core.supports] decorator on each
    dynamics function and describe which optional inputs the dynamics model accepts.

    Args:
        dynamics: A dynamics function, or a ``functools.partial`` wrapping one (as
            returned by [parametrize][crazyflow.dynamics.parametrize]).

    Returns:
        A dict of feature names to booleans. Currently contains:
            - ``"rotor_dynamics"``: ``True`` if the model accepts and integrates
              ``rotor_vel``, ``False`` if passing ``rotor_vel`` raises a
              ``ValueError``.

    Example:
        ```python
        from crazyflow.dynamics import dynamics_features
        from crazyflow.dynamics.first_principles import dynamics

        dynamics_features(dynamics)  # {'rotor_dynamics': True}
        ```
    """
    if hasattr(dynamics, "func"):  # Is a partial function
        return dynamics_features(dynamics.func)
    return getattr(dynamics, "__dynamics_features__")
