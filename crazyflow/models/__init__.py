"""drone-models: quadrotor dynamics models for estimation, control, and simulation.

This package provides numeric and symbolic quadrotor dynamics models at multiple
fidelity levels. Models are pure functions compatible with any Array API backend
(NumPy, JAX, PyTorch, etc.) and with CasADi for symbolic computation.

Use [parametrize][crazyflow.models.parametrize] to bind a dynamics function to a named drone
configuration, and [available_models][crazyflow.models.available_models] to enumerate all registered
models.
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

from crazyflow.models.core import parametrize
from crazyflow.models.first_principles import dynamics as _first_principles_dynamics
from crazyflow.models.so_rpy import dynamics as _so_rpy_dynamics
from crazyflow.models.so_rpy_rotor import dynamics as _so_rpy_rotor_dynamics
from crazyflow.models.so_rpy_rotor_drag import dynamics as _so_rpy_rotor_drag_dynamics

__all__ = ["parametrize", "available_models", "model_features"]


available_models: dict[str, Callable] = {
    "first_principles": _first_principles_dynamics,
    "so_rpy": _so_rpy_dynamics,
    "so_rpy_rotor": _so_rpy_rotor_dynamics,
    "so_rpy_rotor_drag": _so_rpy_rotor_drag_dynamics,
}


def model_features(model: Callable) -> dict[str, bool]:
    """Return the feature flags declared by a dynamics function.

    Feature flags are set by the [supports][crazyflow.models.core.supports] decorator on each
    dynamics function and describe which optional inputs the model accepts.

    Args:
        model: A dynamics function, or a ``functools.partial`` wrapping one (as
            returned by [parametrize][crazyflow.models.parametrize]).

    Returns:
        A dict of feature names to booleans. Currently contains:
            - ``"rotor_dynamics"``: ``True`` if the model accepts and integrates
              ``rotor_vel``, ``False`` if passing ``rotor_vel`` raises a
              ``ValueError``.

    Example:
        ```python
        from crazyflow.models import model_features
        from crazyflow.models.first_principles import dynamics

        model_features(dynamics)  # {'rotor_dynamics': True}
        ```
    """
    if hasattr(model, "func"):  # Is a partial function
        return model_features(model.func)
    return getattr(model, "__drone_model_features__")
