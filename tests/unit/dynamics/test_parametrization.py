"""Tests of the parametrization of the dynamics."""

from __future__ import annotations

from typing import Callable

import pytest

from crazyflow.drones import available_drones
from crazyflow.dynamics import (
    Dynamics,
    available_dynamics,
    load_fn_params,
    load_params,
    parametrize,
)
from crazyflow.dynamics.so_rpy import dynamics as so_rpy


@pytest.mark.unit
@pytest.mark.parametrize("dynamics_name, dynamics", available_dynamics.items())
@pytest.mark.parametrize("drone", available_drones)
def test_dynamics_parameter_loading(dynamics_name: str, dynamics: Callable, drone: str) -> None:
    """Check that parameters can be loaded for all available dynamics and drones."""
    load_fn_params(dynamics, drone)


@pytest.mark.unit
@pytest.mark.parametrize("dynamics_name, dynamics", available_dynamics.items())
@pytest.mark.parametrize("drone", available_drones)
def test_model_parameter_loading(dynamics_name: str, dynamics: Callable, drone: str) -> None:
    """Check that all parameters of a model can be loaded for all drones."""
    params = load_params(dynamics_name, drone)
    assert "mass" in params and "gravity_vec" in params


@pytest.mark.unit
def test_unknown_drone() -> None:
    with pytest.raises(KeyError, match="nonexistent_drone"):
        load_params(Dynamics.so_rpy, "nonexistent_drone")
    with pytest.raises(KeyError, match="nonexistent_drone"):
        load_fn_params(so_rpy, "nonexistent_drone")
    with pytest.raises(KeyError, match="nonexistent_drone"):
        parametrize(so_rpy, "nonexistent_drone")


@pytest.mark.unit
def test_unknown_dynamics() -> None:
    with pytest.raises(ValueError, match="nonexistent_dynamics"):
        load_params("nonexistent_dynamics", "cf2x_L250")


@pytest.mark.unit
@pytest.mark.parametrize("dynamics_name, dynamics", available_dynamics.items())
@pytest.mark.parametrize("drone", available_drones)
def test_dynamics_parametrization(dynamics_name: str, dynamics: Callable, drone: str):
    """Check that we can parametrize all available dynamics with all drones."""
    parametrize(dynamics, drone)
