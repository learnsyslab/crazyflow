"""Tests of the parametrization of the dynamics."""

from __future__ import annotations

from typing import Callable

import pytest

from crazyflow.drones import available_drones
from crazyflow.dynamics import Dynamics, available_dynamics, load_params, parametrize


@pytest.mark.unit
@pytest.mark.parametrize("dynamics_name, dynamics", available_dynamics.items())
@pytest.mark.parametrize("drone", available_drones)
def test_dynamics_parameter_loading(dynamics_name: str, dynamics: Callable, drone: str) -> None:
    """Check that parameters can be loaded for all available dynamics and drones."""
    load_params(dynamics, drone)


@pytest.mark.unit
@pytest.mark.parametrize("dynamics_name, dynamics", available_dynamics.items())
@pytest.mark.parametrize("drone", available_drones)
def test_unfiltered_parameter_loading(dynamics_name: str, dynamics: Callable, drone: str) -> None:
    """Loading by mode returns a superset of the parameters loaded by function."""
    filtered, unfiltered = load_params(dynamics, drone), load_params(Dynamics(dynamics_name), drone)
    assert filtered.keys() <= unfiltered.keys()
    assert {"thrust_min", "thrust_max"} <= unfiltered.keys()


@pytest.mark.unit
@pytest.mark.parametrize("dynamics_name, dynamics", available_dynamics.items())
@pytest.mark.parametrize("drone", available_drones)
def test_dynamics_parametrization(dynamics_name: str, dynamics: Callable, drone: str):
    """Check that we can parametrize all available dynamics with all drones."""
    parametrize(dynamics, drone)
