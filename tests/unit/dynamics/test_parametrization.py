"""Tests of the parametrization of the dynamics."""

from __future__ import annotations

from typing import Callable

import pytest

from crazyflow.drones import available_drones
from crazyflow.dynamics import available_dynamics
from crazyflow.dynamics.core import parametrize


@pytest.mark.unit
@pytest.mark.parametrize("dynamics_name, dynamics", available_dynamics.items())
@pytest.mark.parametrize("drone", available_drones)
def test_dynamics_parametrization(dynamics_name: str, dynamics: Callable, drone: str):
    """TODO."""
    parametrize(dynamics, drone)
