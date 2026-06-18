from __future__ import annotations

import inspect
from typing import Any, Callable

import array_api_strict
import pytest

from crazyflow.control.core import load_params, parametrize
from crazyflow.control.mellinger import (
    attitude2force_torque,
    force_torque2rotor_vel,
    state2attitude,
)
from crazyflow.drones import available_drones

_MELLINGER_FNS = [state2attitude, attitude2force_torque, force_torque2rotor_vel]


@pytest.mark.unit
@pytest.mark.parametrize("fn", _MELLINGER_FNS, ids=lambda fn: fn.__name__)
@pytest.mark.parametrize("drone", available_drones)
def test_load_params_keys(fn: Callable[..., Any], drone: str) -> None:
    params = load_params(fn, drone)
    kwonly = {
        name
        for name, p in inspect.signature(fn).parameters.items()
        if p.kind == inspect.Parameter.KEYWORD_ONLY
    }
    assert kwonly <= set(params.keys()), f"Missing keys: {kwonly - set(params.keys())}"


@pytest.mark.unit
@pytest.mark.parametrize("drone", available_drones)
def test_load_params_values(drone: str) -> None:
    params = load_params(state2attitude, drone)
    assert float(params["mass"]) == pytest.approx(raw["core"]["mass"])


@pytest.mark.unit
def test_load_params_unknown_drone() -> None:
    with pytest.raises(KeyError, match="nonexistent_drone"):
        load_params(state2attitude, "nonexistent_drone")


@pytest.mark.unit
def test_parametrize_unknown_drone() -> None:
    with pytest.raises(KeyError):
        parametrize(state2attitude, "nonexistent_drone")


@pytest.mark.unit
@pytest.mark.parametrize("drone", available_drones)
def test_parametrize_xp_namespace(drone: str) -> None:
    controller = parametrize(state2attitude, drone, xp=array_api_strict)
    xp_array_type = type(array_api_strict.asarray(0.0))
    assert all(isinstance(v, xp_array_type) for v in controller.keywords.values())
