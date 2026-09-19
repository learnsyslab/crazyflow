from __future__ import annotations

import inspect
from typing import Any, Callable

import array_api_strict
import pytest

from crazyflow.control import load_fn_params, load_params, parametrize
from crazyflow.control.mellinger import (
    attitude2force_torque,
    body_rate2force_torque,
    force_torque2rotor_vel,
    state2attitude,
)
from crazyflow.drones import Drone

_MELLINGER_FNS = [
    state2attitude,
    attitude2force_torque,
    body_rate2force_torque,
    force_torque2rotor_vel,
]


@pytest.mark.unit
@pytest.mark.parametrize("fn", _MELLINGER_FNS, ids=lambda fn: fn.__name__)
@pytest.mark.parametrize("drone", Drone)
def test_load_fn_params_keys(fn: Callable[..., Any], drone: str) -> None:
    params = load_fn_params(fn, drone)
    fn_params = inspect.signature(fn).parameters
    fn_kwargs = {k for k, v in fn_params.items() if v.kind == inspect.Parameter.KEYWORD_ONLY}
    assert fn_kwargs <= set(params.keys()), f"Missing keys: {fn_kwargs - set(params.keys())}"


@pytest.mark.unit
def test_unknown_drone() -> None:
    with pytest.raises(KeyError, match="nonexistent_drone"):
        load_params("mellinger", "nonexistent_drone")
    with pytest.raises(KeyError, match="nonexistent_drone"):
        load_fn_params(state2attitude, "nonexistent_drone")
    with pytest.raises(KeyError, match="nonexistent_drone"):
        parametrize(state2attitude, "nonexistent_drone")


@pytest.mark.unit
def test_unknown_controller() -> None:
    with pytest.raises(KeyError, match="nonexistent_controller"):
        load_params("nonexistent_controller", "cf2x_L250")


@pytest.mark.unit
@pytest.mark.parametrize("drone", Drone)
def test_parametrize_xp_namespace(drone: str) -> None:
    controller = parametrize(state2attitude, drone, xp=array_api_strict)
    xp_array_type = type(array_api_strict.asarray(0.0))
    assert all(isinstance(v, xp_array_type) for v in controller.keywords.values())
