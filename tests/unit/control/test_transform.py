from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from crazyflow.control.core import load_params
from crazyflow.control.transform import (
    force2pwm,
    motor_force2rotor_vel,
    pwm2force,
    rotor_vel2body_force,
)


@pytest.fixture(scope="module")
def core_params() -> dict[str, Any]:
    return {k: np.asarray(v) for k, v in load_params("mellinger", "cf2x_L250")["core"].items()}


@pytest.mark.unit
def test_force2pwm_pwm2force_roundtrip(core_params: dict[str, Any]) -> None:
    thrust_max = float(core_params["thrust_max"])
    pwm_max = float(core_params["pwm_max"])
    forces = np.array([0.0, thrust_max * 0.25, thrust_max * 0.5, thrust_max])
    assert np.allclose(
        pwm2force(force2pwm(forces, thrust_max, pwm_max), thrust_max, pwm_max), forces
    )


@pytest.mark.unit
def test_force2pwm_boundary(core_params: dict[str, Any]) -> None:
    thrust_max = float(core_params["thrust_max"])
    pwm_max = float(core_params["pwm_max"])
    assert force2pwm(0.0, thrust_max, pwm_max) == pytest.approx(0.0)
    assert force2pwm(thrust_max, thrust_max, pwm_max) == pytest.approx(pwm_max)


@pytest.mark.unit
def test_motor_force2rotor_vel_shape(core_params: dict[str, Any]) -> None:
    rpm2thrust = core_params["rpm2thrust"]
    assert motor_force2rotor_vel(np.full(4, 0.05), rpm2thrust).shape == (4,)
    assert motor_force2rotor_vel(np.full((3, 2, 4), 0.05), rpm2thrust).shape == (3, 2, 4)


@pytest.mark.unit
def test_motor_force2rotor_vel_positive(core_params: dict[str, Any]) -> None:
    rpm2thrust = core_params["rpm2thrust"]
    forces = np.linspace(0.02, 0.12, 10)
    assert np.all(motor_force2rotor_vel(forces, rpm2thrust) > 0)


@pytest.mark.unit
def test_rotor_vel2body_force_shape(core_params: dict[str, Any]) -> None:
    rpm2thrust = core_params["rpm2thrust"]
    assert rotor_vel2body_force(np.full(4, 10_000.0), rpm2thrust).shape == (3,)
    assert rotor_vel2body_force(np.full((3, 2, 4), 10_000.0), rpm2thrust).shape == (3, 2, 3)


@pytest.mark.unit
def test_rotor_vel2body_force_z_axis_only(core_params: dict[str, Any]) -> None:
    rpm2thrust = core_params["rpm2thrust"]
    body_force = rotor_vel2body_force(np.full(4, 10_000.0), rpm2thrust)
    assert np.allclose(body_force[:2], 0.0)
    assert body_force[2] > 0.0
