"""Tests for the system identification residual functions."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from crazyflow.dynamics.utils.identification import _build_residuals_fun_translation


@pytest.fixture(autouse=True)
def _enable_x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


@pytest.mark.unit
@pytest.mark.parametrize(
    "dynamics, active",
    [
        ("so_rpy", [True, False, False, False]),
        ("so_rpy_rotor", [True, True, False, False]),
        ("so_rpy_rotor_drag", [True, True, True, True]),
    ],
)
def test_translation_jacobian(dynamics: str, active: list[bool]):
    n, mass = 50, 0.03
    t = np.linspace(0.0, 1.0, n)
    phi = 2 * np.pi * t
    quat = np.tile([0.0, 0.0, 0.0, 1.0], (n, 1))
    vel = np.stack((np.cos(phi), np.sin(phi), 0.5 * np.cos(2 * phi)), axis=-1)
    acc = np.stack((-np.sin(phi), np.cos(phi), -np.sin(2 * phi)), axis=-1)
    cmd_f = (1.0 + 0.3 * np.cos(phi)) * mass * 9.81
    constants = {"mass": mass, "gravity_vec": np.array([0, 0, -9.81])}
    args = (
        jnp.array(quat),
        jnp.array(vel),
        jnp.array(cmd_f),
        jnp.array(t),
        constants,
        jnp.array(acc),
    )

    residuals, jacobian = _build_residuals_fun_translation(dynamics)
    params = np.array([1.05, 0.3, -0.02, -0.03])
    jac = jacobian(params, *args)
    assert jac.shape == (n, 4)

    eps = 1e-6
    fd = np.zeros_like(jac)
    for i in range(4):
        step = np.zeros(4)
        step[i] = eps
        fd[:, i] = (residuals(params + step, *args) - residuals(params - step, *args)) / (2 * eps)

    active = np.array(active)
    assert np.all(jac[:, ~active] == 0.0)
    assert np.all(fd[:, ~active] == 0.0)
    np.testing.assert_allclose(jac[:, active], fd[:, active], rtol=1e-6, atol=1e-8)
