from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from crazyflow.dynamics import Dynamics
from crazyflow.sim import Sim
from crazyflow.sim.data import Control, SimData


@pytest.mark.unit
@pytest.mark.parametrize("dynamics", Dynamics)
def test_state_cmd_gradients(dynamics: Dynamics):
    sim = Sim(dynamics=dynamics, control=Control.state, freq=500)
    sim_step = sim._step
    # Hover clear of the floor and command a centimeter up to avoid clipping
    states = sim.data.states.replace(pos=sim.data.states.pos.at[..., 2].set(1.0))
    data = sim.data.replace(states=states)

    def height(cmd: Array, data: SimData) -> Array:
        data = data.replace(
            controls=data.controls.replace(state=data.controls.state.replace(staged_cmd=cmd))
        )
        return sim_step(data, sim.freq // sim.control_freq).states.pos[0, 0, 2]

    cmd = jnp.zeros((1, 1, 13), dtype=jnp.float32)
    cmd = cmd.at[..., 2].set(1.01)

    grad = jax.jit(jax.grad(height))(cmd, data)
    assert not jnp.any(jnp.isnan(grad))
    assert grad[0, 0, 2] > 0, "Commanding a higher setpoint must raise the drone"


@pytest.mark.unit
@pytest.mark.parametrize("dynamics", Dynamics)
def test_attitude_cmd_gradients(dynamics: Dynamics):
    sim = Sim(dynamics=dynamics, control=Control.attitude, freq=500)

    def step(cmd: Array, data: SimData) -> Array:
        data = data.replace(
            controls=data.controls.replace(attitude=data.controls.attitude.replace(staged_cmd=cmd))
        )
        data = sim._step(data, 10)
        return (data.states.pos[0, 0, 2] - 1.0) ** 2  # Quadratic cost to reach 1m height

    step_grad = jax.jit(jax.grad(step))

    cmd = jnp.zeros((1, 1, 4), dtype=jnp.float32)
    cmd = cmd.at[..., 3].set(0.3)

    grad = step_grad(cmd, sim.data)
    assert not jnp.any(jnp.isnan(grad))


@pytest.mark.unit
def test_force_torque_cmd_gradients():
    sim = Sim(dynamics=Dynamics.first_principles, control=Control.force_torque, freq=500)

    def step(cmd: Array, data: SimData) -> Array:
        data = data.replace(
            controls=data.controls.replace(
                force_torque=data.controls.force_torque.replace(staged_cmd=cmd)
            )
        )
        data = sim._step(data, 10)
        return (data.states.pos[0, 0, 2] - 1.0) ** 2  # Quadratic cost to reach 1m height

    step_grad = jax.jit(jax.grad(step))

    cmd = jnp.zeros((1, 1, 4), dtype=jnp.float32)
    cmd = cmd.at[..., 0].set(0.3)

    grad = step_grad(cmd, sim.data)
    assert not jnp.any(jnp.isnan(grad))
