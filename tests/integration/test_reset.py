from dataclasses import fields

import jax.numpy as jnp
import numpy as np
import pytest
from conftest import vectorize

import crazyflow  # noqa: F401, register gymnasium envs
from crazyflow.control import Control
from crazyflow.sim import Dynamics, Sim
from crazyflow.sim.data import SimData
from crazyflow.utils import CORE_NDIM_KEY


def vectorize_state(data: SimData) -> jnp.ndarray:
    """Stack the drone states into a (n_worlds, n_drones, 17) array."""
    s = data.states
    return vectorize(s.pos, s.quat, s.vel, s.ang_vel, s.rotor_vel)


def run(sim: Sim, cmds: np.ndarray) -> jnp.ndarray:
    """Apply the attitude commands one control step each and return the final states."""
    for cmd in cmds:
        sim.attitude_control(cmd)
        sim.step(sim.freq // sim.control_freq)
    return vectorize_state(sim.data)


@pytest.mark.integration
@pytest.mark.parametrize("dynamics", Dynamics)
def test_reset_during_simulation(dynamics: Dynamics):
    """Test reset behavior during an active simulation."""
    sim = Sim(dynamics=dynamics, control=Control.attitude)
    n_steps = 3
    random_cmds = np.random.rand(n_steps, 1, 1, 4)
    states_final = run(sim, random_cmds)

    sim.reset()
    assert jnp.all(sim.data.core.steps == 0)
    assert jnp.all(vectorize_state(sim.data) == vectorize_state(sim.default_data))

    # Verify simulation is identical when running again
    assert jnp.all(run(sim, random_cmds) == states_final)


@pytest.mark.integration
@pytest.mark.parametrize("dynamics", Dynamics)
def test_reset_multi_world(dynamics: Dynamics):
    """Test reset behavior with multiple worlds."""
    n_worlds, n_drones = 2, 2
    sim = Sim(n_worlds=n_worlds, n_drones=n_drones, dynamics=dynamics, control=Control.attitude)
    n_steps = 3
    random_cmds = np.random.rand(n_steps, n_worlds, n_drones, 4)
    states_final = run(sim, random_cmds)
    assert isinstance(sim.data.controls.attitude.staged_cmd, jnp.ndarray)
    assert isinstance(sim.data.controls.attitude.cmd, jnp.ndarray)

    sim.reset()
    assert jnp.all(sim.data.core.steps == 0)
    assert jnp.all(vectorize_state(sim.data) == vectorize_state(sim.default_data))

    # Verify simulation is identical when running again
    assert jnp.all(run(sim, random_cmds) == states_final)


@pytest.mark.integration
@pytest.mark.parametrize("dynamics", Dynamics)
def test_reset_masked_batched_params(dynamics: Dynamics):
    """Masked reset restores per-world parameter arrays only for the masked worlds."""
    n_worlds, n_drones, n_steps = 3, 2, 5
    sim = Sim(n_worlds, n_drones, dynamics=dynamics, control=Control.attitude)
    random_cmds = np.random.rand(n_steps, n_worlds, n_drones, 4)
    # Reference trajectory with the shared default parameters
    states_default = run(sim, random_cmds)
    sim.reset()

    # Give every drone its own copy of all parameters, scaled per world. World 0 keeps the default
    # values. Multiplying with a (n_worlds, n_drones, 1, ...) scale with one trailing axis per core
    # dimension broadcasts shared and already batched parameters alike to (n_worlds, n_drones, ...)
    scale = jnp.array([1.0, 1.1, 0.9])[:, None] * jnp.ones((n_worlds, n_drones))
    params = sim.data.params
    batched_params = {
        f.name: getattr(params, f.name)
        * scale.reshape(scale.shape + (1,) * f.metadata[CORE_NDIM_KEY])
        for f in fields(params)
    }
    batched_params["J_inv"] = jnp.linalg.inv(batched_params["J"])
    sim.data = sim.data.replace(params=params.replace(**batched_params))
    states_stepped = run(sim, random_cmds)
    # Batched copies of the defaults reproduce the default trajectory, scaled parameters change it
    assert jnp.allclose(states_stepped[0], states_default[0])
    assert not jnp.allclose(states_stepped[1], states_default[1])

    mask = np.array([False, True, False])
    sim.reset(mask=mask)
    # Only world 1 is restored to the default parameters, the other worlds keep theirs
    for name, value in batched_params.items():
        current, default = getattr(sim.data.params, name), getattr(sim.default_data.params, name)
        assert current.shape == value.shape, f"{name}: masked reset changed the shape"
        assert jnp.allclose(current[1], jnp.broadcast_to(default, value.shape)[1]), name
        assert jnp.allclose(current[~mask], value[~mask]), name
    # The same holds for the states
    assert jnp.all(vectorize_state(sim.data)[1] == vectorize_state(sim.default_data)[1])
    assert jnp.all(vectorize_state(sim.data)[~mask] == states_stepped[~mask])
    assert jnp.all(sim.data.core.steps[1] == 0)
    assert jnp.all(sim.data.core.steps[~mask] > 0)

    # Simulation keeps running with the mixed parameters
    assert jnp.all(jnp.isfinite(run(sim, random_cmds)))

    # A full reset restores the shared default parameters
    sim.reset()
    for name in batched_params:
        current, default = getattr(sim.data.params, name), getattr(sim.default_data.params, name)
        assert current.shape == default.shape, f"{name}: full reset did not restore the shape"
        assert jnp.all(current == default), name
