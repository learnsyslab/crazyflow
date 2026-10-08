"""Unit tests for simulation integrators."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from crazyflow.sim import Sim
from crazyflow.sim.integration import Integrator


@pytest.mark.unit
@pytest.mark.parametrize("integrator", Integrator)
def test_sim_step_integrators(integrator: Integrator, device: str):
    """Every integrator stays close to the default after two simulation steps."""
    sim = Sim(integrator=integrator, device=device)
    default_sim = Sim(device=device)
    try:
        sim.step(2)
        default_sim.step(2)

        assert jnp.all(sim.data.core.steps == 2)
        assert jnp.all(default_sim.data.core.steps == 2)

        data, structure = jax.tree.flatten_with_path(sim.data)
        default_data, default_structure = jax.tree.flatten(default_sim.data)
        assert structure == default_structure, "Simulation data structures must match"
        for (path, value), default_value in zip(data, default_data, strict=True):
            name = jax.tree_util.keystr(path)
            if isinstance(value, jnp.ndarray):
                assert type(value) is type(default_value), f"{name}: Type mismatch"
                assert value.shape == default_value.shape, f"{name}: Shape mismatch"
                assert value.dtype == default_value.dtype, f"{name}: Dtype mismatch"
                assert value.device == default_value.device, f"{name}: Device mismatch"
                if jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key):
                    np.testing.assert_array_equal(
                        jax.random.key_data(value), jax.random.key_data(default_value), err_msg=name
                    )
                else:
                    assert jnp.all(jnp.isfinite(value)), f"{name}: Non-finite integrator data"
                    assert jnp.all(jnp.isfinite(default_value)), f"{name}: Non-finite default data"
                    if jnp.issubdtype(value.dtype, jnp.inexact):
                        # At 500 Hz, position differences are O(g * dt**2), below 1e-4 m.
                        # Rotor speeds differ by about 1.4% between Euler and RK4 after two steps.
                        np.testing.assert_allclose(
                            value, default_value, rtol=2e-2, atol=1e-4, err_msg=name
                        )
                    else:
                        np.testing.assert_array_equal(value, default_value, err_msg=name)
            else:
                assert value == default_value, f"{name}: Value mismatch"
    finally:
        sim.close()
        default_sim.close()
