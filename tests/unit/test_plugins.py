"""Unit tests for the simulation plugin system.

Plugins are callables of the form ``fn(data: SimData) -> SimData`` that are inserted into
``sim.step_pipeline``. They can store arbitrary state in ``sim.data.plugins`` (a dict of JAX
arrays). Tests use a simple per-world step counter as a canonical plugin.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import pytest

from crazyflow.sim import Sim

if TYPE_CHECKING:
    from crazyflow.sim.data import SimData


def counter_plugin(data: SimData) -> SimData:
    """Increment plugins["counter"] by 1 on every simulation step."""
    return data.replace(plugins=data.plugins | {"counter": data.plugins["counter"] + 1})


def accumulater_plugin(data: SimData) -> SimData:
    """Accumulate plugins["counter"] on every simulation step."""
    return data.replace(
        plugins=data.plugins
        | {"accumulated": data.plugins["accumulated"] + data.plugins["counter"]}
    )


@pytest.mark.unit
def test_empty_by_default():
    """SimData.plugins is an empty dict when no plugins are registered."""
    sim = Sim()
    assert sim.data.plugins == {}, f"Expected empty plugins dict, got {sim.data.plugins}"
    sim.close()


@pytest.mark.unit
def test_builds():
    """Building patterns should not break with plugin data."""
    sim = Sim()
    sim.data = sim.data.replace(plugins={"sentinel": jnp.array([42])})
    sim.build_default_data()
    assert "sentinel" in sim.default_data.plugins, "Plugin data should be in default_data"
    sim.build_reset_fn()
    sim.build_step_fn()


@pytest.mark.unit
@pytest.mark.parametrize("n_worlds", [1, 3])
def test_plugin_data_changes(n_worlds: int):
    """Plugin data changes across simulation steps."""
    sim = Sim(n_worlds=n_worlds)
    sim.data = sim.data.replace(plugins={"counter": jnp.zeros((n_worlds, 1), dtype=jnp.int32)})
    sim.step_pipeline = sim.step_pipeline + (counter_plugin,)
    sim.build_step_fn()
    n_steps = 7
    sim.step(n_steps)
    assert jnp.all(sim.data.plugins["counter"] == n_steps), (
        f"Expected counter={n_steps}, got {sim.data.plugins['counter']}"
    )
    sim.close()


@pytest.mark.unit
def test_plugin_resets():
    """sim.reset() restores the counter to its default value (0)."""
    sim = Sim()
    sim.data = sim.data.replace(plugins={"counter": jnp.zeros((1, 1), dtype=jnp.int32)})
    sim.step_pipeline = sim.step_pipeline + (counter_plugin,)
    sim.build_default_data()
    sim.build_step_fn()
    sim.step(10)
    assert jnp.all(sim.data.plugins["counter"] == 10), "Precondition: counter should be 10"
    sim.reset()
    assert jnp.all(sim.data.plugins["counter"] == 0), "Counter should be 0 after full reset"
    sim.close()


@pytest.mark.unit
def test_plugin_masked_reset():
    """Masked reset only resets the plugin data for selected worlds."""
    n_worlds = 2
    sim = Sim(n_worlds=n_worlds)
    sim.data = sim.data.replace(plugins={"counter": jnp.zeros((n_worlds, 1), dtype=jnp.int32)})
    sim.step_pipeline = sim.step_pipeline + (counter_plugin,)
    sim.build_default_data()
    sim.build_step_fn()
    sim.step(10)
    assert jnp.all(sim.data.plugins["counter"] == 10), "Precondition: both counters should be 10"

    mask = jnp.array([True, False])  # reset world 0 only
    sim.reset(mask)
    assert jnp.all(sim.data.plugins["counter"][0] == 0), "World 0 counter must be reset to 0"
    assert jnp.all(sim.data.plugins["counter"][1] == 10), "World 1 counter must remain at 10"
    sim.close()


@pytest.mark.unit
def test_chained_plugins():
    """Two chained plugins produce different results depending on their order in the pipeline."""
    sim = Sim()
    sim.data = sim.data.replace(
        plugins={
            "counter": jnp.zeros((1, 1), dtype=jnp.int32),
            "accumulated": jnp.zeros((1, 1), dtype=jnp.int32),
        }
    )
    sim.step_pipeline = sim.step_pipeline + (counter_plugin, accumulater_plugin)
    sim.build_default_data()
    sim.build_step_fn()
    n_steps = 3
    sim.step(n_steps)
    assert int(sim.data.plugins["counter"][0, 0]) == n_steps, (
        f"Expected counter={n_steps}, got {sim.data.plugins['counter']}"
    )
    assert int(sim.data.plugins["accumulated"][0, 0]) == sum(range(1, n_steps + 1)), (
        f"Expected accumulated={sum(range(1, n_steps + 1))}, got {sim.data.plugins['accumulated']}"
    )
    sim.close()
