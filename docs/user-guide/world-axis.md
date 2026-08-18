# The world axis

The simulation is batched over worlds, so most arrays in `sim.data` carry a leading `n_worlds` axis. Resetting selected worlds and distributing worlds across devices both need to know which arrays those are. Here we explain how that is declared and what depends on it.

## Why it is declared, not inferred

A world axis is an ordinary axis, so no shape tells you whether it indexes worlds. With three worlds, `params.drag_matrix` of shape `(3, 3)` and `params.gravity_vec` of shape `(3,)` both look world-batched while neither is.

Each field therefore declares if it has a leading world axis with the `WORLD_INDEXED_KEY` metadata:

```python
from crazyflow.sim import Sim
from crazyflow.utils import world_mask

sim = Sim(n_worlds=3)
mask = world_mask(sim.data)

assert mask.states.pos  # (n_worlds, n_drones, 3), indexed by world
assert mask.params.mass  # (n_worlds, n_drones, 1), one mass per drone
assert not mask.params.gravity_vec  # (3,), shared by all worlds
assert not mask.params.drag_matrix  # (3, 3), shared by all worlds
```

`world_mask` returns a pytree of booleans matching the data, with one flag per array. Arrays that no field declares are shared by all worlds.

## Resets

`sim.reset()` restores the whole simulation, both the world-indexed and the shared arrays, with the rng key as the only exception.

`sim.reset(mask)` is the constrained form. It selects worlds along the world axis, so it can only restore arrays that have one. A shared array cannot be restored for some worlds and not others, so a masked reset leaves it untouched:

```{ .python continuation }
import jax.numpy as jnp

gravity = jnp.array([1.0, 2.0, 3.0])
sim.data = sim.data.replace(params=sim.data.params.replace(gravity_vec=gravity))
sim.reset(jnp.array([True, False, False]))

assert jnp.array_equal(sim.data.params.gravity_vec, gravity)  # Shared, so left alone
assert jnp.all(sim.data.states.pos[0] == 0.0)  # World 0 was reset
```

To keep a change to a shared parameter across masked resets, write it into `sim.default_data` as well by calling `sim.build_default_data()`. See [Stepping and resetting](oo-api.md#stepping-and-resetting) for the reset API.

## Sharding

Distributing the simulation partitions the world axis of the declared arrays and replicates the rest, using the same declarations. See [Sharding](sharding.md) for the mesh and placement API.

## Your own state

Anything you add to `sim.data.plugins` follows the same rule. A `dict` has no fields, so bare arrays are shared by default, which means they are neither reset nor distributed. If you need per-world states, you must declare it with a struct:

```python
import flax.struct
from jax import Array

from crazyflow.utils import WORLD_INDEXED_KEY


@flax.struct.dataclass
class PluginState:
    buffer: Array = flax.struct.field(metadata={WORLD_INDEXED_KEY: True})
    lookup: Array = None
```

If you store this under `sim.data.plugins`, `buffer` is restored for the masked worlds and partitioned when the simulation is sharded, while `lookup` is only restored by a full reset and replicated across devices.

## Next steps

- [Sharding](sharding.md) — distributing the worlds over several devices
- [Pipelines](pipelines.md) — customising what runs on a step and on a reset
