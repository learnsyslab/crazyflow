# Sharding

Crazyflow can be sharded across devices for even more performance. We partition the simulation data along the world axis and replicate shared data. `sim.shard` places `sim.data` and `sim.default_data` on multiple devices, and subsequent runs will use all devices to compute the simulation step.

!!! note
    Workloads that do not make use of sharding are unaffected by replication etc.

```python
import jax

from crazyflow.sim import Sim
from crazyflow.sim.sharding import world_mesh

devices = jax.devices()
sim = Sim(n_worlds=4 * len(devices))
sim.shard(world_mesh(devices))
sim.step()

assert sim.data.states.pos.sharding.spec == jax.sharding.PartitionSpec("worlds")
assert sim.data.params.gravity_vec.sharding.spec == jax.sharding.PartitionSpec()
```

The number of worlds has to be divisible by the number of devices. Whether an array is batched over worlds is determined from its shape and the core ndim its field declares, see [The world axis](world-axis.md). Data shared across worlds, such as the gravity vector above, is replicated on every device.

!!! warning
    Sharding relies on JAX's automatic sharding mode, which `world_mesh` requests for you. `jax.make_mesh` defaults to explicit sharding, under which the Mellinger controller fails inside SciPy's rotation backend.

## Functional API

`shard` places a `SimData` without going through the `Sim` object:

```{ .python continuation }
from crazyflow.sim.sharding import shard

mesh = world_mesh(devices)
data, default_data = shard(sim.data, mesh), shard(sim.default_data, mesh)
```

`placement` returns the shardings that `shard` applies, as a pytree mirroring the simulation data. Modify it to place the data by hand:

```{ .python continuation }
from crazyflow.sim.sharding import placement

data = jax.device_put(sim.data, placement(sim.data, mesh))
```

## Plugin data

Bare arrays in `sim.data.plugins` are replicated, since a dict has no fields to declare a core ndim on. Store per-world plugin state in a struct that declares one, as shown in [The world axis](world-axis.md#your-own-state).

## Next steps

- [The world axis](world-axis.md) — declaring which arrays are batched over worlds
- [Functional API](functional-api.md) — composing the pure step, reset and control functions
- [Batching & domain randomization](dynamics/batching.md) — varying the drone parameters per world
