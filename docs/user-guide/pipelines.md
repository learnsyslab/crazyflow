# Pipelines

Crazyflow has two pipelines, one for stepping and one for resetting, each a tuple of pure JAX functions that transform `SimData`. Both are constructed at `Sim` initialisation and compiled into a single `jax.jit`-cached function by `build_step_fn()` / `build_reset_fn()`. You can modify either pipeline by editing the tuple and calling the corresponding build function.

## The step pipeline

`sim.step_pipeline` contains four stages by default:

1. **Control functions** — convert the staged command through the control hierarchy (state → attitude → force/torque → rotor velocities, depending on the selected mode)
2. **Integrator** — advance the ODE one physics step (Euler, RK4, or symplectic Euler)
3. **Step counter** — increment `data.core.steps`
4. **Floor clip** — prevent drones from passing through the floor

```python
from crazyflow.sim import Sim

sim = Sim()
print(sim.step_pipeline)
# (<function ...>, <function rk4...>, <function increment_steps...>, <function clip_floor_pos...>)
```

## The reset pipeline

`sim.reset_pipeline` is empty by default. When `sim.reset()` is called, it first restores `SimData` to the default state, then runs every function in the reset pipeline in order. Each reset stage has the signature `(data: SimData, mask: Array | None) -> SimData`.

Populate `sim.reset_pipeline` to add episode-level randomization without modifying the default state.

## Modifying the step pipeline

Insert or remove stages by slicing and concatenating the tuple.

!!! warning
    Always call `sim.build_step_fn()` after modifying `sim.step_pipeline`. Without it, `sim.step()` still runs the previously compiled kernel and silently ignores your changes.

To see how to modify the step pipeline with a stochastic disturbance, see the [Disturbance injection example](../examples/index.md#disturbance-injection).

## Modifying the reset pipeline

The [domain randomization example](../examples/index.md#domain-randomization) defines two reset stages that randomize mass and inertia. Each function receives the freshly restored `data` and an optional `mask` of worlds that were reset:

```{ .python notest }
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from crazyflow.control import Control
from crazyflow.sim import Sim
from crazyflow.sim.data import SimData
from crazyflow.utils import leaf_replace


@jax.jit
def randomize_mass(data: SimData, mask: Array | None = None) -> SimData:
    key, mass_key = jax.random.split(data.core.rng_key)
    data = data.replace(core=data.core.replace(rng_key=key))  # Make sure to update the rng_key
    mass = (
        data.params.mass
        + jax.random.normal(mass_key, (data.core.n_worlds, data.core.n_drones, 1)) * 2e-3
    )
    return data.replace(params=leaf_replace(data.params, mask, mass=mass))


@jax.jit
def randomize_inertia(data: SimData, mask: Array | None = None) -> SimData:
    key, inertia_key = jax.random.split(data.core.rng_key)
    data = data.replace(core=data.core.replace(rng_key=key))  # Make sure to update the rng_key
    J = (
        data.params.J
        + jax.random.normal(inertia_key, (data.core.n_worlds, data.core.n_drones, 3, 3)) * 1e-8
    )
    return data.replace(params=leaf_replace(data.params, mask, J=J, J_inv=jnp.linalg.inv(J)))

sim = Sim(n_worlds=3, n_drones=4, control=Control.state)
sim.reset_pipeline = (randomize_mass, randomize_inertia)
sim.build_reset_fn()

mask = np.array([True, False, False])  # Only randomize the first world
sim.reset(mask=mask)  # The mask is optional; omit it to reset and randomize all worlds
```

Reset stages run in tuple order, with each stage receiving the output of the previous one. The mask ensures that parameter updates apply only to worlds selected by `sim.reset()`.

## Removing a stage

Remove any stage by excluding it from the tuple. A common case is removing the floor clip when computing gradients through a trajectory that starts high above the ground:

```{ .python notest }
from crazyflow.sim import Sim

sim = Sim()
sim.step_pipeline = sim.step_pipeline[:-1]  # drop clip_floor_pos
sim.build_step_fn()
```

## Writing a custom stage

A step pipeline function must have the signature `(SimData) -> SimData`. A reset pipeline function must have the signature `(SimData, Array | None) -> SimData`. Both must be pure JAX functions with no Python-level side effects, so they can be traced and compiled.

```{ .python notest }
from crazyflow.sim.data import SimData

def my_step_stage(data: SimData) -> SimData:
    # JAX operations only — return updated data
    return data.replace(...)
```

## Next steps

- [Functional API](functional-api.md) — how `build_step_fn` fits into a compiled training loop
- [Examples](../examples/index.md) — disturbance injection and domain randomization scripts
