# Pipelines

Crazyflow has two pipelines, one for stepping and one for resetting, each an ordered collection of named, pure JAX functions that transform `SimData`. Both are constructed at `Sim` initialisation and compiled into a single `jax.jit`-cached function by `build_step_fn()` / `build_reset_fn()`. You can modify either pipeline through its named stages and recompile with the corresponding build function.

## The step pipeline

`sim.step_pipeline` contains four stages by default:

1. **Control functions** — convert the staged command through the control hierarchy (state → attitude → force/torque → rotor velocities, depending on the selected mode)
2. **Integrator** (`integration`) — advance the ODE one physics step (Euler, RK4, or symplectic Euler)
3. **Step counter** (`increment_steps`) — increment `data.core.steps`
4. **Floor clip** (`clip_floor_pos`) — prevent drones from passing through the floor

```python
from crazyflow.sim import Sim

sim = Sim()
print(sim.step_pipeline)
# Pipeline(step_attitude_controller -> step_force_torque_controller -> integration -> increment_steps -> clip_floor_pos)
```

## The reset pipeline

`sim.reset_pipeline` is empty by default. When `sim.reset()` is called, it first restores `SimData` to the default state, then runs every function in the reset pipeline in order. Each reset stage has the signature `(data: SimData, mask: Array | None) -> SimData`.

Populate `sim.reset_pipeline` to add episode-level randomization without modifying the default state.

## Modifying the step pipeline

Stages are addressed by name. Use `insert_before` / `insert_after` to place a function relative to an existing stage, `append` / `prepend` to add it at either end, `replace` to swap a stage's implementation, and `remove` to drop it. New stages are named after the function's `__name__` unless an explicit name is given; names must be unique within a pipeline.

```{ .python continuation }
from crazyflow.sim.data import SimData

def disturbance_fn(data: SimData) -> SimData:
    return data.replace(states=data.states.replace(vel=data.states.vel + 1e-5))

sim.step_pipeline.insert_before("integration", disturbance_fn)
sim.build_step_fn()  # recompile
```

!!! warning
    Always call `sim.build_step_fn()` after modifying `sim.step_pipeline`. Without it, `sim.step()` still runs the previously compiled kernel and silently ignores your changes.

To see how to modify the step pipeline with a stochastic disturbance, see the [Disturbance injection example](../examples/index.md#disturbance-injection).

## Modifying the reset pipeline

Add a function to the reset pipeline to vary initial conditions between episodes. The function receives the freshly-restored `data` and an optional `mask` of worlds that were reset.

```python
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

def randomize_initial_pos(data: SimData, mask: Array | None) -> SimData:
    key, subkey = jax.random.split(data.core.rng_key)
    noise = jax.random.normal(subkey, data.states.pos.shape) * 0.1  # ±10 cm
    return data.replace(
        states=data.states.replace(pos=data.states.pos + noise),
        core=data.core.replace(rng_key=key),
    )
    return data.replace(params=leaf_replace(data.params, mask, mass=mass))

sim = Sim(n_worlds=16)
sim.reset_pipeline += randomize_initial_pos
sim.build_reset_fn()  # recompile
sim.reset()
# Each of the 16 worlds now starts at a slightly different position
```

@jax.jit
def randomize_inertia(data: SimData, mask: Array | None = None) -> SimData:
    key, inertia_key = jax.random.split(data.core.rng_key)
    data = data.replace(core=data.core.replace(rng_key=key))  # Make sure to update the rng_key
    J = (
        data.params.J
        + jax.random.normal(inertia_key, (data.core.n_worlds, data.core.n_drones, 3, 3)) * 1e-8
    )
    return data.replace(params=leaf_replace(data.params, mask, J=J, J_inv=jnp.linalg.inv(J)))

```{ .python continuation }
def randomize_vel(data: SimData, mask: Array | None) -> SimData:
    key, subkey = jax.random.split(data.core.rng_key)
    noise = jax.random.normal(subkey, data.states.vel.shape) * 0.05
    return data.replace(
        states=data.states.replace(vel=data.states.vel + noise),
        core=data.core.replace(rng_key=key),
    )

def log_reset(data: SimData, mask: Array | None) -> SimData:
    return data  # a pure pass-through stage, e.g. a hook for metrics

for fn in (randomize_vel, log_reset):
    sim.reset_pipeline += fn
sim.build_reset_fn()

mask = np.array([True, False, False])  # Only randomize the first world
sim.reset(mask=mask)  # The mask is optional; omit it to reset and randomize all worlds
```

Reset stages run in tuple order, with each stage receiving the output of the previous one. The mask ensures that parameter updates apply only to worlds selected by `sim.reset()`.

## Removing a stage

Remove any stage by name. A common case is removing the floor clip when computing gradients through a trajectory that starts high above the ground:

```python
from crazyflow.sim import Sim

sim = Sim()
sim.step_pipeline.remove("clip_floor_pos")
sim.build_step_fn()
```

## Writing a custom stage

A step pipeline function must have the signature `(SimData) -> SimData`. A reset pipeline function must have the signature `(SimData, Array | None) -> SimData`. Both must be pure JAX functions with no Python-level side effects, so they can be traced and compiled.

```python
from crazyflow.sim.data import SimData

def my_step_stage(data: SimData) -> SimData:
    # JAX operations only — return updated data
    return data.replace(states=data.states.replace(pos=data.states.pos + 0.01))
```

## Next steps

- [Functional API](functional-api.md) — how `build_step_fn` fits into a compiled training loop
- [Examples](../examples/index.md) — disturbance injection and domain randomization scripts
