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

`sim.reset_pipeline` is empty by default. When `sim.reset()` is called, it first restores `SimData` to the default state, then runs every function in the reset pipeline in order. The reset function signature is `(data: SimData, default_data: SimData, mask: Array | None) -> SimData`.

Populate `sim.reset_pipeline` to add episode-level randomization without modifying the default state.

## Modifying the step pipeline

Insert or remove stages by slicing and concatenating the tuple.

!!! warning
    Always call `sim.build_step_fn()` after modifying `sim.step_pipeline`. Without it, `sim.step()` still runs the previously compiled kernel and silently ignores your changes.

To see how to modify the step pipeline with a stochastic disturbance, see the [Disturbance injection example](../examples/index.md#disturbance-injection).

## Modifying the reset pipeline

Add a function to the reset pipeline to vary initial conditions between episodes. The function receives the freshly-restored `data`, the `default_data` it was restored from, and an optional `mask` of worlds that were reset.

```{ .python notest }
import jax
from crazyflow.sim import Sim
from crazyflow.sim.data import SimData
from jax import Array

def randomize_initial_pos(data: SimData, default_data: SimData, mask: Array | None) -> SimData:
    key, subkey = jax.random.split(data.core.rng_key)
    noise = jax.random.normal(subkey, data.states.pos.shape) * 0.1  # ±10 cm
    return data.replace(
        states=data.states.replace(pos=default_data.states.pos + noise),
        core=data.core.replace(rng_key=key),
    )

sim = Sim(n_worlds=16)
sim.reset_pipeline = (randomize_initial_pos,)
sim.build_reset_fn()  # recompile
sim.reset()
# Each of the 16 worlds now starts at a slightly different position
```

Multiple stages can be chained; the output of each function is passed as input to the next:

```{ .python notest }
sim.reset_pipeline = (randomize_initial_pos, randomize_mass_fn, log_reset_fn)
sim.build_reset_fn()
```

## Removing a stage

Remove any stage by excluding it from the tuple. A common case is removing the floor clip when computing gradients through a trajectory that starts high above the ground:

```{ .python notest }
from crazyflow.sim import Sim

sim = Sim()
sim.step_pipeline = sim.step_pipeline[:-1]  # drop clip_floor_pos
sim.build_step_fn()
```

## Writing a custom stage

A step pipeline function must have the signature `(SimData) -> SimData`. A reset pipeline function must have the signature `(SimData, SimData, Array | None) -> SimData`. Both must be pure JAX functions with no Python-level side effects, so they can be traced and compiled.

```{ .python notest }
from crazyflow.sim.data import SimData

def my_step_stage(data: SimData) -> SimData:
    # JAX operations only — return updated data
    return data.replace(...)
```

## Next steps

- [Functional API](functional-api.md) — how `build_step_fn` fits into a compiled training loop
- [Examples](../examples/index.md) — disturbance injection and domain randomization scripts
