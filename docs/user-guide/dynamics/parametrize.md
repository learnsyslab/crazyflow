# Parametrize

Each dynamics function has a large number of keyword-only parameters — mass, inertia matrix, thrust curves, drag coefficients, and so on. Passing all of them at every call would be impractical. [`parametrize`][crazyflow.dynamics.parametrize] solves this by loading the parameters for a named drone configuration and returning a [`functools.partial`](https://docs.python.org/3/library/functools.html#functools.partial) with those parameters pre-filled. You then call the returned object with just the state and command arrays.

```python
from crazyflow.dynamics import parametrize
from crazyflow.dynamics.first_principles import dynamics

dynamics = parametrize(dynamics, drone="cf2x_L250")

# Inspect what was pre-filled
list(dynamics.keywords.keys())
# ['mass', 'L', 'prop_inertia', 'gravity_vec', 'J', 'J_inv',
#  'rpm2thrust', 'rpm2torque', 'mixing_matrix', 'rotor_dyn_coef', 'drag_matrix']
```

## Available drone configurations

The following configurations ship with pre-fitted parameters. They cover both the brushed Crazyflie 2.x series and the brushless Crazyflie 2.1:

```python
from crazyflow.drones import Drone

list(Drone)  # [Drone.cf21B_500, Drone.cf2x_L250, Drone.cf2x_P250, Drone.cf2x_T350, Drone.hb_x500]
```

| `drone` | Platform |
|---|---|
| `"cf2x_L250"` | Crazyflie 2.x |
| `"cf2x_P250"` | Crazyflie 2.x, plus propellers |
| `"cf2x_T350"` | Crazyflie 2.x, thrust upgrade kit |
| `"cf21B_500"` | Crazyflie 2.1 Brushless |
| `"hb_x500"` | Holybro X500 V2 |

If your drone is not listed, you can [add it](../adding-drones.md). The fitted models need coefficients identified from flight data with the [system identification pipeline](system-identification.md).

Not every dynamics is available for every drone. [`supported_dynamics`][crazyflow.dynamics.supported_dynamics] and [`supported_drones`][crazyflow.dynamics.supported_drones] list the available pairs:

```python
from crazyflow.dynamics import supported_drones, supported_dynamics

supported_dynamics("cf21B_500")  # (first_principles, so_rpy, so_rpy_rotor, so_rpy_rotor_drag)
supported_drones("so_rpy_rotor_drag")  # ('cf21B_500', 'cf2x_L250', 'cf2x_P250', 'cf2x_T350')
```

## Switching array backends

By default, `parametrize` stores parameters as NumPy arrays. For frameworks that would otherwise need to convert those arrays on every call — such as PyTorch, where NumPy arrays must become tensors — passing `xp` converts the parameters upfront. The backend of the outputs is always inferred from whatever arrays you pass in at call time.

<!-- notest: requires torch (not a dependency) -->
```{ .python notest }
import torch
import jax.numpy as jnp
from crazyflow.dynamics import parametrize
from crazyflow.dynamics.first_principles import dynamics

dynamics_torch = parametrize(dynamics, drone="cf2x_L250", xp=torch)
dynamics_jax = parametrize(dynamics, drone="cf2x_L250", xp=jnp)
```

You can also specify a compute device — for example, to move JAX parameters to GPU at construction time:

<!-- notest: requires a GPU -->
```{ .python notest }
import jax
import jax.numpy as jnp
dynamics_gpu = parametrize(
    dynamics, drone="cf2x_L250", xp=jnp, device=jax.devices("gpu")[0]
)
```

## Overriding parameters at call time

Because `parametrize` returns a `functools.partial`, the stored parameters are just keyword argument defaults. You can override any of them for a single call by passing a new value as a keyword argument — the call-time value takes precedence and the stored defaults are unchanged.

```python
import numpy as np
from crazyflow.dynamics import parametrize
from crazyflow.dynamics.first_principles import dynamics

dynamics = parametrize(dynamics, drone="cf2x_L250")

pos = np.zeros(3)
quat = np.array([0.0, 0.0, 0.0, 1.0])
vel = np.zeros(3)
ang_vel = np.zeros(3)
rotor_vel = np.zeros(4)
cmd = np.zeros(4)

# Simulate with a 10 g payload for this call only — dynamics.keywords is not modified.
pos_dot, *_ = dynamics(pos, quat, vel, ang_vel, cmd, rotor_vel, mass=0.0419)
```

This becomes particularly useful for domain randomization: instead of baking randomized parameters into the partial, you can pass a batch of them as call-time arguments and keep the step function JIT-compiled across parameter changes. See [Batching & domain randomization](batching.md) for the full pattern.

## Mutating stored parameters

You can also modify `dynamics.keywords` directly for changes that should persist across all future calls:

```python
import numpy as np
from crazyflow.dynamics import parametrize
from crazyflow.dynamics.first_principles import dynamics

dynamics = parametrize(dynamics, drone="cf2x_L250")
dynamics.keywords["mass"] = np.float64(0.040)  # heavier drone — applies to every call
```

!!! warning
    `dynamics.keywords` is a mutable dict shared across all references to the same partial. Modifying it affects every call. Call `parametrize` again for an independent copy.

## Selecting dynamics programmatically

`available_dynamics` is a dict mapping each [`Dynamics`][crazyflow.dynamics.Dynamics] mode to its unparametrized function. `Dynamics` is a string enum, so plain names work as keys too.

```python
from crazyflow.dynamics import Dynamics, available_dynamics, parametrize

list(available_dynamics)  # [Dynamics.first_principles, Dynamics.so_rpy, ...]

dynamics = available_dynamics[Dynamics.so_rpy_rotor_drag]
parametrized_dynamics = parametrize(dynamics, drone="cf2x_T350")
```

## Loading raw parameters

If you need the parameter values directly, for example, to pass them to [`symbolic_dynamics`](symbolic.md), use [`load_fn_params`][crazyflow.dynamics.load_fn_params] for exactly what a dynamics function accepts, or [`load_params`][crazyflow.dynamics.load_params] for everything a model defines for a drone:

```python { .python continuation }
from crazyflow.dynamics import Dynamics, load_fn_params, load_params

params = load_fn_params(dynamics, "cf2x_L250")
params["mass"]  # 0.0319
params["J_inv"]  # array([...])

params = load_params(Dynamics.first_principles, "cf2x_L250")
params["thrust_max"]  # 0.12, used by the simulator but not by the dynamics function
```

---

With a parametrized dynamics in hand, you can evaluate a single state. The next page covers running many drones simultaneously by adding batch dimensions — and how to randomize physical parameters across that batch for domain randomization.
