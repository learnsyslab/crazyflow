# Batching and domain randomization

All dynamics are built on Array API operations that broadcast over leading dimensions. There is no explicit batch size argument — just add a leading batch dimension to your state and command arrays and the dynamics evaluate all instances in a single call. This works identically across all backends.

```python
import jax.numpy as jnp
from crazyflow.dynamics import parametrize
from crazyflow.dynamics.first_principles import dynamics

dynamics = parametrize(dynamics, drone="cf2x_L250", xp=jnp)

N = 1_000
pos = jnp.zeros((N, 3))
quat = jnp.tile(jnp.array([0.0, 0.0, 0.0, 1.0]), (N, 1))
vel = jnp.zeros((N, 3))
ang_vel = jnp.zeros((N, 3))
cmd = jnp.full((N, 4), 15_000.0)
rotor_vel = jnp.full((N, 4), 15_000.0)

pos_dot, quat_dot, vel_dot, ang_vel_dot, rotor_vel_dot = dynamics(
    pos, quat, vel, ang_vel, cmd, rotor_vel=rotor_vel
)
vel_dot.shape  # (1000, 3)
```

## Higher-dimensional batches

Any number of leading dimensions works. A common pattern is a grid of environments, each containing multiple drones:

```python
import jax.numpy as jnp
from crazyflow.dynamics import parametrize
from crazyflow.dynamics.first_principles import dynamics

dynamics = parametrize(dynamics, drone="cf2x_L250", xp=jnp)
# 50 environments, 20 drones each
pos = jnp.zeros((50, 20, 3))
quat = jnp.broadcast_to(jnp.array([0.0, 0.0, 0.0, 1.0]), (50, 20, 4))
vel = jnp.zeros((50, 20, 3))
ang_vel = jnp.zeros((50, 20, 3))
rotor_vel = jnp.full((50, 20, 4), 12_000.0)
cmd = jnp.full((50, 20, 4), 15_000.0)

vel_dot = dynamics(pos, quat, vel, ang_vel, cmd, rotor_vel)[2]
vel_dot.shape  # (50, 20, 3)
```

## Domain randomization

Training policies across a distribution of physical parameters — domain randomization — improves sim-to-real transfer. Because `parametrize` returns a `functools.partial`, physical parameters are just keyword argument defaults. There are two ways to vary them across a batch.

**Option 1 — pass parameters as call-time kwargs.** This is the preferred pattern when using JIT compilation, because JAX traces the parameters as inputs rather than capturing them as constants. You can then draw fresh parameters each rollout without triggering a recompile.

```python
import jax
import jax.numpy as jnp
from crazyflow.dynamics import parametrize
from crazyflow.dynamics.first_principles import dynamics

N = 256
key = jax.random.PRNGKey(0)

pos, vel, ang_vel = jnp.zeros((N, 3)), jnp.zeros((N, 3)), jnp.zeros((N, 3))
quat = jnp.tile(jnp.array([0.0, 0.0, 0.0, 1.0]), (N, 1))
cmd = jnp.full((N, 4), 15_000.0)
rotor_vel = jnp.full((N, 4), 15_000.0)
dynamics = parametrize(dynamics, drone="cf2x_L250", xp=jnp)
nominal_mass = dynamics.keywords["mass"]
nominal_J = dynamics.keywords["J"]


@jax.jit
def step(pos, quat, vel, ang_vel, cmd, rotor_vel, mass, J, J_inv):
    return dynamics(pos, quat, vel, ang_vel, cmd, rotor_vel=rotor_vel, mass=mass, J=J, J_inv=J_inv)


key, k1, k2 = jax.random.split(key, 3)
mass_batch = nominal_mass * jax.random.uniform(k1, (N, 1), minval=0.9, maxval=1.1)
J_batch = nominal_J * jax.random.uniform(k2, (N, 3, 3), minval=0.9, maxval=1.1)
J_inv_batch = jnp.linalg.inv(J_batch)

vel_dot = step(pos, quat, vel, ang_vel, cmd, rotor_vel, mass_batch, J_batch, J_inv_batch)[2]
```

**Option 2 — mutate `dynamics.keywords` directly.** Simpler when you don't need JIT or are happy to retrace. Replace a scalar parameter with a `(N,)` array and each element in the batch uses its own value.

```{ .python continuation }
dynamics.keywords["mass"] = nominal_mass * mass_batch  # shape (N,)
vel_dot = dynamics(pos, quat, vel, ang_vel, cmd, rotor_vel)[2]
```

!!! note
    Matrix parameters like `J` have shape `(3, 3)`. To randomize per-drone, reshape to `(N, 3, 3)` and update `J_inv` accordingly. Scalar parameters like `mass` only need shape `(N,)`.

---

So far everything has been numeric. Many control frameworks — MPC, trajectory optimization, and state estimators — require symbolic dynamics representations. The next page covers the CasADi symbolic variants.
