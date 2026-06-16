import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from crazyflow.control import Control
from crazyflow.sim import Sim
from crazyflow.sim.data import SimData
from crazyflow.sim.pipeline import append_fn
from crazyflow.utils import grid_2d, leaf_replace


@jax.jit
def randomize_mass(data: SimData, _: SimData, mask: Array | None = None) -> SimData:
    key, mass_key = jax.random.split(data.core.rng_key)
    data = data.replace(core=data.core.replace(rng_key=key))  # Make sure to update the rng_key
    dist = jax.random.normal(mass_key, (data.core.n_worlds, data.core.n_drones, 1)) * 2e-3
    mass = data.params.mass + dist
    return data.replace(params=leaf_replace(data.params, mask, mass=mass))


@jax.jit
def randomize_inertia(data: SimData, _: SimData, mask: Array | None = None) -> SimData:
    key, inertia_key = jax.random.split(data.core.rng_key)
    data = data.replace(core=data.core.replace(rng_key=key))  # Make sure to update the rng_key
    dist = jax.random.normal(inertia_key, (data.core.n_worlds, data.core.n_drones, 3, 3)) * 1e-8
    J = data.params.J + dist
    return data.replace(params=leaf_replace(data.params, mask, J=J, J_inv=jnp.linalg.inv(J)))


def main():
    sim = Sim(n_worlds=3, n_drones=4, control=Control.state)
    append_fn(sim.reset_pipeline, randomize_mass)
    append_fn(sim.reset_pipeline, randomize_inertia)
    sim.build_reset_fn()

    mask = np.array([True, False, False])  # Only randomize the first world
    duration = 5.0
    fps = 60

    for _ in range(3):
        cmd = np.zeros((sim.n_worlds, sim.n_drones, 13))
        cmd[..., 2] = 0.4
        cmd[..., :2] = grid_2d(sim.n_drones) * 0.25

        # After the first reset, each drone should behave slightly differently
        for i in range(int(duration * sim.control_freq)):
            sim.state_control(cmd)
            sim.step(sim.freq // sim.control_freq)
            if ((i * fps) % sim.control_freq) < fps:
                sim.render()

        # Note: The mask is optional.
        # We can also randomize all worlds at once by not passing anything
        sim.reset(mask=mask)  # Only reset the first world, the other two will stay the same

    sim.close()


if __name__ == "__main__":
    main()
