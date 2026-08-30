"""Example showing how to randomize parameters of the simulation.

All shown parameters can be randomized per world and per drone. All randomizations scale each
element of the default parameters by an independent uniform factor in [1 - x, 1 + x]. Using the
default parameters as the base value ensures that repeated resets do not compound.

Warning: The some randomizations shown change the shape of the parameters, which will trigger a
recompile of the simulation.
"""

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
def randomize_mass(data: SimData, default_data: SimData, mask: Array | None = None) -> SimData:
    key, mass_key = jax.random.split(data.core.rng_key)
    data = data.replace(core=data.core.replace(rng_key=key))  # Make sure to update the rng_key
    # The default mass is shared by all drones with shape (1,). Multiplying it with a
    # (n_worlds, n_drones, 1) scale gives every drone its own mass.
    shape = (data.core.n_worlds, data.core.n_drones, 1)
    amount = 0.2  # +-20%
    scale = jax.random.uniform(mass_key, shape, minval=1.0 - amount, maxval=1.0 + amount)
    mass = default_data.params.mass * scale
    return data.replace(params=leaf_replace(data.params, mask, mass=mass))


@jax.jit
def randomize_inertia(data: SimData, default_data: SimData, mask: Array | None = None) -> SimData:
    key, inertia_key = jax.random.split(data.core.rng_key)
    data = data.replace(core=data.core.replace(rng_key=key))  # Make sure to update the rng_key
    # The default inertia matrix is shared by all drones with shape (3, 3). Multiplying it with a
    # (n_worlds, n_drones, 3, 3) scale gives every drone its own inertia.
    shape = (data.core.n_worlds, data.core.n_drones, 3, 3)
    amount = 0.2  # +-20%
    scale = jax.random.uniform(inertia_key, shape, minval=1.0 - amount, maxval=1.0 + amount)
    J = default_data.params.J * scale
    return data.replace(params=leaf_replace(data.params, mask, J=J, J_inv=jnp.linalg.inv(J)))


@jax.jit
def randomize_thrust_curve(
    data: SimData, default_data: SimData, mask: Array | None = None
) -> SimData:
    key, thrust_key = jax.random.split(data.core.rng_key)
    data = data.replace(core=data.core.replace(rng_key=key))  # Make sure to update the rng_key
    # The default thrust curve coefficients are shared by all drones and motors with shape (1, 3).
    # Multiplying them with a (n_worlds, n_drones, 4, 3) scale gives every motor its own curve.
    shape = (data.core.n_worlds, data.core.n_drones, 4, 3)
    amount = 0.1  # +-10%
    scale = jax.random.uniform(thrust_key, shape, minval=1.0 - amount, maxval=1.0 + amount)
    rpm2thrust = default_data.params.rpm2thrust * scale
    return data.replace(params=leaf_replace(data.params, mask, rpm2thrust=rpm2thrust))


@jax.jit
def randomize_torque_curve(
    data: SimData, default_data: SimData, mask: Array | None = None
) -> SimData:
    key, torque_key = jax.random.split(data.core.rng_key)
    data = data.replace(core=data.core.replace(rng_key=key))  # Make sure to update the rng_key
    # The default torque curve coefficients are shared by all drones and motors with shape (1, 3).
    # Multiplying them with a (n_worlds, n_drones, 4, 3) scale gives every motor its own curve.
    shape = (data.core.n_worlds, data.core.n_drones, 4, 3)
    amount = 0.1  # +-10%
    scale = jax.random.uniform(torque_key, shape, minval=1.0 - amount, maxval=1.0 + amount)
    rpm2torque = default_data.params.rpm2torque * scale
    return data.replace(params=leaf_replace(data.params, mask, rpm2torque=rpm2torque))


@jax.jit
def randomize_rotor_dynamics(
    data: SimData, default_data: SimData, mask: Array | None = None
) -> SimData:
    key, rotor_key = jax.random.split(data.core.rng_key)
    data = data.replace(core=data.core.replace(rng_key=key))  # Make sure to update the rng_key
    # The default rotor dynamics coefficients are shared by all drones and motors with shape (1, 4).
    # Multiplying them with a (n_worlds, n_drones, 4, 4) scale gives every motor its own dynamics.
    shape = (data.core.n_worlds, data.core.n_drones, 4, 4)
    amount = 0.3  # +-30%
    scale = jax.random.uniform(rotor_key, shape, minval=1.0 - amount, maxval=1.0 + amount)
    rotor_dyn_coef = default_data.params.rotor_dyn_coef * scale
    return data.replace(params=leaf_replace(data.params, mask, rotor_dyn_coef=rotor_dyn_coef))


@jax.jit
def randomize_prop_inertia(
    data: SimData, default_data: SimData, mask: Array | None = None
) -> SimData:
    key, prop_key = jax.random.split(data.core.rng_key)
    data = data.replace(core=data.core.replace(rng_key=key))  # Make sure to update the rng_key
    # The default propeller inertia is shared by all drones and motors with shape (1,). Multiplying
    # it with a (n_worlds, n_drones, 4) scale gives every propeller its own inertia.
    shape = (data.core.n_worlds, data.core.n_drones, 4)
    amount = 0.2  # +-50%
    scale = jax.random.uniform(prop_key, shape, minval=1.0 - amount, maxval=1.0 + amount)
    prop_inertia = default_data.params.prop_inertia * scale
    return data.replace(params=leaf_replace(data.params, mask, prop_inertia=prop_inertia))


@jax.jit
def randomize_arm_length(
    data: SimData, default_data: SimData, mask: Array | None = None
) -> SimData:
    key, arm_key = jax.random.split(data.core.rng_key)
    data = data.replace(core=data.core.replace(rng_key=key))  # Make sure to update the rng_key
    # The default arm length is shared by all drones and motors with shape (1,). Multiplying it
    # with a (n_worlds, n_drones, 4) scale gives every motor its own arm length.
    shape = (data.core.n_worlds, data.core.n_drones, 4)
    amount = 0.01  # +-1%
    scale = jax.random.uniform(arm_key, shape, minval=1.0 - amount, maxval=1.0 + amount)
    L = default_data.params.L * scale
    return data.replace(params=leaf_replace(data.params, mask, L=L))


@jax.jit
def randomize_drag(data: SimData, default_data: SimData, mask: Array | None = None) -> SimData:
    key, drag_key = jax.random.split(data.core.rng_key)
    data = data.replace(core=data.core.replace(rng_key=key))  # Make sure to update the rng_key
    # The default drag matrix is shared by all drones with shape (3, 3). Multiplying it with a
    # (n_worlds, n_drones, 3, 3) scale gives every drone its own drag matrix.
    shape = (data.core.n_worlds, data.core.n_drones, 3, 3)
    amount = 0.3  # +-30%
    scale = jax.random.uniform(drag_key, shape, minval=1.0 - amount, maxval=1.0 + amount)
    drag_matrix = default_data.params.drag_matrix * scale
    return data.replace(params=leaf_replace(data.params, mask, drag_matrix=drag_matrix))


def main():
    sim = Sim(n_worlds=3, n_drones=4, control=Control.state)
    append_fn(sim.reset_pipeline, randomize_mass)
    append_fn(sim.reset_pipeline, randomize_inertia)
    append_fn(sim.reset_pipeline, randomize_thrust_curve)
    append_fn(sim.reset_pipeline, randomize_torque_curve)
    append_fn(sim.reset_pipeline, randomize_rotor_dynamics)
    append_fn(sim.reset_pipeline, randomize_prop_inertia)
    append_fn(sim.reset_pipeline, randomize_arm_length)
    append_fn(sim.reset_pipeline, randomize_drag)
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
